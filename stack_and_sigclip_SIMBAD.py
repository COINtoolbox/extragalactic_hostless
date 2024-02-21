import astropy.io.fits as fits
from astropy.stats import sigma_clip
from astropy.table import Table
# from astropy.time import Time
# import glob
import gzip
import io
import matplotlib.pylab as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.colors as colors
import numpy as np
import os
import pandas as pd
# from progressbar import progressbar
from tqdm import tqdm
import pyarrow.parquet as pq
from scipy import stats
import sys


# Declaring all auxiliary functions...


def make_three_plots(data_plot, name, labels, epoch="", pdf=""):

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    norm = colors.AsinhNorm(vmin=data_plot[0].min(), vmax=data_plot[0].max())
    for i, ax in enumerate(axes.flat):
        if i < 2:
            im = ax.imshow(data_plot[i], norm=norm)
        else:
            im = ax.imshow(data_plot[i])
        if i == 1:
            plt.colorbar(im, ax=axes.ravel().tolist())
        ax.set_title(f"{labels[i]}: {name} ({epoch})")
        ax.axvline(63/2., color='red', alpha=0.2)
        ax.axhline(63/2., color='red', alpha=0.2)
    if pdf != "":
        pdf.savefig(fig)
    # plt.show()
    plt.close()


def read_bytes_img(bytes_str):

    hdu_list = fits.open(gzip.open(io.BytesIO(bytes_str)))
    primary_hdu = hdu_list[0]
    return primary_hdu.data


def stack_epochs(data_epochs, obj_id, labels, pdf=""):
    stacked = [[], [], []]
    for i in range(len(data_epochs)):
        stacked[i] = np.median(data_epochs[i], axis=0)
        # print('Hi', type(stacked[i]), stacked[i].shape)
        # print(len(stacked[i][~np.isnan(stacked[i])]))

    return stacked


def resampling_alberto(input_data, bad_fmt, random_inside=[]):

    if len(random_inside) == 0:
        if bad_fmt is None:
            cleaned_data = input_data[np.where(input_data != None)]
        elif bad_fmt is np.nan:
            cleaned_data = input_data[~np.isnan(input_data)]
        else:
            raise Exception("Invalid bad format for resampling (None or np.nan)")
        output = np.copy(input_data).reshape(-1)

        data_1d = cleaned_data.reshape(-1)
        gkde_obj = stats.gaussian_kde(data_1d.astype(float), 0.1) # bw_method='scott')
        x_pts = np.linspace(0, data_1d.max(), 1000)
        estimated_pdf = gkde_obj.evaluate(x_pts)
        random_inside = gkde_obj.resample(5000)[0]
        # plt.figure()
        # plt.hist(data_1d, bins=100, density=True)  # bins='scott'
        # plt.hist(random_inside, bins=100, density=True)
        # plt.plot(x_pts, estimated_pdf, label="kde estimated PDF", color="r")
        # plt.xlim(0,1000)
        # plt.show()

        # xy_min = [0, 0]
        # xy_max = [data_1d.max(), 1.05*estimated_pdf.max()]
        # random_xy = np.random.uniform(low=xy_min, high=xy_max, size=(20000,2))
        # random_pdf = gkde_obj.evaluate(random_xy[:,0])
        # inside_pdf = random_xy[:,1] < random_pdf
        # # plt.scatter(random_xy[:,0][inside_pdf], random_xy[:,1][inside_pdf], s=1,color='red')
        # # plt.scatter(random_xy[:,0][~inside_pdf], random_xy[:,1][~inside_pdf], s=1,color='blue')
        # # plt.plot(x_pts, estimated_pdf, label="kde estimated PDF", color="r")
        # # plt.show()
        # random_inside = random_xy[:,0][inside_pdf]

    if bad_fmt is None:
        num_invalids = len(input_data[np.where(input_data == None)])
        output[np.where(output == None)] = np.random.choice(random_inside, size=num_invalids)
    elif bad_fmt is np.nan:
        num_invalids = len(input_data[np.isnan(input_data)])
        output[np.isnan(output)] = np.random.choice(random_inside, size=num_invalids)

    return output.reshape((63, 63))


def replace_clipped(clipped_sci, clipped_temp, invalid):

    # TRY TO FIX IT TOMORROW (4-5 TIMES FASTER)
    final_sci, final_temp = np.copy(clipped_sci).reshape(-1), np.copy(clipped_temp).reshape(-1)
    num_invalids = len(clipped_sci[np.isnan(clipped_sci)])
    dist_sci = np.random.normal(np.median(clipped_sci[~np.isnan(clipped_sci)]),
                                np.std(clipped_sci[~np.isnan(clipped_sci)]), num_invalids)
    final_sci[np.isnan(final_sci)] = dist_sci
    num_invalids = len(clipped_temp[np.isnan(clipped_temp)])
    dist_temp = np.random.normal(np.median(clipped_temp[~np.isnan(clipped_temp)]),
                                 np.std(clipped_temp[~np.isnan(clipped_temp)]), num_invalids)
    final_temp[np.isnan(final_temp)] = dist_temp
    # print(final_sci.shape, final_temp.shape)

    return final_sci.reshape((63, 63)), final_temp.reshape((63, 63))


def plot_clipped_and_resampled(clipped, final, obj_id, lab, pdf=""):
    fig, axes = plt.subplots(1,2, figsize=(10,4))
    im = axes[0].imshow(clipped)
    axes[0].set_title(f"{lab}: {obj_id} (sig-clip)")
    axes[1].imshow(final)
    axes[1].set_title(f"{lab}: {obj_id} (resampled)")
    fig.colorbar(im, ax=axes.ravel().tolist())
    if pdf != "":
        pdf.savefig(fig)
    # plt.show()
    plt.close()


def declare_Table():

    columns_63_63 = [[np.ones((63, 63))]]*7
    nms = ['stacked_science', 'stacked_template', 'stacked_difference',
           'clipped_science', 'clipped_template', 'resampled_science',
           'resampled_template']
    final_table = Table(columns_63_63, names=nms)
    final_table.add_column(['example'], name='obj_id', index=0)
    # final_table.add_column([[0., 0.]], name='mjds', index=1)
    final_table.add_column([1], name='n_epochs', index=1)
    final_table.remove_row(0)

    return final_table


# path = "/Volumes/LaCie/postdoc-Bkpis/coin-crp7/data/SIMBAD/"
path = "/media3/CRP7/hosts/data/SIMBAD/"
data_path = path + "Nov2019Sep2023_with_candidates/all_data_not_in_mangrove/"
data_list = sorted(os.listdir(data_path))
if '_SUCCESS' in data_list:
    data_list = data_list[1:]
labels = ['Science', 'Template', 'Difference']

if len(sys.argv) == 3:
    pq_idx_start, pq_idx_end = int(sys.argv[1]), int(sys.argv[2])
else:
    raise ValueError("Usage: python3 stack_and_sigclip_SIMBAD.py n_start n_end." + \
                     " The two last values correspond to which parquet files you want to use.")

#
# *** THIS BLOCK IS WORKING SLOW BUT WELL, I WILL TRY ANOTHER APPROACH *** #
#
# Let's try to reproduce this object list, as a double check
print('\n\033[1m - Start gathering data in dictionaries...\n\033[0m')
# dict_mjds, dict_sci_img, dict_temp_img, dict_diff_img = {}, {}, {}, {}
final_table = declare_Table()
for pq_file in data_list[pq_idx_start:pq_idx_end]:
    pq_data = pq.read_table(data_path + pq_file).to_pandas()
    n_exceptions = 0
    for idx in tqdm(range(len(pq_data))):
        event = pq_data.iloc[idx]
        obj_id, mjds = event['objectId'], event['i:jd']
        lst_sci_img, lst_temp_img, lst_diff_img = [], [], []
        for epoch in range(len(mjds)):
            sci_img = read_bytes_img(event['b:cutoutScience_stampData'][epoch])
            temp_img = read_bytes_img(event['b:cutoutTemplate_stampData'][epoch])
            diff_img = read_bytes_img(event['b:cutoutDifference_stampData'][epoch])
            try:
                sci_img = resampling_alberto(sci_img, np.nan)
                temp_img = resampling_alberto(temp_img, np.nan)
                diff_img = resampling_alberto(diff_img, np.nan)
            except Exception:
                n_exceptions += 1  # this number is wrong -- epochs...
                # print(obj_id, 'is an exception')
                continue
            # print(obj_id, mjd[epoch], sci_img.shape, temp_img.shape, diff_img.shape)

            sci_img = sci_img.astype(float)
            temp_img = temp_img.astype(float)
            diff_img = diff_img.astype(float)
            # if len(sci_img[~np.isnan(sci_img)]) != 0:
            #     if obj_id not in dict_mjds.keys():
            #         dict_mjds[obj_id] = [mjds[epoch]]
            #         dict_sci_img[obj_id] = [sci_img]
            #         dict_temp_img[obj_id] = [temp_img]
            #         dict_diff_img[obj_id] = [diff_img]
            #     else:
            #         dict_mjds[obj_id].append(mjds[epoch])
            #         dict_sci_img[obj_id].append(sci_img)
            #         dict_temp_img[obj_id].append(temp_img)
            #         dict_diff_img[obj_id].append(diff_img)

            lst_sci_img.append(sci_img)
            lst_temp_img.append(temp_img)
            lst_diff_img.append(diff_img)

        # stack here...
        # pdf = PdfPages(f'{path}/images_stack_sigclip/{obj_id}_plots.pdf')
        pdf = ""
        exception_ids = []
        data_epochs = [lst_sci_img, lst_temp_img, lst_diff_img]
        try:
            stacked = stack_epochs(data_epochs, obj_id, labels, pdf)
            stacked[0] = resampling_alberto(stacked[0], np.nan)
            stacked[1] = resampling_alberto(stacked[1], np.nan)
            stacked[2] = resampling_alberto(stacked[2], np.nan)
        except Exception:
            # print(f'Object {obj_id} skipped, it has only strange values...')
            exception_ids.append(obj_id)
            # pdf.close()
            continue
        make_three_plots(stacked, obj_id, labels, "median", pdf=pdf)
        clipped_sci = sigma_clip(stacked[0], sigma=3, maxiters=10)  # 3 or 5
        clipped_temp = sigma_clip(stacked[1], sigma=3, maxiters=10)  # 3 or 5
        clipped_sci, clipped_temp = clipped_sci.filled(np.nan), clipped_temp.filled(np.nan)
        # final_sci, final_temp = replace_clipped(clipped_sci, clipped_temp, np.ma.masked)
        final_sci = resampling_alberto(clipped_sci, np.nan)
        final_temp = resampling_alberto(clipped_temp, np.nan)
        plot_clipped_and_resampled(clipped_sci, final_sci, obj_id, "Science", pdf)
        plot_clipped_and_resampled(clipped_temp, final_temp, obj_id, "Template", pdf)
        # pdf.close()
        # to_table = [obj_id, mjds, len(data_epochs[0]), stacked[0], stacked[1], stacked[2],
        #             clipped_sci, clipped_temp, final_sci, final_temp]
        to_table = [obj_id, len(data_epochs[0]), stacked[0], stacked[1], stacked[2],
                    clipped_sci, clipped_temp, final_sci, final_temp]
        final_table.add_row(to_table)

    # print(pq_file.split('_')[-1].split('.')[0], end=' ')
    # number_of_imgs = sum(len(dict_mjds[key]) for key in dict_mjds.keys())
    # print(len(pq_data)-n_exceptions, len(dict_mjds.keys()))

# *** NEW BLOCK ::: PRODUCE FILE WITH THE FILENAMES OF EACH EVENT *** #
# dict_ids_files = {}
# n_tot, n_exceptions = 0, 0
# for pq_file in sorted(os.listdir(data_path))[:7]:
#     pq_data = pq.read_table(data_path + pq_file).to_pandas()
#     file_id = [pq_file.split('-')[1]]
#     n_tot += len(pq_data)
#     for idx in tqdm(range(len(pq_data))):
#         event = pq_data.iloc[idx]
#         obj_id, mjd = event['objectId'], event['i:jd']
#         sci_img = read_bytes_img(event['b:cutoutScience_stampData'][0])
#         temp_img = read_bytes_img(event['b:cutoutTemplate_stampData'][0])
#         diff_img = read_bytes_img(event['b:cutoutDifference_stampData'][0])

#         try:
#             sci_img = resampling_alberto(sci_img, np.nan)
#             temp_img = resampling_alberto(temp_img, np.nan)
#             diff_img = resampling_alberto(diff_img, np.nan)
#         except Exception:
#             n_exceptions += 1
#             # print(obj_id, 'is an exception')
#             continue

#         if len(sci_img[~np.isnan(sci_img)]) != 0:
#             if obj_id not in dict_ids_files.keys():
#                 dict_ids_files[obj_id] = [file_id]
#             else:
#                 print('a')
#                 dict_ids_files[obj_id].append(file_id)


### Stacking images for each of the 1123(5) objects

# Declaring tables with all the objects
# pdf_dir = f'{path}/images_stack_sigclip/'
# if not os.path.isdir(pdf_dir): os.mkdir(pdf_dir)
# n_epochs = [len(dict_mjds[key][0]) for key in dict_mjds]

# idx = -1
# exception_ids = []
# print('\n\n\033[1m - Start stacking and sigma-clipping data...\n\033[0m')
# for key, value in tqdm(dict_mjds.items()):

#     # idx += 1
#     # if idx < 977:
#     #     continue

#     print(key, len(value), end=' ')
#     breakpoint()
#     data_epochs = [dict_sci_img[key], dict_temp_img[key], dict_diff_img[key]]
#     print(len(data_epochs[0]), 'epochs:', data_epochs[0][0].shape)
#     pdf = PdfPages(f'{pdf_dir}/{key}_plots.pdf')
#     # pdf = ""
#     try:
#         stacked = stack_epochs(data_epochs, key, labels, pdf)
#         stacked[0] = resampling_alberto(stacked[0], np.nan)
#         stacked[1] = resampling_alberto(stacked[1], np.nan)
#         stacked[2] = resampling_alberto(stacked[2], np.nan)
#     except Exception:
#         # print(f'Object {key} skipped, it has only strange values...')
#         exception_ids.append(key)
#         pdf.close()
#         continue
#     make_three_plots(stacked, key, labels, "median", pdf=pdf)

#     clipped_sci = sigma_clip(stacked[0], sigma=3, maxiters=10)  # 3 or 5
#     clipped_temp = sigma_clip(stacked[1], sigma=3, maxiters=10)  # 3 or 5
#     clipped_sci, clipped_temp = clipped_sci.filled(np.nan), clipped_temp.filled(np.nan)
#     # final_sci, final_temp = replace_clipped(clipped_sci, clipped_temp, np.ma.masked)
#     final_sci = resampling_alberto(clipped_sci, np.nan)
#     final_temp = resampling_alberto(clipped_temp, np.nan)
#     plot_clipped_and_resampled(clipped_sci, final_sci, key, "Science", pdf)
#     plot_clipped_and_resampled(clipped_temp, final_temp, key, "Template", pdf)
#     pdf.close()
#     to_table = [key, len(data_epochs[0]), stacked[0], stacked[1], stacked[2],
#                 clipped_sci, clipped_temp, final_sci, final_temp]
#     final_table.add_row(to_table)


#### Saving final table
# breakpoint()
print(f'final_table has {len(final_table.colnames)} columns and {len(final_table)} rows')
table_name = f'{path}/SIMBAD_stacked_sigclipped_resampled_pq{pq_idx_start}to{pq_idx_end}.fits'
final_table.write(table_name, format='fits', overwrite=True)
print(exception_ids)
