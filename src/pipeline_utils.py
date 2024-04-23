"""
    pipeline utils functions
"""

import gzip
import io
import json
import os.path
from typing import Dict, List, Tuple, Union

from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.visualization import simple_norm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import powerspectrum as ps


def load_json(file_path: str) -> Dict:
    """
    Loads json file

    Parameters
    ----------
    file_path
       input json file path
    """
    with open(file_path) as json_file:
        return json.load(json_file)


def read_parquet_to_df(file_path: str) -> pd.DataFrame:
    """
    Loads parquet file data to pandas dataframe

    Parameters
    ----------
    file_path
       input parquet file path
    """
    return pd.read_parquet(file_path)


def _select_indices_with_fwhm_bins(data: pd.DataFrame,
                                   bins: List) -> Union[List, None]:
    """
    Finds epoch indices with lower FWHM bin

    Parameters
    ----------
    data
       candidate data
    bins
        FWHM bins list
    """
    indices = np.where(data['i:fwhm'] < bins[0])[0]
    if len(indices) > 0:
        return indices
    indices = np.where((data['i:fwhm'] >= bins[0]) &
                       (data['i:fwhm'] < bins[1]))[0]
    if len(indices) > 0:
        return indices
    indices = np.where((data['i:fwhm'] >= bins[1]) &
                       (data['i:fwhm'] < bins[2]))[0]
    if len(indices) > 0:
        return indices
    return None


def maybe_filter_stamps_with_fwhm(
        data: pd.Series, fwhm_bins: List) -> pd.DataFrame:
    """
    Filters candidate's epochs based on FWHM bins for stacking

    Parameters
    ----------
    data
       candidate data
    bins
        FWHM bins list
    """
    data_df = pd.DataFrame.from_dict(dict(zip(data.index, data.values)))
    number_of_epochs = len(data_df)
    if number_of_epochs > 1:
        larger_fwhm_indices = np.where(data_df['i:fwhm'] >= 3)[0]
        if number_of_epochs == len(larger_fwhm_indices):
            indices = np.where(data_df['i:fwhm'] == min(data_df['i:fwhm']))[0]
            return data_df.iloc[indices]
        indices = _select_indices_with_fwhm_bins(data_df, fwhm_bins)
        if indices is not None:
            data_df = data_df.iloc[indices]
    return data_df


def read_bytes_image(bytes_str: bytes) -> np.ndarray:
    """
    Reads bytes image stamp

    Parameters
    ----------
    bytes_str
       input byte string
    """
    hdu_list = fits.open(gzip.open(io.BytesIO(bytes_str)))
    primary_hdu = hdu_list[0]
    return primary_hdu.data


def read_stamp_bytes_data(data: pd.DataFrame) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Reads candidate's science, template and difference stamps
    data of all epochs

    Parameters
    ----------
    data
        candidate data
    """
    science_stamp = (data["b:cutoutScience_stampData"].apply(
        read_bytes_image))
    template_stamp = (data["b:cutoutTemplate_stampData"].apply(
        read_bytes_image))
    difference_stamp = (data["b:cutoutDifference_stampData"].apply(
        read_bytes_image))
    return science_stamp, template_stamp, difference_stamp


def resample_with_gaussian_kde(stamp_data, output_shape: List) -> Tuple[
        None, np.ndarray]:
    """
    Replace none and nan values by resampling data with gaussian kernels

    Parameters
    ----------
    stamp_data
        candidate stamp data
    output_shape
        expected output shape of stamp
    """
    if stamp_data.shape != tuple(output_shape):
        return None
    cleaned_data = stamp_data[stamp_data != np.array(None)]
    cleaned_data = cleaned_data[~np.isnan(cleaned_data)]
    if cleaned_data.size != stamp_data.size:
        stamp_data_copy = stamp_data.flatten()
        gaussian_kde_class = stats.gaussian_kde(
            cleaned_data.flatten(), bw_method=0.1)
        gaussian_kde_samples = gaussian_kde_class.resample(size=5000)[0]
        number_of_invalid_samples = len(
            stamp_data[np.where(stamp_data == np.array(None))])
        stamp_data_copy[np.where(stamp_data_copy == np.array(None))] = (
            np.random.choice(gaussian_kde_samples,
                             size=number_of_invalid_samples))
        number_of_invalid_samples = len(stamp_data[np.isnan(stamp_data)])
        stamp_data_copy[np.isnan(stamp_data_copy)] = np.random.choice(
            gaussian_kde_samples, size=number_of_invalid_samples)
        return stamp_data_copy.reshape(output_shape)
    return stamp_data


def apply_median_stacking(data: pd.DataFrame, output_shape):
    """
    Applies median stacking

    Parameters
    ----------
    data
        candidate stamp data
    output_shape
        expected output shape of stamp
    """
    data = data.apply(resample_with_gaussian_kde, args=(output_shape,))
    data = data.dropna()
    if data.size == 0:
        return None
    return np.median(np.stack(data), axis=0)


def maybe_save_stacked_images(
        data_to_plot: List, candidate_id: str, subplot_labels: List,
        config: Dict, folder_name: str):
    """
    Saves stacked images

    Parameters
    ----------
    data_to_plot
        list of stacked images to plot
    candidate_id
        object id of the candidate
    subplot_labels
        labels for subplots
    config
        config dict
    folder_name
        folder name to save images which will be created in save folder
        provided in config
    """
    if config["is_save_stacked_images"]:
        save_folder = os.path.join(config["save_directory"], folder_name)
        os.makedirs(save_folder, exist_ok=True)
        save_file_name = os.path.join(
            save_folder, candidate_id + ".png")
        fig, axes = plt.subplots(1, len(data_to_plot), figsize=(15, 5))
        for index, ax in enumerate(axes.flat):
            norm = simple_norm(data_to_plot[index], 'log', percent=99.)
            ax.imshow(data_to_plot[index], norm=norm)
            ax.set_title(f"{subplot_labels[index]}: {candidate_id}")
        plt.savefig(save_file_name)
        # plt.show()
        plt.close()


def apply_sigma_clipping(input_data: np.ndarray,
                         sigma_clipping_kwargs: Dict) -> [
        np.ma.masked_array, np.ma.masked]:
    """
    Applies sigma clippng

    Parameters
    ----------
    input_data
        stacked input data
    sigma_clipping_kwargs
        parameters for astropy sigma_clip function
    """
    return sigma_clip(input_data, **sigma_clipping_kwargs)


def crop_center_patch(input_image: np.ndarray, patch_radius: int = 7) -> np.ndarray:
    """
    crops rectangular patch around image center with a given patch scale

    Parameters
    ----------
    input_image
       input image
    patch_radius
        patch radius in pixels
    """
    image_shape = input_image.shape[0:2]
    center_coords = [image_shape[0] / 2, image_shape[1] / 2]
    center_patch_x = int(center_coords[0] - patch_radius)
    center_patch_y = int(center_coords[1] - patch_radius)
    return input_image[center_patch_x:center_patch_x + patch_radius * 2,
           center_patch_y:center_patch_y + patch_radius * 2]


def _check_hostless_conditions(
        science_clipped: np.ndarray, template_clipped: np.ndarray,
        detection_config: Dict):
    num_science_pixels_masked = np.ma.count_masked(science_clipped)
    num_template_pixels_masked = np.ma.count_masked(template_clipped)
    if (num_science_pixels_masked > detection_config[
        "max_number_of_pixels_clipped"] and
            num_template_pixels_masked < detection_config[
                "min_number_of_pixels_clipped"]):
        return True
    if (num_template_pixels_masked > detection_config[
        "max_number_of_pixels_clipped"] and
            num_science_pixels_masked < detection_config[
                "min_number_of_pixels_clipped"]):
        return True
    return False


def run_hostless_detection_with_clipped_data(
        science_stamp: np.ndarray, template_stamp: np.ndarray,
        configs: Dict, image_shape: List) -> bool:
    """
    Detects potential hostless candidates with sigma clipped stamp images by
     cropping an image patch from the center of the image.
    If pixels are rejected in scientific image but not in corresponding
     template image, such candidates are flagged as potential hostless

    Parameters
    ----------
    science_stamp
       science image
    template_stamp
        template image
    detection_configs
        configs with detection threshold
    image_shape
        image shape
    """
    sigma_clipping_config = configs["sigma_clipping_kwargs"]

    science_clipped = apply_sigma_clipping(science_stamp, sigma_clipping_config)
    template_clipped = apply_sigma_clipping(
        template_stamp, sigma_clipping_config)
    detection_config = configs["hostless_detection_with_clipping"]
    is_hostless_candidate = _check_hostless_conditions(
        science_clipped, template_clipped, detection_config)
    if is_hostless_candidate:
        return is_hostless_candidate
    science_stamp = crop_center_patch(
        science_stamp, detection_config["crop_radius"])
    template_stamp = crop_center_patch(
        template_stamp, detection_config["crop_radius"])
    science_clipped = apply_sigma_clipping(
        science_stamp, sigma_clipping_config)
    template_clipped = apply_sigma_clipping(
        template_stamp, sigma_clipping_config)
    is_hostless_candidate = _check_hostless_conditions(
        science_clipped, template_clipped, detection_config)
    return is_hostless_candidate

def calculate_distance(table_seg: np.ndarray, radius: float = -1):
    #TODO(utthishtastro): Refactor
    """
    Calculates euclidian distance between the transient and
     the closest masked source (in pixels)

    Args:
        table_seg (np.ndarray): stamp of the segmented image
        radius (float, optional): radius of the search. Defaults to -1,
         which will be set to the size of the stamp.

    Returns:
        int: x position of the closest masked source
        int: y position of the closest masked source
        float: euclidian distance between the transient and the closest masked source
    """

    # Converting the stamp to a binary mask
    table_seg[table_seg > 0] = 1

    # Defining the position of the transient
    transient_idx = 30  # assuming the size of the stamp is always ~60x60

    # Initializing variables
    breaker = False

    # Initializing lists
    list_distance = []
    list_mask_i = []
    list_mask_j = []

    # Setting the radius of the search to the size of the stamp if not defined
    if radius == -1:
        radius = transient_idx

    # If the transient is masked, the distance is 0
    if table_seg[transient_idx][transient_idx] == 1:
        return transient_idx, transient_idx, 0

    # Searching for the closest masked source
    for step in np.arange(0, radius + 1):
        array = np.arange(transient_idx - 1 - step, transient_idx + 2 + step)
        for indx, i in enumerate(array):
            if indx == 0 or i == int(transient_idx + 1 + step):
                for j in array:
                    if table_seg[i][j] == 1:
                        list_distance.append(np.sqrt((transient_idx - i) ** 2 + (transient_idx - j) ** 2))
                        list_mask_i.append(i)
                        list_mask_j.append(j)

            else:
                for j in [array[0], array[-1]]:
                    if table_seg[i][j] == 1:
                        list_distance.append(np.sqrt((transient_idx - i) ** 2 + (transient_idx - j) ** 2))
                        list_mask_i.append(i)
                        list_mask_j.append(j)

            if len(list_distance) > 0:
                breaker = True
                break
        if breaker:
            break

            # Getting the closest masked source
    index = np.where(np.array(list_distance) == np.min(list_distance))[0][0]
    mask_i = list_mask_i[index]
    mask_j = list_mask_j[index]
    distance = list_distance[index]
    return mask_i, mask_j, distance


def run_distance_calculation(
        science_image_clipped: np.ma.masked_array,
        template_image_clipped: np.ma.masked_array) -> Tuple[float, float]:
    """
    Runs distance computation between the transient and closest masked source

    Parameters
    ----------
    science_image_clipped
       sigma clipped science image
    template_image_clipped
       sigma clipped template image
    """
    science_masked = science_image_clipped.mask.astype(int)
    template_masked = template_image_clipped.mask.astype(int)
    # TODO(utthishtastro): Check why try and exception is required here
    try:
        _, _, distance_science = calculate_distance(science_masked)
        _, _, distance_template = calculate_distance(template_masked)
    except Exception:
        distance_science = -1
        distance_template = -1
    return distance_science, distance_template


def create_noise_filled_mask(image_data: np.ndarray,
                             mask_data: np.ndarray, image_size: List):
    """
    Creates input image data with noise filled mask
    Parameters
    ----------
    image_data
        input stacked image data
    mask_data
        corresponding input masked data
    image_size
        output image size
    """
    mask = mask_data > 0
    for_filling = np.random.normal(
        np.median(image_data[~mask]), np.std(image_data[~mask]),
        image_size)
    for_filling = np.where(mask, for_filling, 0)
    to_fill = np.where(mask, 0, image_data)
    return to_fill + for_filling


def run_powerspectrum_analysis(
        science_image: np.ndarray, template_image: np.ndarray,
        science_mask: np.ndarray, template_mask: np.ndarray,
        image_size: List, number_of_iterations: int = 200) -> Dict:
    """

    Parameters
    ----------
    science_image
        science stamp
    template_image
        template stamp
    science_mask
        sigma clipped science image
    template_mask
        sigma clipped template image
    image_size
        output image size
    number_of_iterations
        number of iterations for powerspectrum analysis shuffling

    """
    science_data = create_noise_filled_mask(
        science_image, science_mask, image_size)
    template_data = create_noise_filled_mask(
        template_image, template_mask, image_size)
    _, kstest_results_dict = ps.detect_host_with_powerspectrum(
        science_data, template_data, number_of_iterations=number_of_iterations,
        metric="kstest")
    return kstest_results_dict

