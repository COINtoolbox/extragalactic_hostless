"""
    extragalactic potential hostless candidates analysis pipeline
"""
import glob
import os
from typing import Dict, Tuple

from multiprocessing import Pool
import numpy as np
import pandas as pd
from tqdm import tqdm
from pipeline_utils import (
    load_json, read_parquet_to_df, maybe_filter_stamps_with_fwhm,
    read_stamp_bytes_data, apply_median_stacking, maybe_save_stacked_images,
    apply_sigma_clipping, run_hostless_detection_with_clipped_data,
    run_distance_calculation, resample_with_gaussian_kde, read_bytes_image,
    run_powerspectrum_analysis)


class HostLessExtragalactic:
    """
    Main class

    Parameters
    ----------
    configs
       input config file with different input parameters to use in the class
       Check example pipeline_config.json file
    """
    def __init__(self, configs: Dict):
        self.configs = configs
        self._parquet_files_list = sorted(glob.glob(
            self.configs["parquet_files_list"]))  # List
        self._image_shape = self.configs["image_shape"]  # List
        self._subplot_labels = ["Science", "Template", "Difference"]  # List
        self._last_stamp_data_list = []  # List
        self._stacked_data_list = []  # List

    def process_candidate(self, data: Tuple):
        """
        Processes each candidate

        Parameters
        ----------
        data
           candidate data
        """
        object_id = data[1]["objectId"]
        # temporary hack..
        data_original = data[1].drop(["tracklet"])
        if np.array(data_original["b:cutoutTemplate_stampData"]).size == 1:
            return None
        data_df = maybe_filter_stamps_with_fwhm(
            data_original, self.configs["fwhm_bins"])
        science_stamp, template_stamp, difference_stamp = (
                read_stamp_bytes_data(data_df))
        number_of_stamps_in_stacking = len(science_stamp)
        science_stamp, template_stamp, difference_stamp = (
            self._run_median_stacking(
                science_stamp, template_stamp, difference_stamp))
        if science_stamp is None:
            return
        data_to_plot = [science_stamp, template_stamp, difference_stamp]
        maybe_save_stacked_images(
            data_to_plot, object_id, self._subplot_labels, self.configs,
            "stacked")
        science_stamp_clipped, template_stamp_clipped = (
            self._run_sigma_clipping(science_stamp, template_stamp))
        data_to_plot = [science_stamp_clipped, template_stamp_clipped]
        is_hostless_candidate = run_hostless_detection_with_clipped_data(
            science_stamp_clipped, template_stamp_clipped,
            self.configs, self._image_shape)

        maybe_save_stacked_images(
            data_to_plot, object_id, self._subplot_labels, self.configs,
            "sigma_clipped")

        last_df_resampled = self._get_resampled_last_df(
            data_df.iloc[-1])
        data_df["b:cutoutScience_stampData"] = last_df_resampled[
            "b:cutoutScience_stampData"]
        data_df["b:cutoutTemplate_stampData"] = last_df_resampled[
            "b:cutoutTemplate_stampData"]
        data_df["b:cutoutDifference_stampData"] = last_df_resampled[
            "b:cutoutDifference_stampData"]
        self._last_stamp_data_list.append(self._reformat_last_df(data_original))
        distance_science, distance_template = run_distance_calculation(
            science_stamp_clipped, template_stamp_clipped)
        power_spectrum_results = {}
        if is_hostless_candidate:
            power_spectrum_results = run_powerspectrum_analysis(
                science_stamp, template_stamp, science_stamp_clipped.mask.astype(int),
                template_stamp_clipped.mask.astype(int), self._image_shape)
        self._create_stacked_df(
            object_id, science_stamp, template_stamp, difference_stamp,
            science_stamp_clipped, template_stamp_clipped,
            number_of_stamps_in_stacking, is_hostless_candidate,
            distance_science, distance_template, power_spectrum_results)

    def _get_resampled_last_df(self, last_data_df: pd.DataFrame):
        """
        Resamples last image stamp data

        Parameters
        ----------
        last_data_df
            data of last stamp
        """
        last_data_df_copy = last_data_df.copy(deep=True)
        last_science_stamp = read_bytes_image(
            last_data_df["b:cutoutScience_stampData"])
        last_science_stamp = resample_with_gaussian_kde(
            last_science_stamp, self._image_shape)
        if last_science_stamp is not None:
            last_data_df_copy["b:cutoutScience_stampData"] = (
                last_science_stamp.tobytes())
        last_template_stamp = read_bytes_image(
            last_data_df["b:cutoutTemplate_stampData"])
        last_template_stamp = resample_with_gaussian_kde(
            last_template_stamp, self._image_shape)
        if last_template_stamp is not None:
            last_data_df_copy["b:cutoutTemplate_stampData"] = (
                last_template_stamp.tobytes())
        last_difference_stamp = read_bytes_image(
            last_data_df["b:cutoutDifference_stampData"])
        last_difference_stamp = resample_with_gaussian_kde(
                last_difference_stamp, self._image_shape)
        if last_difference_stamp is not None:
            last_data_df_copy["b:cutoutDifference_stampData"] = (
                last_difference_stamp.tobytes())
        return last_data_df_copy

    def _create_stacked_df(
            self, object_id: str, science_stacked: np.ndarray,
            template_stacked: np.ndarray, difference_stacked: np.ndarray,
            science_clipped: np.ndarray, template_clipped: np.ndarray,
            number_of_stamps_in_stacking: int, is_hostless_candidate: bool,
            distance_science: float, distance_template: float,
            power_spectrum_results: Dict):
        """
        Creates stacked science, template and difference stamp with median FWHM
        and corresponding hostless analysis results

        Parameters
        ----------
        object_id
           candidate id
        science_stacked
            stacked science image
        template_stacked
            stacked template image
        difference_stacked
            stacked difference image
        number_of_stamps_in_stacking
            number of stamps used in stacking after FWHM filtering
        is_hostless_candidate
            is the candidate potential hostless
        distance_science
            euclidian distance between the transient and the
             closest masked source in science image
        distance_template
            euclidian distance between the transient and the
            closest masked source in template image
        power_spectrum_results
            dict with power spectrum results
        """
        data_dict = {
            "objectId": object_id,
            "b:cutoutScience_stampData_stacked": science_stacked.tobytes(),
            "b:cutoutTemplate_stampData_stacked": template_stacked.tobytes(),
            "b:cutoutDifference_stampData_stacked": difference_stacked.tobytes(),
            "science_clipped": science_clipped.mask.astype(int).tobytes(),
            "template_clipped": template_clipped.mask.astype(int).tobytes(),
            "number_of_stamps_in_stacking": number_of_stamps_in_stacking,
            "is_hostless_candidate_clipping": is_hostless_candidate,
            "distance_science": distance_science,
            "distance_template": distance_template
        }
        data_dict.update(power_spectrum_results)
        self._stacked_data_list.append(pd.Series(data=data_dict))

    def process_parquet_file(self, file_path: str):
        """
        Processes current parquet file data

        Parameters
        ----------
        file_path
           path to parquet file
        """
        parquet_data = read_parquet_to_df(file_path)
        for each_candidate in tqdm(parquet_data.iterrows()):
            self.process_candidate(each_candidate)

    def run(self, index):
        """
        Main run method
        """
        each_file = self._parquet_files_list[index]
        parquet_file_name = os.path.basename(each_file).replace(
            ".parquet", "")
        self.process_parquet_file(each_file)
        last_stamp_df = pd.DataFrame(self._last_stamp_data_list)
        stacked_results_df = pd.DataFrame(self._stacked_data_list)
        combined_results_save_fname = os.path.join(
            self.configs["save_directory"],
            parquet_file_name + "_final_results_df.parquet")
        combined_results = pd.merge(
            last_stamp_df, stacked_results_df, on="objectId")
        combined_results.to_parquet(combined_results_save_fname)
        self._last_stamp_data_list = []
        self._stacked_data_list = []

    @staticmethod
    def _reformat_last_df(candidate_df_original: pd.DataFrame):
        columns_to_ignore = [
            "objectId", "b:cutoutScience_stampData",
            "b:cutoutTemplate_stampData", "b:cutoutDifference_stampData"]
        new_df = pd.DataFrame()
        candidate_df = pd.DataFrame.from_dict(dict(zip(
            candidate_df_original.index, candidate_df_original.values)))

        for each_column in candidate_df.columns:
            if each_column not in columns_to_ignore:
                new_df[each_column] = candidate_df.groupby(
                    'objectId').agg({each_column: lambda x: list(x)})
        for each_column in columns_to_ignore:
            new_df[each_column] = candidate_df[each_column].values[0]
        return new_df.squeeze()

    def _run_median_stacking(self, science_stamp, template_stamp,
                             difference_stamp) -> Tuple[
            np.ndarray, np.ndarray, np.ndarray]:
        """
        Runs median stacking for science, template and difference stamps

        Parameters
        ----------
        science_stamp
           science stamp images
        template_stamp
            template stamp images
        difference_stamp
            difference stamp images
        """
        science_stamp = apply_median_stacking(science_stamp, self._image_shape)
        template_stamp = apply_median_stacking(
            template_stamp, self._image_shape)
        difference_stamp = apply_median_stacking(
            difference_stamp, self._image_shape)
        return science_stamp, template_stamp, difference_stamp

    def _run_sigma_clipping(
            self, science_stamp: np.ndarray,
            template_stamp: np.ndarray) -> Tuple[np.ma.masked_array,
                np.ma.masked_array]:
        """
        Runs sigma clipping

        Parameters
        ----------
        science_stamp
           science stamp images
        template_stamp
            template stamp images
        """
        science_stamp_clipped = apply_sigma_clipping(
            science_stamp, self.configs["sigma_clipping_kwargs"])
        template_stamp_clipped = apply_sigma_clipping(
            template_stamp, self.configs["sigma_clipping_kwargs"])
        return science_stamp_clipped, template_stamp_clipped


def run_in_parallel(index):
    CONFIG_PATH = "pipeline_config.json"
    CONFIG_DATA = load_json(CONFIG_PATH)
    run_class = HostLessExtragalactic(CONFIG_DATA)
    run_class.run(index)


if __name__ == '__main__':
    CONFIG_PATH = "pipeline_config.json"
    CONFIG_DATA = load_json(CONFIG_PATH)

    parquet_files_list = sorted(glob.glob(
        CONFIG_DATA["parquet_files_list"]))
    pool = Pool(processes=CONFIG_DATA["number_of_processes"])
    pool.map(run_in_parallel, range(len(parquet_files_list)))
