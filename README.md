# ELEPHANT: ExtragaLactic Pipeline for Hostless AstroNomical Transients
This repository contains the pipeline for potential hostless transients detection discussed in the paper(link).


Install the requirements with the command below 

    pip install -r requirements.txt

The pipeline parameters can be configured in pipeline_config.json file

    {
        "parquet_files_list": "data/*.parquet",
        "save_directory": "/path/to/save/results/",
        ...
    }

To start the pipeline use the command below

    python run_pipeline.py

The pipeline generates a result parquet file with the following columns for each input parquet file

- **b:cutoutScience_stampData_stacked:** stacked science images 
- **b:cutoutTemplate_stampData_stacked:** stacked template images
- **b:cutoutDifference_stampData_stacked:** stacked difference images
- **science_clipped:** stacked sigma clipped science image
- **template_clipped:** stacked sigma clipped template image
- **number_of_stamps_in_stacking:** number of images used for stacking after FWHM stamp preprocessing
- **is_hostless_candidate_clipping:** True, if the candidate flagged as hostless by sigma clipping approach
- **distance_science:** distance from transient to nearest mask in pixels
- **anderson-darling_SCIENCE_N_statistic:** Anderson darling test statistic value for N x N cutout science image
- **anderson-darling_SCIENCE_N_pvalue:** Anderson darling test p-value for N x N cutout science image
- **anderson-darling_TEMPLATE_N_statistic:** Anderson darling test statistic value for N x N cutout template image
- **anderson-darling_TEMPLATE_N_pvalue:** Anderson darling test p-value for N x N cutout template image
- **kstest_SCIENCE_N_statistic:** Kolmogorov-Smirnov test statistic value for N x N cutout science image
- **kstest_SCIENCE_N_pvalue:** Kolmogorov-Smirnov test p-value for N x N cutout science image
- **kstest_TEMPLATE_N_statistic:** Kolmogorov-Smirnov test statistic value for N x N cutout template image
- **kstest_TEMPLATE_N_pvalue:** Kolmogorov-Smirnov test p-value for N x N cutout template image

The project is part of [COIN Residence Program #7, Portugal, 2023](https://cosmostatistics-initiative.org/residence-programs/crp7/)




