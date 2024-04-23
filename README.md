# ELEPHANT: ExtragaLactic alErt Pipeline for Hostless AstroNomical Transients
This repository contains the ELEPHANT pipeline described in the paper(link).

To use the pipeline, you can clone this repository with the command below

    git clone https://github.com/COINtoolbox/extragalactic_hostless.git

After cloning install the necessary packages with the command below 

    pip install -r requirements.txt

The pipeline parameters can be configured in pipeline_config.json file.
A sample input parquet file is available in `data` folder

    {
        "parquet_files_list": "data/*.parquet",
        "save_directory": "/path/to/save/results/",
        ...
    }

To run the pipeline use the command below

    python run_pipeline.py

The pipeline generates a result parquet file with the following columns for each input parquet file

- **b:cutoutScience_stampData_stacked:** stacked science images 
- **b:cutoutTemplate_stampData_stacked:** stacked template images
- **b:cutoutDifference_stampData_stacked:** stacked difference images
- **science_clipped:** stacked sigma clipped science image
- **template_clipped:** stacked sigma clipped template image
- **number_of_stamps_in_stacking:** number of images used for stacking after FWHM stamp preprocessing
- **is_hostless_candidate_clipping:** True, if the candidate flagged as hostless by sigma clipping approach
- **distance_science:** distance from transient to the nearest mask in pixels
- **kstest_SCIENCE_N_statistic:** Kolmogorov-Smirnov test statistic value for N x N cutout science image
- **kstest_SCIENCE_N_pvalue:** Kolmogorov-Smirnov test p-value for N x N cutout science image
- **kstest_TEMPLATE_N_statistic:** Kolmogorov-Smirnov test statistic value for N x N cutout template image
- **kstest_TEMPLATE_N_pvalue:** Kolmogorov-Smirnov test p-value for N x N cutout template image

The project is part of [COIN Residence Program #7, Portugal, 2023](https://cosmostatistics-initiative.org/residence-programs/crp7/)




