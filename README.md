[![crp7](https://img.shields.io/badge/CRP-7-%23ED9145?labelColor=%23ED9145&color=%2321609D)](https://cosmostatistics-initiative.org/residence-programs/crp7/)
[![arXiv](https://img.shields.io/badge/arXiv-astro--ph%2FXXXX.XXX-%23ED9145?labelColor=%23ED9145&color=%2321609D)](https://arxiv.org/abs/XXXX.XXXX) 


# <img align="left" src="images/Elephant.png">  <img align="right" src="images/coin_logo.png" width="350"> ELEPHANT: ExtragaLactic alErt Pipeline for Hostless AstroNomical Transients  


This repository contains the ELEPHANT pipeline described in the paper(link).

To use the pipeline, you can clone this repository with the command below

    git clone https://github.com/COINtoolbox/extragalactic_hostless.git

After cloning install the necessary packages with the command below 

    pip install -r requirements.txt

The pipeline parameters can be configured in pipeline_config.json file.
A subset of sample data used in the paper is available in `data` folder. The entire dataset used in the paper 
can be downloaded from the [Fink broker data server](https://fink-portal.org)


    {
        "parquet_files_list": path to downloaded input parquet files (An example file available in data folder)
        "save_directory": path to a folder to save results
        "fwhm_bins":  A list of FWHM bin values (check paper)
        "image_shape": Input stamps shape
        "is_save_stacked_images": If true, stacked images are saved in results "save_directory" folder,
        "sigma_clipping_kwargs": kwargs parameters for astropy sigma_clip function
        "hostless_detection_with_clipping": sigma clipping hosteless detection threshold parameters defined in pixels
        "number_of_processes": Number of workers used in pythong multiprocessing to process files in parallel
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




