# fMRI data analysis

## Signal processing from scanner
0. `[0.preprocess fmri.py](https://github.com/nmningmei/unconfeats/blob/main/scripts/fMRI/nipype_preprocessing_pipeline/0.preprocess%20fmri.py)` is used for removing motion noise and smoothing for each run.
1. And then we use`[1.ICA_AROMA.py](https://github.com/nmningmei/unconfeats/blob/main/scripts/fMRI/nipype_preprocessing_pipeline/1.ICA_AROMA.py)` to apply ICA AROMA to further denoising.
2. In parallel, we use `[2.freesurfer reconall.py](https://github.com/nmningmei/unconfeats/blob/main/scripts/fMRI/nipype_preprocessing_pipeline/2.freesurfer%20reconall.py)` to generate the brain model using Freesurfer.
3. And then we apply highpass filer to the preprocessed runs using `[4.highpass filter.py](4.highpass filter.py)`.
4. `[3.extract ROI and convert to BOLD space.py](https://github.com/nmningmei/unconfeats/blob/main/scripts/fMRI/nipype_preprocessing_pipeline/3.extract%20ROI%20and%20convert%20to%20BOLD%20space.py)` is used for ROI extraction and we convert the extracted ROI masks from high-resolution (freesurfer) space to functional space (BOLD-FSL).


## Event-related fMRI partitioning, detrending, and zscoring
0. We manually match the event files for each subject: `matching_files.py`
1. We first label the volumes based on the [event files we collected from the Psychopy experiment](https://github.com/nmningmei/unconfeats/tree/main/data/behavioral) using `[create event file.py](https://github.com/nmningmei/unconfeats/blob/main/scripts/fMRI/nipype_convert_create_folders/create%20event%20file.py)`. This allows us to check if the event-related volumes are created correctly easily since we are dealing wit CSV files at this stage.
2. And then we apply detrending to all the volumes and zscoring volumes that is 4 - 7 seconds after the onset of the image using `[stacking runs combining left and right.py](https://github.com/nmningmei/unconfeats/blob/main/scripts/fMRI/nipype_convert_create_folders/stacking%20runs%20combining%20left%20and%20right.py)`. This script is written for extracting volumes of interest for each ROI mask. If you want to extract the whole-brain data: `stacking runs+whole brain.py`


## Machine learning -- ROI-based decoding
1. Decoding for each ROI data: `[decoding_pipeline.py](https://github.com/nmningmei/unconfeats/blob/main/scripts/fMRI/cross_state_decoding_leave_2_instance_out/decoding_pipeline.py)`. 
2. Perform nonparametric statistics on the decoding scores, comparing against to empirical chance level: `[
decoding_pipeline_post_stats.py ](https://github.com/nmningmei/unconfeats/blob/main/scripts/fMRI/cross_state_decoding_leave_2_instance_out/decoding_pipeline_post_stats.py)`
3. Figures in the paper could be plotted using: `[decoding_pipeline_plot.py](https://github.com/nmningmei/unconfeats/blob/main/scripts/fMRI/cross_state_decoding_leave_2_instance_out/decoding_pipeline_plot.py)`
