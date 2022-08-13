#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 13:53:38 2021

@author: nmei
"""

import os
import numpy as np
from glob import glob

from nilearn.input_data import NiftiMasker

from shutil import copyfile
#copyfile('../../../utils.py','utils.py')
from utils import nipype_fsl_randomise

if __name__ == "__main__":
    radius              = 6
    folder_name         = f'RSA_searchlight_long_process_{radius}mm'
    working_dir         = '../../../../results/MRI/nilearn'
    working_data        = np.sort(glob(os.path.join(working_dir,
                                                    folder_name,
                                                    "*",
                                                    "*",
                                                    "*RSA.nii.gz")))
    chance_data         = np.sort(glob(os.path.join(working_dir,
                                                    folder_name,
                                                    "*",
                                                    "*",
                                                    "*RSA_chance.nii.gz")))
    mask_dir            = '../../../../data/MRI/{}/anat/ROI_BOLD/combine_BOLD.nii.gz'
    functional_rain     = '../../../../data/MRI/{}/func/example_func.nii.gz'
    
    idx                 = 0
    filename            = working_data[idx]
    temp                = filename.split('/')[-1]
    (sub,
     conscious_source,
     conscious_target,
     model_name,
     radius,
     _)                 = temp.split('_')
    
    # load the data and get the difference between them
    masker              = NiftiMasker(mask_img = mask_dir.format(sub),).fit()
    BOLD_signals        = masker.transform(filename)
    chance_signals      = masker.transform(filename.replace("RSA.nii","RSA_chance.nii"))
    
    diff                = np.zeros(BOLD_signals.shape)
    signs               = np.sign(BOLD_signals) == np.sign(chance_signals)
    temp                = np.abs(BOLD_signals) - np.abs(chance_signals)
    diff[signs]         = temp[signs]
    
    
    whole_brain_mask    = mask_dir.format(sub)
    example_func        = functional_rain.format(sub)
    input_file          = os.path.abspath(filename)
    mask_file           = os.path.abspath(whole_brain_mask)
    base_name           = os.path.abspath(filename.split('_RSA.nii')[0])
    
    if not os.path.exists(base_name + "_tfce_corrp_tstat1.nii.gz"):
        nipype_fsl_randomise(input_file,
                             mask_file,
                             base_name,
                             tfce                   = True,
                             var_smooth             = 6,
                             demean                 = False,
                             one_sample_group_mean  = True,
                             n_permutation          = int(1e4),
                             quiet                  = True,
                             run_algorithm          = True,
                             )
        