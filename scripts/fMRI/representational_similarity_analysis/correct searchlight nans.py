#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 07:52:14 2020

@author: nmei
"""

import os
from glob import glob

import numpy as np

from nilearn.input_data import NiftiMasker

for ii in np.arange(1,8):
    sub = f'sub-0{ii}'
    working_dir = f'../../../../results/MRI/nilearn/RSA_searchlight/{sub}'
    working_data = glob(os.path.join(working_dir,'*.nii.gz'))
    mask_dir = f'/bcbl/home/home_n-z/nmei/MRI/uncon_feat/MRI/{sub}/anat/ROI_BOLD/'
    mask_file = glob(os.path.join(mask_dir,'combine_BOLD.nii.gz'))[0]
    
    saving_dir = f'../../../../results/MRI/nilearn/RSA_searchlight_corrected/{sub}'
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)
    
    for file_name in working_data:
        file_name
        conscious_state = file_name.split('/')[-1].replace('.nii.gz','')
        
        masker = NiftiMasker(mask_img = mask_file)
        temp = masker.fit_transform(file_name)
        
        temp_inv = masker.inverse_transform(temp)
        
        temp_inv.to_filename(os.path.join(saving_dir,f'{conscious_state}.nii.gz'))
