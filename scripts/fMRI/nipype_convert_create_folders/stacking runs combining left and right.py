#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 10:09:31 2019

@author: nmei

this script takes each of the fMRI data paired with its baharioval psychophy file
and selects volumes of interests for stacking

we combine the left and right ROIs before processing

"""

import os
import re
import pandas as pd
import numpy  as np
from glob               import glob
from tqdm               import tqdm
from nilearn.input_data import NiftiMasker
from nilearn.image      import index_img,new_img_like,concat_imgs
from nilearn.signal     import clean as clean_signal
from nibabel            import load as load_img
from shutil             import copyfile,rmtree
copyfile('../../utils.py','utils.py')
from utils              import groupby_average,load_csv

sub         = 'sub-07'
folder_name = ''
target_folder_name = 'detrend_all_zscore_some'
main_dir    = '/bcbl/home/home_n-z/nmei/MRI/uncon_feat/MRI/'
parent_dir  = '{}{}/func/'.format(main_dir,sub)
mask_dir    = '{}{}/anat/ROI_BOLD'.format(main_dir,sub)
BOLD_mask   = glob('{}{}/func/*/*/*/*/mask.nii.gz'.format(main_dir,sub))[0]
masks       = glob(os.path.join(mask_dir,'*.nii.gz'))
# 1: pick volumes -> signal_clearn; 2: signal_clean -> pick volumes; 3: separately
method      = 3

processed   = glob(os.path.join(parent_dir,
                                '*',
                                '*',
                                'outputs',
                                'func',
                                'ICAed_filtered',
                                '*.nii.gz'))
processed   = np.sort(processed)
events      = glob(os.path.join(parent_dir,
                                '*',
                                '*',
                                'outputs',
                                # f'{folder_name}',# comment this out when do normal
                                '*.csv'))
events      = np.sort(events)

left = np.sort([item for item in masks if 'lh' in item])
right = np.sort([item for item in masks if 'rh' in item])
unique_mask_names = [item.split('/')[-1].split('-')[2].split('_BOLD')[0] for item in left]

for average in [True]:
    if average:
        output_dir = '../../../data/BOLD_average_{}/{}/'.format(target_folder_name,
                                                 sub)
    else:
        output_dir = '../../../data/BOLD_no_average_{}/{}/'.format(target_folder_name,
                                                    sub)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    functional_mask = load_img(BOLD_mask)
    
    for mask_pick in unique_mask_names:
        mask_name   = [item for item in masks if mask_pick in item]
        # combine the left and right ROIs
        array_1 = np.asanyarray(load_img(mask_name[0]).dataobj)
        array_2 = np.asanyarray(load_img(mask_name[1]).dataobj)
        array_combined = array_1 + array_2
        array_combined[array_combined > 0] = 1
        array_combined[np.asanyarray(functional_mask.dataobj) == 0] = 0
        roi_mask_combined = new_img_like(functional_mask,array_combined)
        
        temp_BOLD   = []
        temp_event  = []
        for item,csv in zip(processed,events):
            print(item,csv,'\n')
            
            # do not perform any preprocessing when applying the masking
            masker      = NiftiMasker(mask_img          = roi_mask_combined,
                                      standardize       = False,
                                      detrend           = False,
                                      memory            = 'nilarn_cashed')
            BOLD        = masker.fit_transform(X        = item)
            
            df_concat   = load_csv(csv)
            # clean_signal works on ndarray only
            idx             = df_concat['volume_interest'] == 1
            if method == 1:
                processed_BOLD  = BOLD[idx]
                processed_df    = df_concat[idx]
                # preprocessing is applied on the picked volumes
                processed_BOLD  = clean_signal(processed_BOLD,
                                               t_r          = 0.85,
                                               detrend      = True,
                                               standardize  = True)
            elif method == 2:
                # preprocessing is applied on the all volumes
                processed_BOLD  = clean_signal(BOLD,
                                               t_r          = 0.85,
                                               detrend      = True,
                                               standardize  = True)
                processed_BOLD  = processed_BOLD[idx]
                processed_df    = df_concat[idx]
            elif method == 3:
                # detrending is applied on the all volumes
                processed_BOLD  = clean_signal(BOLD,
                                               t_r          = 0.85,
                                               detrend      = True,
                                               standardize  = False)
                # pick the volumes between 4 - 7 seconds
                processed_BOLD  = processed_BOLD[idx]
                processed_df    = df_concat[idx]
                # zscroing is applied on the picked volumes
                processed_BOLD  = clean_signal(processed_BOLD,
                                               t_r          = None,
                                               detrend      = False,
                                               standardize  = True,)
            
            if average:
                processed_BOLD,processed_df = groupby_average(
                        processed_BOLD, # all the volumes
                        processed_df.reset_index(), # just the volume of interest
                        ['id']) # groupby
                
            else:
                processed_BOLD, processed_df = processed_BOLD, processed_df
            
            temp_BOLD.append( processed_BOLD)
            temp_event.append(processed_df)
        BOLD        = np.concatenate(temp_BOLD)
        df_event    = pd.concat(temp_event).reset_index()
        
        np.save(os.path.join(output_dir,'{}_BOLD.npy'.format(mask_pick)),
                BOLD,)
        df_event.to_csv(os.path.join(output_dir,'{}_events.csv'.format(mask_pick)),
                        index = False)



try:
    rmtree('nilarn_cashed')
except:
    pass

















