#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 10:09:31 2019

@author: nmei

this script takes each of the fMRI data paired with its baharioval psychophy file
and selects volumes of interests for stacking

"""

import os
import re
import pandas as pd
import numpy  as np
from glob               import glob
from tqdm               import tqdm
from nilearn.input_data import NiftiMasker
from nilearn.image      import index_img
from nilearn.image      import concat_imgs
from nilearn.signal     import clean as clean_signal
from shutil             import copyfile,rmtree
copyfile('../../utils.py','utils.py')
from utils              import groupby_average,load_csv

sub         = 'sub-01'
parent_dir  = '../../../data/MRI/{}/func/'.format(sub)
mask_dir    = '../../../data/MRI/{}/anat/ds_ROI_BOLD'.format(sub)
event_dir   = '../../../data/MRI/{}/func/*/*/'.format(sub)

masks       = glob(os.path.join(mask_dir,'*.nii.gz'))

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
                                '*.csv'))
events      = np.sort(events)

for average in [True,]:
    if average:
        output_dir = '../../../data/BOLD_average_test/{}/'.format(sub)
    else:
        output_dir = '../../../data/BOLD_no_average_test/{}/'.format(sub)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for mask_pick in masks:
        mask_name   = mask_pick.split('/')[-1].split('_BOLD')[0].replace('ctx-','')
        
        temp_BOLD   = []
        temp_event  = []
        for item,csv in zip(processed,events):
            print(item,csv,'\n')
            
            # do not perform any preprocessing when applying the masking
            masker      = NiftiMasker(mask_img          = os.path.abspath(mask_pick),
                                      standardize       = False,
                                      detrend           = False,
                                      memory            = 'nilarn_cashed')
            BOLD        = masker.fit_transform(X        = item)
            
            df_concat   = load_csv(csv)
            
            idx             = df_concat['volume_interest'] == 1
            processed_BOLD  = BOLD[idx]
            processed_df    = df_concat[idx]
            # preprocessing is applied on the picked volumes
            processed_BOLD  = clean_signal(processed_BOLD,
                                           t_r          = 0.85,
                                           detrend      = True,
                                           standardize  = True)
            
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
        
        np.save(os.path.join(output_dir,'{}_BOLD.npy'.format(mask_name)),
                BOLD,)
        df_event.to_csv(os.path.join(output_dir,'{}_events.csv'.format(mask_name)),
                        index = False)



try:
    rmtree('nilarn_cashed')
except:
    pass

















