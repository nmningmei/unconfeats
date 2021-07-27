#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 16:39:51 2019

@author: nmei
"""

import os
import re
import pandas as pd
import numpy  as np
from glob               import glob
from tqdm               import tqdm
from nilearn.input_data import NiftiMasker
from nilearn.image      import index_img
from nilearn.signal     import clean as clean_signal
from shutil             import copyfile,rmtree
copyfile('../../utils.py','utils.py')
from utils              import groupby_average,load_csv

sub         = 'sub-07'
folder_name = 'post_response'
target_folder_name = 'postresp'
first_session = 1
parent_dir  = '../../../data/MRI/{}/func/'.format(sub)
mask_dir    = '../../../data/MRI/{}/anat/ROI_BOLD'.format(sub)
event_dir   = '../../../data/MRI/{}/func/*/*/'.format(sub)

output_dir = '../../../data/BOLD_whole_brain_averaged_{}/{}'.format(target_folder_name,
                                                      sub)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

BOLD_to_file_name   = 'whole_brain.nii.gz'
csv_to_file_name    = 'whole_brain.csv'
whole_brain_mask    = os.path.join(parent_dir,
                                   f'session-0{first_session}/{sub}_unfeat_run-01/outputs/func/ICA_AROMA',
                                   'mask.nii.gz')

if  not os.path.exists(
        os.path.join(output_dir,
                     BOLD_to_file_name.replace('.nii.gz','_conscious.nii.gz'))):
    # if there is no data in the folder, we will create them, otherwise, don't waste our time
    processed   = glob(os.path.join(parent_dir,
                                    '*',
                                    '*',
                                    'outputs',
                                    'func',
                                    'ICAed_filtered',
                                    '*.nii.gz'))
    processed   = np.sort(processed)
    events      = np.sort(glob(os.path.join(parent_dir,
                                    '*',
                                    '*',
                                    'outputs',
                                    f'{folder_name}',
                                    '*.csv')))
    
    def extract_volumes(BOLD_file,csv_file,session_index):
        masker          = NiftiMasker(mask_img          = whole_brain_mask,
                                      standardize       = False,
                                      detrend           = False,
                                      memory            = 'nilarn_cashed')
        BOLD            = masker.fit_transform(X        = BOLD_file)
        df              = load_csv(csv_file)
        temp            = re.findall(r'\d+',csv_file)
        df['session']   = int(temp[-2])
        df['run']       = int(temp[-1])
        idx             = df['volume_interest'] == 1
        # just average them !!!
        BOLD,df         = groupby_average(BOLD[idx],df[idx].reset_index(),['id'])
        return BOLD,df,session_index
    
    temp        = [extract_volumes(f,c,idx) for idx,(f,c) in tqdm(enumerate(zip(processed,events)))]
    images      = [a for a,b,c in temp]
    dfs         = [b for a,b,c in temp]
    sessions    = [[c] * b.shape[0] for a,b,c in temp]
    
    sessions    = np.concatenate(sessions)
    stacked_images_whole_brain = np.concatenate(images)
    stacked_images_whole_brain = clean_signal(stacked_images_whole_brain,
                                              sessions = sessions,
                                              t_r = 0.85,
                                              standardize = True,
                                              detrend = True)
    df = pd.concat(dfs)
    masker          = NiftiMasker(mask_img          = whole_brain_mask,
                                  sessions          = np.arange(len(processed)),
                                  standardize       = False,
                                  detrend           = False,
                                  memory            = 'nilarn_cashed')
    masker.fit(processed)
    stacked_images_whole_brain = masker.inverse_transform(stacked_images_whole_brain)
    for conscious_state in ['unconscious','glimpse','conscious']:
        conditioned_mask = df['visibility'] == conscious_state
        indexed_image = index_img(stacked_images_whole_brain,conditioned_mask)
        indexed_image.to_filename(os.path.join(output_dir,
                    BOLD_to_file_name.replace('.nii.gz',f'_{conscious_state}.nii.gz')))
        df[conditioned_mask].to_csv(os.path.join(output_dir,
                               csv_to_file_name.replace('.csv',f'_{conscious_state}.csv')),
        index=False)


try:
    rmtree('nilarn_cashed')
except:
    pass
