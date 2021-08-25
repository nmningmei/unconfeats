#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 07:14:40 2020

@author: nmei
"""

import os
import gc

from glob import glob
from tqdm import tqdm
from shutil import copyfile
copyfile('../../../utils.py','utils.py')

import numpy as np

from nilearn import plotting,image
from nilearn.input_data import NiftiMasker
from matplotlib import pyplot as plt
from utils import plot_stat_map

standard_brain = '../../../../data/standard_brain/MNI152_T1_2mm_brain.nii.gz'

for sub in [1,2,3,4,5,6,7]:
    figure_dir = f'../../../../figures/MRI/nilearn/decode_searchlight/sub-0{sub}'
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)
    mask_dir = f'../../../../data/MRI/sub-0{sub}/anat/ROI_BOLD/'
    mask_file = glob(os.path.join(mask_dir,'combine_BOLD.nii.gz'))[0]
    original_brain = f'../../../../data/MRI/sub-0{sub}/func/example_func.nii.gz'
    # decoding scores
    working_dir = f'../../../../results/MRI/nilearn/decode_searchlight_LOSO/sub-0{sub}'
    
    masker = NiftiMasker(mask_img = mask_file,)
    
    # unconscious -> unconscious
    unconscious = image.mean_img(np.sort(glob(os.path.join(
                working_dir,
                f'unconscious_unconscious*SVM*.nii.gz'))))
    
    # conscious -> conscious
    conscious = image.mean_img(np.sort(glob(os.path.join(
                working_dir,
                f'conscious_conscious*SVM*.nii.gz'))))
    
    # conscious -> unconscious
    generalization = image.mean_img(np.sort(glob(os.path.join(
                working_dir,
                f'conscious_unconscious*SVM*.nii.gz'))))
    
    
    fig,axes = plt.subplots(figsize = (16,5 * 3),
                            nrows = 3)
    
    ax = axes[0]
    plot_stat_map(unconscious,
                  bg_img = original_brain,
                  draw_cross = False,
                  axes = ax,
                  vmin_ = .3,
                  vmax = .7,
                  threshold = .5,
                  cmap = plt.cm.coolwarm,
                  colorbar = True,
                  symmetric_cbar = False,
                  title = 'Decode unconscious',)
    ax = axes[1]
    plot_stat_map(conscious,
                  bg_img = original_brain,
                  draw_cross = False,
                  axes = ax,
                  vmin_ = .3,
                  vmax = .7,
                  threshold = .5,
                  cmap = plt.cm.coolwarm,
                  colorbar = True,
                  symmetric_cbar = False,
                  title = 'Decode conscious',)
    ax = axes[2]
    plot_stat_map(generalization,
                  bg_img = original_brain,
                  draw_cross = False,
                  axes = ax,
                  vmin_ = .3,
                  vmax = .7,
                  threshold = .5,
                  cmap = plt.cm.coolwarm,
                  colorbar = True,
                  symmetric_cbar = False,
                  title = 'Generalize conscious to unconscious',)
    fig.savefig(os.path.join(figure_dir,
                             'decoding_scores.jpg'),
                bbox_inches = 'tight')
    # p values
    working_dir = f'../../../../results/MRI/nilearn/decoding_searchlight_standarded/sub-0{sub}'
    
    # unconscious -> unconscious
    unconscious_ = glob(os.path.join(working_dir,
                '*_unconscious_unconscious.nii.gz'))[0]
    # conscious -> conscious
    conscious_ = glob(os.path.join(working_dir,
                '*_conscious_conscious.nii.gz'))[0]
    
    # conscious -> unconscious
    generalization_ = glob(os.path.join(working_dir,
                '*_conscious_unconscious.nii.gz'))[0]
    
    fig,axes = plt.subplots(figsize = (16,5 * 3),
                            nrows = 3)
    
    ax = axes[0]
    plotting.plot_glass_brain(unconscious_,
                  black_bg = True,
                  draw_cross = False,
                  axes = ax,
                  vmin = 0,
                  vmax = 5,
                  threshold = 1.3,
                  cmap = plt.cm.coolwarm,
                  colorbar = True,
                  symmetric_cbar = False,
                  title = 'Decode unconscious',)
    ax = axes[1]
    plotting.plot_glass_brain(conscious_,
                  black_bg = True,
                  draw_cross = False,
                  axes = ax,
                  vmin = 0,
                  vmax = 5,
                  threshold = 1.3,
                  cmap = plt.cm.coolwarm,
                  colorbar = True,
                  symmetric_cbar = False,
                  title = 'Decode conscious',)
    ax = axes[2]
    plotting.plot_glass_brain(generalization_,
                  black_bg = True,
                  draw_cross = False,
                  axes = ax,
                  vmin = 0,
                  vmax = 5,
                  threshold = 1.3,
                  cmap = plt.cm.coolwarm,
                  colorbar = True,
                  symmetric_cbar = False,
                  title = 'Generalize conscious to unconscious',)
    fig.savefig(os.path.join(figure_dir,
                             'pvalues.jpg'),
                bbox_inches = 'tight')































