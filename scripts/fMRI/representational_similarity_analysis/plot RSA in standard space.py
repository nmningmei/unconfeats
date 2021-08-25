#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 07:15:41 2020

@author: nmei
"""
import os
import numpy as np
from glob import glob

from nipype.interfaces import fsl
from nilearn.plotting import plot_stat_map
from nilearn.datasets import load_mni152_template
from nilearn.image import new_img_like
from nibabel import load as load_fmri

from matplotlib import pyplot as plt
for model_name in os.listdir('../../../../results/MRI/nilearn/univariate_permutation'):
    for k in [1,2,3,4,5,6,7]:
        sub = f'sub-0{k}'
        mask_dir = f'/bcbl/home/home_n-z/nmei/MRI/uncon_feat/MRI/{sub}/anat/ROI_BOLD/'
        mask_file = glob(os.path.join(mask_dir,'combine_BOLD.nii.gz'))[0]
        univariate_test_dir = f'../../../../results/MRI/nilearn/univariate_permutation/{model_name}/{sub}'
        standard_brain = '../../../../data/standard_brain/MNI152_T1_2mm_brain.nii.gz'
        transformation_dir_single = os.path.abspath(
                        f'../../../../data/MRI/{sub}/reg/example_func2standard.mat')
        standarded_dri = f'../../../../results/MRI/nilearn/RSA_searchlight_standarded/{model_name}/{sub}'
        figure_dir = f'../../../../figures/MRI/nilearn/RSA_3mm/{model_name}'
        
        
        conscious_state1 = 'conscious'
        conscious_state2 = 'unconscious'
        
        conscious_z = os.path.abspath(os.path.join(standarded_dri,
                                                    'conscious_z.nii.gz'))
        unconscious_z = os.path.abspath(os.path.join(standarded_dri,
                                                    'unconscious_z.nii.gz'))
        diff_z = os.path.abspath(os.path.join(standarded_dri,
                                              'conscious_unconscious_z.nii.gz'))
        diff_p = os.path.abspath(os.path.join(standarded_dri,
                                          f'{conscious_state1}_{conscious_state2}.nii.gz'))
        unconscious_p = os.path.abspath(os.path.join(standarded_dri,
                                          'unconscious_zero.nii.gz'))
        
        print('plotting')
        figure = plt.figure(figsize = (16,5))
        
        cut_coords = 0,0,0
        threshold = 1e-3
        
        
        # p value: conscious > unconscious
        ax = figure.add_subplot(121)
        p_thres = 1e-3
        threshold = -np.log10(p_thres) # % corrected
        
        display = plot_stat_map(diff_p,
                                standard_brain,
                                threshold = threshold,
                                cmap = plt.cm.RdBu_r,
                                draw_cross = False,
                                cut_coords = cut_coords,
                                black_bg = True,
                                figure = figure,
                                axes = ax,
                                title = f'Log P values, {sub} {conscious_state1} > {conscious_state2}')
        
        # p value: unconscious v.s. zero
        ax = figure.add_subplot(122)
        display = plot_stat_map(unconscious_p,
                                standard_brain,
                                threshold = threshold,
                                cmap = plt.cm.RdBu_r,
                                draw_cross = False,
                                cut_coords = cut_coords,
                                black_bg = True,
                                figure = figure,
                                axes = ax,
                                title = f'Log P values unconscious > 0')
        
        figure.subplots_adjust(wspace = 0,hspace = 0)
        
        figure.savefig(os.path.join(
            figure_dir,
            f'pvalue_RSA_searchlight_{sub}.png'),
            # dpi = 400,
            bbox_inches = 'tight')
        plt.close('all')






































