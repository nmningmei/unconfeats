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
        working_dir = f'../../../../results/MRI/nilearn/RSA_searchlight/{model_name}/{sub}'
        working_data = glob(os.path.join(working_dir,'*.nii.gz'))
        mask_dir = f'/bcbl/home/home_n-z/nmei/MRI/uncon_feat/MRI/{sub}/anat/ROI_BOLD/'
        mask_file = glob(os.path.join(mask_dir,'combine_BOLD.nii.gz'))[0]
        univariate_test_dir = f'../../../../results/MRI/nilearn/univariate_test/{model_name}/{sub}'
        standard_brain = '../../../../data/standard_brain/MNI152_T1_2mm_brain.nii.gz'
        transformation_dir_single = os.path.abspath(
                        f'../../../../data/MRI/{sub}/reg/example_func2standard.mat')
        standarded_dir = f'../../../../results/MRI/nilearn/RSA_searchlight_standarded/{model_name}/{sub}'
        figure_dir = f'../../../../figures/MRI/nilearn/RSA_searchlight_3mm/{model_name}'
        if not os.path.exists(standarded_dir):
            os.makedirs(standarded_dir)
        if not os.path.exists(figure_dir):
            os.mkdir(figure_dir)
        
        conscious_state1 = 'conscious'
        conscious_state2 = 'unconscious'
        
        conscious_z = os.path.abspath(os.path.join(univariate_test_dir,
                                                    'conscious_z.nii.gz'))
        unconscious_z = os.path.abspath(os.path.join(univariate_test_dir,
                                                    'unconscious_z.nii.gz'))
        diff_z = os.path.abspath(os.path.join(univariate_test_dir,
                                              'conscious_unconscious_z.nii.gz'))
        diff_p = os.path.abspath(os.path.join(univariate_test_dir,
                                          f'{conscious_state1}_{conscious_state2}.nii.gz'))
        unconscious_p = os.path.abspath(os.path.join(univariate_test_dir,
                                          'unconscious_zero.nii.gz'))
    
        # transformation
        for maps in [conscious_z,unconscious_z,diff_z,diff_p,unconscious_p]:
            flt = fsl.FLIRT()
            flt.inputs.in_file = maps
            flt.inputs.reference = os.path.abspath(standard_brain)
            flt.inputs.output_type = 'NIFTI_GZ'
            flt.inputs.in_matrix_file = transformation_dir_single
            flt.inputs.out_matrix_file = os.path.abspath(
                os.path.join(standarded_dir,f'{maps.split("/")[-1].split(".")[0]}_flirt.mat'))
            flt.inputs.out_file = os.path.abspath(
                os.path.join(standarded_dir,maps.split('/')[-1])
                )
            flt.inputs.apply_xfm = True
            res = flt.run()
        
        # template = load_mni152_template()
        # for maps in [conscious_z,unconscious_z,diff_z,diff_p,unconscious_p]:
        #     resampled_img = resample_to_img(load_img(maps),template)
        #     resampled_img.to_filename(os.path.abspath(
        #         os.path.join(standarded_dir,maps.split('/')[-1])
        #         ))
        
        conscious_z = os.path.abspath(os.path.join(standarded_dir,
                                                    'conscious_z.nii.gz'))
        unconscious_z = os.path.abspath(os.path.join(standarded_dir,
                                                    'unconscious_z.nii.gz'))
        diff_z = os.path.abspath(os.path.join(standarded_dir,
                                              'conscious_unconscious_z.nii.gz'))
        diff_p = os.path.abspath(os.path.join(standarded_dir,
                                          f'{conscious_state1}_{conscious_state2}.nii.gz'))
        unconscious_p = os.path.abspath(os.path.join(standarded_dir,
                                          'unconscious_zero.nii.gz'))
        
        print('plotting')
        figure = plt.figure(figsize = (16 * 2,5 * 3))
        
        cut_coords = 0,0,0
        threshold = 1e-3
        
        # conscious
        ax = figure.add_subplot(231)
        display = plot_stat_map(conscious_z,
                                standard_brain,
                                cmap = plt.cm.RdBu_r,
                                draw_cross = False,
                                threshold = threshold,
                                cut_coords = cut_coords,
                                axes = ax,
                                figure = figure,
                                title = f'{conscious_state1} positive z scores',)
        
        # unconscious
        ax = figure.add_subplot(232)
        display = plot_stat_map(unconscious_z,
                                standard_brain,
                                cmap = plt.cm.RdBu_r,
                                draw_cross = False,
                                threshold = threshold,
                                cut_coords = cut_coords,
                                axes = ax,
                                figure = figure,
                                title = f'{conscious_state2} positive z scores',)
        
        # conscious > unconscious
        ax = figure.add_subplot(233)
        display = plot_stat_map(diff_z,
                                standard_brain,
                                cmap = plt.cm.RdBu_r,
                                draw_cross = False,
                                cut_coords = cut_coords,
                                axes = ax,
                                threshold = threshold,
                                title = f'Positive difference of z scores between {conscious_state1} and {conscious_state2}')
        
        # p value: conscious > unconscious
        ax = figure.add_subplot(223)
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
                                title = f'Negative P values, {sub} {conscious_state1} > {conscious_state2}\nthresholding by log p critical = {threshold:.1f}')
        
        # p value: unconscious v.s. zero
        ax = figure.add_subplot(224)
        display = plot_stat_map(unconscious_p,
                                standard_brain,
                                threshold = threshold,
                                cmap = plt.cm.RdBu_r,
                                draw_cross = False,
                                cut_coords = cut_coords,
                                black_bg = True,
                                figure = figure,
                                axes = ax,
                                title = f'P values of positive correlation, {sub}\nunconscious against chance level')
        
        figure.subplots_adjust(wspace = 0,hspace = 0)
        
#        figure.savefig(os.path.join(
#            '/bcbl/home/home_n-z/nmei/properties_of_unconscious_processing/figures',
#            f'RSA_searchlight_{sub}_{model_name}.png'),
##            dpi = 400,
#            bbox_inches = 'tight')
        figure.savefig(os.path.join(
            figure_dir,
            f'RSA_searchlight_{sub}.png'),
#            dpi = 400,
            bbox_inches = 'tight')
        plt.close('all')






































