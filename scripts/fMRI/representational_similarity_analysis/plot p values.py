#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 11:26:48 2020

@author: nmei
"""
import os
from glob import glob

import numpy as np

from nilearn.input_data import NiftiMasker
from nilearn.image      import new_img_like
from nibabel            import load as load_img
from nilearn.mass_univariate import permuted_ols
from nilearn.plotting import plot_stat_map

from itertools          import combinations
from matplotlib import pyplot as plt

from nipype.interfaces import fsl

from scipy import stats

model_names = os.listdir('../../../../results/MRI/nilearn/univariate_permutation')

for iii in [1,2,3,4,5,6,7]:
    for model_name in model_names:
        sub = 'sub-0{}'.format(iii)
        mask_dir = f'../../../../data/MRI/{sub}/anat/ROI_BOLD/'
        mask_file = glob(os.path.join(mask_dir,'combine_BOLD.nii.gz'))[0]
        univariate_test_dir = f'../../../../results/MRI/nilearn/univariate_permutation/{model_name}/{sub}'
        functional_brain = f'../../../../data/MRI/{sub}/func/example_func.nii.gz'
        standard_brain = '../../../../data/standard_brain/MNI152_T1_2mm_brain.nii.gz'
        transformation_dir_single = os.path.abspath(
                        f'../../../../data/MRI/{sub}/reg/example_func2standard.mat')
        standarded_dir = f'../../../../results/MRI/nilearn/RSA_searchlight_standarded/{model_name}/{sub}'
        
        
        figure_dir = f'../../../../figures/MRI/nilearn/RSA_3mm/{model_name}'
        if not os.path.exists(figure_dir):
            os.makedirs(figure_dir)
        conscious_state1 = 'conscious'
        conscious_state2 = 'unconscious'
        # transformation
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
                                title = f'Negative log10 P values, {sub} {conscious_state1} > {conscious_state2}\nthresholding by log p critical = {threshold:.1f}')
        
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
                                title = f'Negative log10 P values of positive correlation, {sub}\nunconscious against chance level')
        
        figure.subplots_adjust(wspace = 0,hspace = 0)
        
    #    figure.savefig(os.path.join(
    #        '/bcbl/home/home_n-z/nmei/properties_of_unconscious_processing/figures',
    #        f'RSA_searchlight_{sub}.png'),
    #        dpi = 400,
    #        bbox_inches = 'tight')
        figure.savefig(os.path.join(
            figure_dir,
            f'RSA_searchlight_{sub}.png'),
    #        dpi = 400,
            bbox_inches = 'tight')
        plt.close('all')