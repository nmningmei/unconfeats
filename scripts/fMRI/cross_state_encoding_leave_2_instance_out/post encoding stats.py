#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 08:19:37 2020

@author: nmei
"""

import os
import gc

from glob import glob
from tqdm import tqdm
from nilearn import input_data,plotting
from joblib import Parallel,delayed
from matplotlib import pyplot as plt
from nipype.interfaces import fsl

import numpy as np
import pandas as pd

def _temp_func(item):
    return masker.transform(item)
def _standardize(BOLD_native,standard_brain,transformation_dir_single,standarded_dir):
    flt = fsl.FLIRT()
    flt.inputs.in_file = BOLD_native
    flt.inputs.reference = os.path.abspath(standard_brain)
    flt.inputs.output_type = 'NIFTI_GZ'
    flt.inputs.in_matrix_file = transformation_dir_single
    flt.inputs.out_matrix_file = os.path.abspath(
        os.path.join(standarded_dir,f'{BOLD_native.split("/")[-1].split(".")[0]}_flirt.mat'))
    flt.inputs.out_file = BOLD_native.replace('native','standard')
    flt.inputs.apply_xfm = True
    flt.run()

standard_brain = '../../../../data/standard_brain/MNI152_T1_2mm_brain.nii.gz'
standard_mask = '../../../../data/standard_brain/MNI152_T1_2mm_brain_mask_dil.nii.gz'
figure_dir = f'../../../../figures/MRI/nilearn/encoding/'
if not os.path.exists(figure_dir):
    os.mkdir(figure_dir)
standard_dir = f'../../../../results/MRI/nilearn/encoding_standard'
if not os.path.exists(standard_dir):
    os.mkdir(standard_dir)

for ii_sub in [1,2,3,4,5,6,7]:
    sub = f'sub-0{ii_sub}'
    working_dir = f'../../../../results/MRI/nilearn/encoding/{sub}'
    mask_dir = f'../../../../data/MRI/{sub}/anat/ROI_BOLD/combine_BOLD.nii.gz'
    functional_brain = f'../../../../data/MRI/{sub}/func/example_func.nii.gz'
    standard_dir_single = os.path.join(standard_dir,sub)
    if not os.path.exists(standard_dir_single):
        os.mkdir(standard_dir_single)
    transformation_dir_single = os.path.abspath(
                    f'../../../../data/MRI/{sub}/reg/example_func2standard.mat')
    sources = ['conscious','conscious','unconscious']
    targets = ['conscious','unconscious','unconscious']
    masker = input_data.NiftiMasker(mask_dir,).fit()
    fig_native,axes_native = plt.subplots(figsize = (12,4*3),
                                          nrows = 3,
                                          )
    fig_standard,axes_standard = plt.subplots(figsize = (12,4*3),
                                              nrows = 3,
                                              )
    for ax_native,ax_standard,conscious_source,conscious_target in zip(axes_native.flatten(),axes_standard.flatten(),sources,targets):
        df = pd.read_csv(os.path.join(working_dir,f'{sub}_{conscious_source}_{conscious_target}.csv'))
        working_data = np.array([os.path.join(working_dir,f'{sub}_{conscious_source}_{conscious_target}_{ii}.nii.gz') for ii in df['fold'].values])
        gc.collect()
        BOLD_arrays = Parallel(n_jobs = -1, verbose = 1,)(delayed(_temp_func)(**{'item':item}) for item in working_data)
        gc.collect()
        BOLD_arrays = np.concatenate(BOLD_arrays,axis = 0)
        BOLD_arrays[0 > BOLD_arrays] = 0
        score = BOLD_arrays.mean(1).mean()
        BOLD_average = BOLD_arrays.mean(0)
        title = f'{sub}-{conscious_source}->{conscious_target}:{score:.4f}'
        plotting.plot_stat_map(masker.inverse_transform(BOLD_average),
                               functional_brain,
                               threshold=1e-2,
                               vmax = .05,
                               draw_cross = False,
                               cmap = plt.cm.Reds,
                               cut_coords = (0,0,0),
                               figure = fig_native,
                               axes = ax_native,
                               title = title,
                               )
        BOLD_native = os.path.abspath(os.path.join(standard_dir_single,f'{sub}_{conscious_source}_{conscious_target}_native.nii.gz'))
        masker.inverse_transform(BOLD_average).to_filename(BOLD_native)
        _standardize(BOLD_native,standard_brain,transformation_dir_single,standard_dir_single)
        temp_name = BOLD_native.replace('native','standard')
        temp_masker = input_data.NiftiMasker(standard_mask).fit()
        temp_array = temp_masker.transform(temp_name)
        plotting.plot_stat_map(temp_masker.inverse_transform(temp_array),
                               standard_brain,
                               threshold=1e-2,
                               vmax = .05,
                               draw_cross = False,
                               cmap = plt.cm.Reds,
                               cut_coords = (0,0,0),
                               figure = fig_standard,
                               axes = ax_standard,
                               title = title,
                               )
    fig_native.savefig(os.path.join(figure_dir,f'{sub}_native.jpg'),
                       dpi = 250,
                       bbox_inches = 'tight')
    fig_standard.savefig(os.path.join(figure_dir,f'{sub}_standard.jpg'),
                         dpi = 250,
                         bbox_inches = 'tight')
















