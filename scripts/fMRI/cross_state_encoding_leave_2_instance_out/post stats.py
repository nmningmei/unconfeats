#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 14:22:28 2020

@author: nmei
"""

import os
from glob import glob
from tqdm import tqdm

from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('whitegrid')
sns.set_context('poster',font_scale = 1.5)

figure_dir = '../../../../figures/MRI/nilearn/collection_of_results'
paper_dir = '/bcbl/home/home_n-z/nmei/properties_of_unconscious_processing/figures'

working_dir = '../../../../results/MRI/nilearn/encoding'
working_data = glob(os.path.join(working_dir,'*','*.csv'))

df = pd.concat([pd.read_csv(f) for f in working_data])


g = sns.catplot(x = 'sub_name',
                y = 'scores',
                data = df,
                seed = 12345,
                kind = 'bar',
                estimator = np.median,
                aspect = 3,
                row = 'condition_source',
                col = 'condition_target',
                row_order = ['unconscious','conscious'],
                col_order = ['unconscious','conscious'],
                )
(g.set_titles('{row_name} -> {col_name}')
  .set_axis_labels('Subjects','Ave. Scores'))

g.savefig(os.path.join(figure_dir,'encoding_scores.jpeg'),
          dpi = 400,
          bbox_inches = 'tight')
g.savefig(os.path.join(paper_dir,'encoding_scores.jpeg'),
          dpi = 400,
          bbox_inches = 'tight')

# plot the brain
brain_scores_dir = '../../../../results/MRI/nilearn/encoding'
brain_scores = np.sort(glob(os.path.join(brain_scores_dir,'*','*.npy')))
brain_scores = brain_scores.reshape(-1,4)
masks_dir = '../../../../data/MRI/*/func/mask.nii.gz'
masks = np.sort(glob(masks_dir))
funcs_dir = '../../../../data/MRI/*/func/example_func.nii.gz'
example_funcs = np.sort(glob(funcs_dir))
affine_dir = '../../../../data/MRI/*/reg/example_func2standard.nii.gz'
affines = np.sort(glob(affine_dir))

from nilearn.input_data import NiftiMasker
from nilearn.plotting import plot_stat_map
from nilearn.image import resample_to_img
from nilearn.datasets import load_mni152_template

standard_brain = load_mni152_template()

for brain_scores_,masks_,example_funcs_,affines_ in zip(
        brain_scores,masks,example_funcs,affines):
    masker = NiftiMasker(masks_,
                         standardize = False,
                         )
    masker.fit(example_funcs_)
    fig,axes = plt.subplots(figsize = (8 * 4,4 * 4),
                            nrows = 2,
                            ncols = 2,
                            )
    for idx,ss in zip([3,2,1,0],brain_scores_):
        title = ss.split('/')[-1].replace('.npy','')
        sub,_source,_target, = title.split('_')
        ax = axes.flatten()[idx]
        temp = np.load(ss)
        temp[0>temp] = 0
        temp_mean = temp.mean(0)
        # temp_mean[0 > temp_mean] = 0
        ss_ = masker.inverse_transform(temp_mean)
        ss_for_plot = resample_to_img(ss_,
                                      affines_)
        if sub == 'sub-07': threshold = 1e-2
        else: threshold = 2e-2
        
        plot_stat_map(ss_,
                      example_funcs_,
                      # cut_coords = (2,-1,20),
                      draw_cross = False,
                      colorbar = True,
                      cmap = plt.cm.coolwarm,
                      axes = ax,
                      threshold = threshold,
                       vmax = .2,
                      title = f'{_source} -> {_target}',
                      )
    
    fig.savefig(os.path.join(paper_dir,
                              f'encoding_{sub}_{_source}_{_target}.jpeg'),
                dpi = 400,
                bbox_inches = 'tight')






































