#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 07:20:55 2020

@author: nmei
"""

import os
import gc

from glob  import glob
from tqdm  import tqdm
from scipy import stats

import pandas  as pd
import numpy   as np
import seaborn as sns

sns.set_style('whitegrid')
sns.set_context('poster',font_scale = 1.5)
from matplotlib import rc
rc('font',weight = 'bold')
from matplotlib import pyplot as plt
plt.rcParams['axes.labelsize'] = 45
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 45
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['ytick.labelsize'] = 32
plt.rcParams['xtick.labelsize'] = 32

from shutil import copyfile
copyfile('../../../../scripts/utils.py','utils.py')
import utils

figure_dir = '../../../../figures/MRI/nilearn/collection_of_results'
if not os.path.exists(figure_dir):
    os.mkdir(figure_dir)
    
df_plot,df_stat = [],[]
for k in [1,2,3,4,5,6,7]:
    sub                 = f'sub-0{k}'
    target_folder       = 'decoding'
    target_file         = '*csv'
    working_dir         = '../../../../results/MRI/nilearn/{}/{}'.format(target_folder,sub)
    working_data        = glob(os.path.join(working_dir,target_file))
    beha_dir            = '../../../../data/behavioral/{}'.format(sub)
    saving_dir          = '../../../../results/MRI/nilearn/{}/{}_stats'.format(sub,target_folder)
    
    df      = pd.concat([pd.read_csv(f) for f in working_data])
    n_folds = df['fold'].max()
    
    if 'model_name' not in df.columns:
        df['model_name']    = df['model']
    if 'score'          in df.columns:
        df['roc_auc']       = df['score']
    if 'roi_name'   not in df.columns:
        df['roi_name']      = df['roi']
    
    df['feature_selector']  = df['model_name'].apply(utils.get_fs)
    df['estimator']         = df['model_name'].apply(utils.get_clf)
    
    sort_by                 = ['sub', 
                               'roi', 
                               'model', 
                               'condition_target', 
                               'condition_source', 
                               'flip',
                               'language', 
                               'transfer', 
                               'model_name',
                               'feature_selector', 
                               'estimator', 
                               'roi_name',]
    df                      = df.sort_values(sort_by)
    df['mask'] = df['condition_target'] == df['condition_source']
    df_picked = df[df['mask'] == True]
    df_picked['sub'] = sub
    
    results = pd.read_csv(os.path.join(saving_dir,'decoding cross stats.csv'))
    results['mask'] = results['condition_target'] == results['condition_source']
    results_picked = results[results['mask'] == True]
    results_picked['sub'] = sub
    
    col = 'ps'
    temp = []
    for f,df_sub in results_picked.groupby(['feature_selector']):
        df_sub = df_sub.sort_values([col])
        converter = utils.MCPConverter(pvals = df_sub[col].values)
        d = converter.adjust_many()
        df_sub['ps_corrected'] = d['bonferroni'].values
        temp.append(df_sub)
    results_picked = pd.concat(temp)
    
    df_plot.append(df_picked)
    df_stat.append(results_picked)

df_plot = pd.concat(df_plot)
df_stat = pd.concat(df_stat)

df_plot['roi_name'] = df_plot['roi_name'].map(utils.rename_ROI_for_plotting())
df_stat['roi_name'] = df_stat['roi_name'].map(utils.rename_ROI_for_plotting())

feature_selector = 'None'
df_plot = df_plot[df_plot['feature_selector'] == feature_selector]
df_stat = df_stat[df_stat['feature_selector'] == feature_selector]

df_stat['stars'] = df_stat['ps_corrected'].map(utils.stars)

g = sns.catplot(x = 'roi_name',
                y = 'roc_auc',
                row = 'sub',
                col = 'condition_source',
                col_order = ['unconscious','glimpse','conscious'],
                data = df_plot,
                kind = 'bar',
                aspect = 3)
(g.set_axis_labels('ROIs','ROC AUC')
  .set_titles('{col_name} | {row_name}')
  .set(ylim=(0.35,.75)))
[ax.set_xticklabels(ax.xaxis.get_majorticklabels(),rotation = 90, ha = 'center',
                    # fontsize = 32
                    ) for ax in g.axes[-1]]
[ax.axhline(0.5,linestyle='--',color='k',alpha=0.5) for ax in g.axes.flatten()]
[ax.axvline(6.5,linestyle='-' ,color='k',alpha=0.5) for ax in g.axes.flatten()]
xtick_order = list(g.axes[-1][-1].xaxis.get_majorticklabels())

for n_row,k in enumerate([1,2,3,4,5,6,7]):
    sub = f'sub-0{k}'
    df_sub = df_stat[df_stat['sub'] == sub]
    for n_col, conscious_state in enumerate(['unconscious','glimpse','conscious']):
        ax = g.axes[n_row][n_col]
        
        for ii,text_obj in enumerate(xtick_order):
            position = text_obj.get_position()
            xtick_label = text_obj.get_text()
            rows = df_sub[df_sub['condition_source'] == conscious_state]
            rows = rows[rows['roi_name'] == xtick_label]
            if '*' in rows['stars'].values[0]:
                ax.annotate(rows['stars'].values[0],
                            xy = (position[0],0.7),
                            ha = 'center',
                            fontsize = 16,)

g.savefig(os.path.join(figure_dir,
                       'Decoding_LOO.jpeg'),
                    dpi = 400,
                    bbox_inches = 'tight')
g.savefig(os.path.join(figure_dir,
                       'Decoding_LOO_light.jpeg'),
#                    dpi = 400,
                    bbox_inches = 'tight')

paper_figure_dir = '/export/home/nmei/nmei/properties_of_unconscious_processing/figures'
g.savefig(os.path.join(paper_figure_dir,
                       'Decoding_LOO.jpeg'),
                    dpi = 400,
                    bbox_inches = 'tight')
g.savefig(os.path.join(paper_figure_dir,
                       'Decoding_LOO_light.jpeg'),
#                    dpi = 400,
                    bbox_inches = 'tight')


































