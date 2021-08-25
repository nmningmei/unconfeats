#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 03:26:22 2020

@author: nmei
"""

import os
from glob import glob
from tqdm import tqdm

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns

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
working_dir = "../../../../results/MRI/nilearn/decoding_13_11_2020/sub*"
figure_dir = '../../../../figures/MRI/nilearn/collection_of_results_PCA'
if not os.path.exists(figure_dir):
    os.mkdir(figure_dir)

paper_dir = '/export/home/nmei/nmei/properties_of_unconscious_processing/stats'

n_permutation = int(1e5)

file_names = [item for item in glob(os.path.join(working_dir,'*.csv')) if ('PCA' in item)]

df = []
for f in tqdm(file_names):
    df_temp = pd.read_csv(f)
    f = f.split('/')[-1].replace('(','').replace(')','').replace('.csv','')
    sub,roi,source,target,feature_selector,estimator = f.split('_')
    df_temp['feature_selector'] = feature_selector
    df_temp['estimator'] = estimator
    if 'model_name' not in df_temp.columns:
        df_temp['model_name']    = df_temp['model']
    if 'score'          in df_temp.columns:
        df_temp['roc_auc']       = df_temp['score']
    if 'roi_name'   not in df_temp.columns:
        df_temp['roi_name']      = df_temp['roi']
    df.append(df_temp)
df_plot = pd.concat(df)

df_plot['roi_name'] = df_plot['roi_name'].map(utils.rename_ROI_for_plotting())
df_plot['region'] = df_plot['roi'].map(utils.define_roi_category())

sort_by = ['sub','roi_name','region','condition_target','condition_source',
           'feature_selector','estimator']
inverse_dict = {value:key for key,value in utils.rename_ROI_for_plotting().items()}

stat_file_name = 'decoding_stat.csv'
if True:#not os.path.exists(os.path.join(paper_dir,stat_file_name)):
    df_chance = df_plot[df_plot['estimator'] == 'Dummy'].sort_values(sort_by)
    df_decode = df_plot[df_plot['estimator'] != 'Dummy'].sort_values(sort_by)
    
    df_stat = dict(sub = [],
                   roi_name = [],
                   condition_source = [],
                   condition_target = [],
                   p = [],
                   )
    
    groupby = ['sub','roi_name','condition_source','condition_target']
    for (attrs,df_chance_sub),(_,df_decode_sub) in zip(
                        df_chance.groupby(groupby),
                        df_decode.groupby(groupby)):
        chance = df_chance_sub['roc_auc'].values
        scores = df_decode_sub['roc_auc'].values
        
        diff = scores - chance
        
        null = diff - diff.mean()
        null_dist = np.array([np.random.choice(null,size = null.shape[0],replace = True).mean() for _ in tqdm(range(n_permutation),
                                                                                                              desc = f'{attrs[0]} {attrs[1]} {attrs[2]}->{attrs[3]}')])
        p = (np.sum(null_dist >= diff.mean()) + 1) / (n_permutation + 1)
        
        df_stat['sub'].append(attrs[0])
        df_stat['roi_name'].append(attrs[1])
        df_stat['condition_source'].append(attrs[2])
        df_stat['condition_target'].append(attrs[3])
        df_stat['p'].append(p)
    df_stat = pd.DataFrame(df_stat)
#    df_stat.to_csv(os.path.join(paper_dir,
#                                stat_file_name),
#                   index = False)
    
else:
    df_stat = pd.read_csv(os.path.join(paper_dir,stat_file_name))


temp = []
for sub,df_sub in df_stat.groupby(['sub']):
    df_sub = df_sub.sort_values(['p'])
    converter = utils.MCPConverter(df_sub['p'].values)
    d = converter.adjust_many()
    df_sub['ps_corrected'] = d['bonferroni'].values
    temp.append(df_sub)
df_stat = pd.concat(temp)

df_stat['star'] = df_stat['ps_corrected'].apply(utils.stars)



x_order = ['Lateral occipital cortex',
           'Pericalcarine cortex',
           'Fusiform gyrus', 
           'Inferior parietal lobe',
           'Superior parietal gyrus',
           'Precuneus',
           'Lingual',
           'Parahippocampal gyrus',
           'Inferior temporal lobe',
           'Inferior frontal gyrus',
           'Middle fontal gyrus',
           'Superior frontal gyrus',]
x_order = {name:ii for ii,name in enumerate(x_order)}

df_plot['x_order'] = df_plot['roi_name'].map(x_order)
df_stat['x_order'] = df_stat['roi_name'].map(x_order)

sort_by = ['sub',
#           'roi_name',
           'condition_source',
           'condition_target',
           'x_order',
           ]

df_plot = df_plot.sort_values(sort_by)
df_plot = df_plot[df_plot['estimator'] != 'Dummy']
df_stat = df_stat.sort_values(sort_by)

fig,axes = plt.subplots(figsize = (7*5,7*5),
                        nrows = 7,
                        ncols = 3,
                        sharey = True,
                        )
for n in range(7):
    sub = f'sub-0{n+1}'
    
    df_plot_sub = df_plot[df_plot['sub'] == sub]
    df_stat_sub = df_stat[df_stat['sub'] == sub]
    
    ax = axes[n][0]
    idx = np.logical_and(df_plot_sub['condition_source'] == 'unconscious',
                         df_plot_sub['condition_target'] == 'unconscious')
    ax = sns.barplot(x = 'roi_name',
                     y = 'roc_auc',
                     data = df_plot_sub[idx],
                     ax = ax,
#                     order = x_order,
                     )
    ax.set(xlabel = '',
           ylabel = 'ROC AUC',
           xticklabels = [])
    
    idx_ = np.logical_and(df_stat_sub['condition_source'] == 'unconscious',
                          df_stat_sub['condition_target'] == 'unconscious')
    df_stat_picked = df_stat_sub[idx_]
    for ii,(roi_name,df_sub) in enumerate(df_stat_picked.groupby('x_order')):
        if df_sub['star'].values != 'n.s.':
            ax.annotate(df_sub['star'].values[0],
                        xy = (ii,0.88),
                        ha = 'center',
                        fontsize = 12,)
    
    if n == 0:
        ax.set(title = 'Unconscious')
    if n == 6:
        ax.set(xlabel = 'ROIs',)
        ax.set_xticklabels(x_order,rotation = 90)

    ax = axes[n][1]
    idx = np.logical_and(df_plot_sub['condition_source'] == 'conscious',
                         df_plot_sub['condition_target'] == 'conscious')
    ax = sns.barplot(x = 'roi_name',
                     y = 'roc_auc',
                     data = df_plot_sub[idx],
                     ax = ax,
#                     order = x_order,
                     )
    ax.set(xlabel = '',
           ylabel = '',
           xticklabels = [])
    
    idx_ = np.logical_and(df_stat_sub['condition_source'] == 'conscious',
                          df_stat_sub['condition_target'] == 'conscious')
    df_stat_picked = df_stat_sub[idx_]
    for ii,(roi_name,df_sub) in enumerate(df_stat_picked.groupby('x_order')):
        if df_sub['star'].values != 'n.s.':
            ax.annotate(df_sub['star'].values[0],
                        xy = (ii,0.88),
                        ha = 'center',
                        fontsize = 12,)
        
    if n == 0:
        ax.set(title = 'Conscious')
    if n == 6:
        ax.set(xlabel = 'ROIs',)
        ax.set_xticklabels(x_order,rotation = 90)

    ax = axes[n][2]
    idx = np.logical_and(df_plot_sub['condition_source'] == 'conscious',
                         df_plot_sub['condition_target'] == 'unconscious')
    ax = sns.barplot(x = 'roi_name',
                     y = 'roc_auc',
                     data = df_plot_sub[idx],
                     ax = ax,
#                     order = x_order,
                     )
    ax.set(xlabel = '',
           ylabel = '',
           xticklabels = [])
    
    idx_ = np.logical_and(df_stat_sub['condition_source'] == 'conscious',
                          df_stat_sub['condition_target'] == 'unconscious')
    df_stat_picked = df_stat_sub[idx_]
    for ii,(roi_name,df_sub) in enumerate(df_stat_picked.groupby('x_order')):
        if df_sub['star'].values != 'n.s.':
            ax.annotate(df_sub['star'].values[0],
                        xy = (ii,0.88),
                        ha = 'center',
                        fontsize = 12,)
        
    if n == 0:
        ax.set(title = 'Conscious --> Unconscious')
    
    if n == 6:
        ax.set(xlabel = 'ROIs',)
        ax.set_xticklabels(x_order,rotation = 90)
    
    ax.set(ylim = (0.35,0.95))
[ax.axhline(0.5,linestyle = '--',color = 'black',alpha = 0.5) for ax in axes.flatten()]
[ax.axvline(6.5,linestyle = '-', color = 'black',alpha = 0.5) for ax in axes.flatten()]

fig.savefig(os.path.join(paper_dir.replace('stats','figures'),
                         'LOO_specific_pca.jpeg'),
            dpi = 450,
            bbox_inches = 'tight')
fig.savefig(os.path.join(paper_dir.replace('stats','figures'),
                         'LOO_specific_pca_light.jpeg'),
            # dpi = 450,
            bbox_inches = 'tight')































