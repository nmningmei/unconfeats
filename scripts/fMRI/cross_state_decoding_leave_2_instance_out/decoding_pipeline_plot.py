#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 15:44:59 2021

@author: nmei
"""

import os
import utils

import pandas as pd
import numpy as np
import seaborn as sns

from glob import glob
from matplotlib import pyplot as plt

sns.set_context('poster',font_scale = 1.5)
from matplotlib import rc
rc('font',weight = 'bold')
plt.rcParams['axes.labelsize'] = 40
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 40
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['ytick.labelsize'] = 32
plt.rcParams['xtick.labelsize'] = 32

np.random.seed(12345)
# decoding_stratified,decoding_LOO_balanced,decoding_stratified_balanced
# decoding_LOO_balanced_glimpse
# decoding_loocv,decoding_loocv_balanced,decoding_loocv_balanced_glimpse
# decoding_13_11_2020
folder_name = 'decoding_stratified_95'
working_dir = f'../../../../results/MRI/nilearn/{folder_name}'
working_data = glob(os.path.join(working_dir,'*','*_None_Linear-SVM*.csv'))
stats_dir = f'../../../../results/MRI/nilearn/decoding_stats/{folder_name}'
figure_dir = f'../../../../figures/MRI/nilearn/{folder_name}'
if not os.path.exists(figure_dir):
    os.mkdir(figure_dir)
paper_dir = '/export/home/nmei/nmei/properties_of_unconscious_processing/stats'

df = pd.concat([pd.read_csv(f) for f in working_data])
missing_col_names = dict(roi_name='roi',
                         conscious_state_source='condition_source',
                         conscious_state_target='condition_target',)
for key,val in missing_col_names.items():
    if key not in df.columns:
        df[key] = df[val]
df['region'] = df['roi_name'].map(utils.define_roi_category())
df['ROI'] = df['roi_name'].map(utils.rename_ROI_for_plotting())
df['sub'] = df['sub'].map(utils.subj_map())

from scipy.stats import trim_mean
from functools import partial
plot_stats = True
metric_col = 'roc_auc'
use_median = False
figure_name = 'decoding_adjust.jpg' if 'new' in metric_col else 'decoding.jpg'
#figure_name = 'decoding_median.jpg' if use_median else 'decoding.jpg'
my_func = partial(trim_mean,proportiontocut = 0.05,)
est_func = np.mean if use_median else np.mean

x_order = ['Pericalcarine cortex',
           'Lingual',
           'Lateral occipital cortex',
           'Fusiform gyrus',
           'Inferior temporal lobe', 
           'Parahippocampal gyrus',
           'Precuneus',
           'Superior parietal gyrus',
           'Inferior parietal lobe',
           'Superior frontal gyrus',
           'Middle fontal gyrus',
           'Inferior frontal gyrus',]
x_order = {name:ii for ii,name in enumerate(x_order)}
df['x_order'] = df['ROI'].map(x_order)
if plot_stats:
    if folder_name == 'decoding_13_11_2020':
        df_stat = pd.read_csv(os.path.join(paper_dir,f'{folder_name}_stat.csv'))
        df_stat['pval'] = df_stat['p']
        df_stat['conscious_state_source'] = df_stat['condition_source']
        df_stat['conscious_state_target'] = df_stat['condition_target']
        df_stat['roi_name'] = df_stat['roi_name'].map({val:key for key,val in utils.rename_ROI_for_plotting().items()})
        temp = []
        for sub,df_sub in df_stat.groupby(['sub']):
            df_sub = df_sub.sort_values(['p'])
            converter = utils.MCPConverter(df_sub['p'].values)
            d = converter.adjust_many()
            df_sub['p_corrected'] = d['bonferroni'].values
            temp.append(df_sub)
        df_stat = pd.concat(temp)
        df_stat['sub']  = df_stat['sub'].map(utils.subj_map())
    else:
        df_stat = pd.read_csv(os.path.join(stats_dir,'stats.csv'))
    df_stat['star'] = df_stat['p_corrected'].apply(utils.stars)
    df_stat['ROI'] = df_stat['roi_name'].map(utils.rename_ROI_for_plotting())
    df_stat['region'] = df_stat['roi_name'].map(utils.define_roi_category())
    df_stat['x_order'] = df_stat['ROI'].map(x_order)
sort_by = ['sub',
           'region',
           'conscious_state_source',
           'conscious_state_target',
           'x_order',
           ]
df_plot = df.sort_values(sort_by)
if plot_stats:
    df_stat = df_stat.sort_values(sort_by)

if "new" in metric_col:
    groupby = ['sub','roi_name','conscious_state_source','conscious_state_target']
    temp = []
    for _attrs,df_sub in df_plot.groupby(groupby):
        df_sub['n'] = df_sub['tn'] + df_sub['fp']+ df_sub['fn'] + df_sub['tp']
        N = df_sub['n'].sum()
        df_sub[metric_col] = df_sub['roc_auc'] * df_sub['n'] / N * df_sub.shape[0]
        temp.append(df_sub)
    df_plot = pd.concat(temp)

# additional process for conscious and cross-consciousness decoding stats
df_stat_unconscious = df_stat[np.logical_and(df_stat['conscious_state_source'] == 'unconscious',
                                             df_stat['conscious_state_target'] == 'unconscious')].reset_index()
df_stat_conscious = df_stat[np.logical_and(df_stat['conscious_state_source'] == 'conscious',
                                           df_stat['conscious_state_target'] == 'conscious')].reset_index()
df_stat_cross  = df_stat[np.logical_and(df_stat['conscious_state_source'] == 'conscious',
                                        df_stat['conscious_state_target'] == 'unconscious')].reset_index()
font_color = []
for ii, row in df_stat_conscious.iterrows():
    if row['p_corrected'] > 0.05 and 0.05 > df_stat_cross.loc[ii,'p_corrected']:
        font_color.append('red')
    else:
        font_color.append('black')
df_stat_unconscious['font colors'] = 'black'
df_stat_conscious['font colors'] = 'black'
df_stat_cross['font colors'] = font_color
df_stat = pd.concat([df_stat_unconscious,df_stat_conscious,df_stat_cross])


df_plot = df_plot.sort_values(sort_by)
df_stat = df_stat.sort_values(sort_by)

fig,axes = plt.subplots(figsize = (7*5,7*5),
                        nrows = 7,
                        ncols = 3,
                        sharey = True,
                        sharex = True,
                        )
x_order = ['Pericalcarine cortex',
           'Lingual',
           'Lateral occipital cortex',
           'Fusiform gyrus',
           'Inferior temporal lobe', 
           'Parahippocampal gyrus',
           'Precuneus',
           'Superior parietal gyrus',
           'Inferior parietal lobe',
           'Superior frontal gyrus',
           'Middle fontal gyrus',
           'Inferior frontal gyrus',]
for n in range(7):
    sub = f'sub-0{n+1}'
    print(sub)
    df_plot_sub = df_plot[df_plot['sub'] == sub]
    if plot_stats:
        df_stat_sub = df_stat[df_stat['sub'] == sub]
    
    ax = axes[n][0]
    idx = np.logical_and(df_plot_sub['conscious_state_source'] == 'unconscious',
                         df_plot_sub['conscious_state_target'] == 'unconscious')
    ax = sns.barplot(x = 'roi_name',
                     y = metric_col,
                     data = df_plot_sub[idx],
                     ax = ax,
                     estimator = est_func,
                     )
    ax.set(xlabel = '',
           ylabel = 'ROC AUC',
           xticklabels = [])
    
    if plot_stats:
        idx_ = np.logical_and(df_stat_sub['conscious_state_source'] == 'unconscious',
                              df_stat_sub['conscious_state_target'] == 'unconscious')
        df_stat_picked = df_stat_sub[idx_]
        for ii,(roi_name,df_sub) in enumerate(df_stat_picked.groupby('x_order')):
            if df_sub['star'].values != 'n.s.':
                ax.text(x = ii,
                        y = .88,
                        s = df_sub['star'].values[0],
                        ha = 'center',
                        fontsize = 12,
                        color = df_sub['font colors'].values[0])
    
    if n == 0:
        ax.set(title = 'Unconscious')
    if n == 6:
        ax.set(xlabel = 'ROIs',)
        ax.set_xticklabels(x_order,rotation = 90)

    ax = axes[n][1]
    idx = np.logical_and(df_plot_sub['conscious_state_source'] == 'conscious',
                         df_plot_sub['conscious_state_target'] == 'conscious')
    ax = sns.barplot(x = 'roi_name',
                     y = metric_col,
                     data = df_plot_sub[idx],
                     ax = ax,
                     estimator = est_func,
                     )
    ax.set(xlabel = '',
           ylabel = '',
           xticklabels = [])
    
    if plot_stats:
        idx_ = np.logical_and(df_stat_sub['conscious_state_source'] == 'conscious',
                              df_stat_sub['conscious_state_target'] == 'conscious')
        df_stat_picked = df_stat_sub[idx_]
        for ii,(roi_name,df_sub) in enumerate(df_stat_picked.groupby('x_order')):
            if df_sub['star'].values != 'n.s.':
                ax.text(x = ii,
                        y = .88,
                        s = df_sub['star'].values[0],
                        ha = 'center',
                        fontsize = 12,
                        color = df_sub['font colors'].values[0])
        
    if n == 0:
        ax.set(title = 'Conscious')
    if n == 6:
        ax.set(xlabel = 'ROIs',)
        ax.set_xticklabels(x_order,rotation = 90)

    ax = axes[n][2]
    idx = np.logical_and(df_plot_sub['conscious_state_source'] == 'conscious',
                         df_plot_sub['conscious_state_target'] == 'unconscious')
    ax = sns.barplot(x = 'roi_name',
                     y = metric_col,
                     data = df_plot_sub[idx],
                     ax = ax,
                     )
    ax.set(xlabel = '',
           ylabel = '',
           xticklabels = [])
    
    if plot_stats:
        idx_ = np.logical_and(df_stat_sub['conscious_state_source'] == 'conscious',
                              df_stat_sub['conscious_state_target'] == 'unconscious')
        df_stat_picked = df_stat_sub[idx_]
        for ii,(roi_name,df_sub) in enumerate(df_stat_picked.groupby('x_order')):
            if df_sub['star'].values[0] != 'n.s.':
                ax.text(x = ii,
                        y = .88,
                        s = df_sub['star'].values[0],
                        ha = 'center',
                        fontsize = 12,
                        color = df_sub['font colors'].values[0])
                
        
    if n == 0:
        ax.set(title = 'Conscious --> Unconscious')
    
    if n == 6:
        ax.set(xlabel = 'ROIs',)
        ax.set_xticklabels(x_order,rotation = 90)
    
    ax.set(ylim = (0.35,0.95))
[ax.axhline(0.5,linestyle = '--',color = 'black',alpha = 0.5) for ax in axes.flatten()]
[ax.axvline(8.5,linestyle = '-', color = 'black',alpha = 0.5) for ax in axes.flatten()]

for ii,(ax,(sub,df_stat_sub)) in enumerate(zip(axes,df_plot.groupby(['sub']))):
    if ii >=4:
        props = dict(boxstyle = 'round',facecolor = 'red',alpha = .5)
        ax[0].text(1,.75,f'Observer {ii+1}',fontsize = 32,bbox = props)
    else:
        props = dict(boxstyle = 'round',facecolor = None,alpha = .5)
        ax[0].text(1,.75,f'Observer {ii+1}',fontsize = 32,bbox = props)

fig.savefig(os.path.join(figure_dir,figure_name),
#            dpi = 300,
            bbox_inches = 'tight')
