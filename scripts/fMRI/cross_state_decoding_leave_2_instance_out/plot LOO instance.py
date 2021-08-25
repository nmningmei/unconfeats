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

working_dir = '../../../../results/MRI/nilearn/decoding_instance'
working_data = glob(os.path.join(working_dir,'*','*.csv'))

df = pd.concat([pd.read_csv(f) for f in working_data])
df['region'] = df['roi_name'].map(utils.define_roi_category())
df['ROI'] = df['roi_name'].map(utils.rename_ROI_for_plotting())
df['SUB'] = df['sub'].map(utils.subj_map())

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

sort_by = ['SUB',
           'conscious_state_source',
           'conscious_state_target',
           'x_order',
           ]
df_plot = df.sort_values(sort_by)

df_plot['star'] = df_plot['pval'].apply(utils.stars)
df_plot['roc_auc'] = df_plot['score']

fig,axes = plt.subplots(figsize = (7*5,7*5),
                        nrows = 7,
                        ncols = 3,
                        sharey = True,
                        )
for n in range(7):
    sub = f'sub-0{n+1}'
    
    df_plot_sub = df_plot[df_plot['sub'] == sub]
    
    ax = axes[n][0]
    idx = np.logical_and(df_plot_sub['conscious_state_source'] == 'unconscious',
                         df_plot_sub['conscious_state_target'] == 'unconscious')
    ax = sns.barplot(x = 'roi_name',
                     y = 'roc_auc',
                     data = df_plot_sub[idx],
                     ax = ax,
#                     order = x_order,
                     )
    ax.set(xlabel = '',
           ylabel = 'ROC AUC',
           xticklabels = [])
    
    df_stat_picked = df_plot_sub[idx]
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
    idx = np.logical_and(df_plot_sub['conscious_state_source'] == 'conscious',
                         df_plot_sub['conscious_state_target'] == 'conscious')
    ax = sns.barplot(x = 'roi_name',
                     y = 'roc_auc',
                     data = df_plot_sub[idx],
                     ax = ax,
#                     order = x_order,
                     )
    ax.set(xlabel = '',
           ylabel = '',
           xticklabels = [])
    
    df_stat_picked = df_plot_sub[idx]
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
    idx = np.logical_and(df_plot_sub['conscious_state_source'] == 'conscious',
                         df_plot_sub['conscious_state_target'] == 'unconscious')
    ax = sns.barplot(x = 'roi_name',
                     y = 'roc_auc',
                     data = df_plot_sub[idx],
                     ax = ax,
#                     order = x_order,
                     )
    ax.set(xlabel = '',
           ylabel = '',
           xticklabels = [])
    
    df_stat_picked = df_plot_sub[idx]
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
[ax.axvline(8.5,linestyle = '-', color = 'black',alpha = 0.5) for ax in axes.flatten()]

for ii,(ax,(sub,df_stat_sub)) in enumerate(zip(axes,df_plot.groupby(['SUB']))):
    if ii >=4:
        props = dict(boxstyle = 'round',facecolor = 'red',alpha = .5)
        ax[0].text(1,.75,f'Observer {ii+1}',fontsize = 32,bbox = props)
    else:
        props = dict(boxstyle = 'round',facecolor = None,alpha = .5)
        ax[0].text(1,.75,f'Observer {ii+1}',fontsize = 32,bbox = props)

