#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 07:06:50 2021

@author: nmei
"""

import os

import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('whitegrid')
sns.set_context('poster',font_scale = 1.5)
from matplotlib import rc
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
rc('font',weight = 'bold')
plt.rcParams['axes.labelsize'] = 45
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 45
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['ytick.labelsize'] = 32
plt.rcParams['xtick.labelsize'] = 32

working_dir = '../results/first_layer_only'
figure_dir      = '../figures'
paper_dir = '/export/home/nmei/nmei/properties_of_unconscious_processing/figures'
marker_factor   = 10
marker_type     = ['8','s','p','*','+','D','o']
alpha_level     = .75
dict_folder = dict(Resnet50 = 'resnet50',
                   VGG19 = 'vgg19_bn')
n_noise_levels  = 50
noise_levels    = np.concatenate([[0],[item for item in np.logspace(-1,3,n_noise_levels)]])
x_map           = {round(item,9):ii for ii,item in enumerate(noise_levels)}
inverse_x_map   = {round(value,9):key for key,value in x_map.items()}

dfs = []
df_chance = []
for model_name,folder_name in dict_folder.items():
    df              = pd.read_csv(os.path.join(working_dir,folder_name,'decodings.csv'))
    df['x']         = df['noise_level'].round(9).map(x_map)
    df['x_id']      = df['noise_level'].round(9).map(x_map)
    df['x']         = df['x'].apply(lambda x: [x + np.random.normal(0,0.1,size = 1)][0][0])
    df_chance.append(df)
    df_plot = pd.melt(df,id_vars = ['model_name',
                                    'noise_level',
                                    'x',],
                      value_vars = ['cnn_score',
                                    'first_score_mean',],
                      value_name = 'ROC AUC')
    temp = pd.melt(df,id_vars = ['model_name',
                                 'noise_level',
                                 'x',],
                      value_vars = ['cnn_pval',
                                    'svm_first_pval',])
    df_plot['model_name'] = model_name
    df_plot['pvals'] = temp['value'].values.copy()
    df_plot['Type'] = df_plot['variable'].apply(lambda x: x.split('_')[0].upper())
    df_plot['Type'] = df_plot['Type'].map({'CNN':'CNN',
                                           'FIRST':'Decode first layer'})
    dfs.append(df_plot)
df_plot = pd.concat(dfs)
df_chance = pd.concat(df_chance)




df_chance_plot = df_chance[np.logical_and(df_chance['cnn_pval'] > 0.05,
                                          0.05 > df_chance['svm_first_pval'])]
df_chance_melt_plot = pd.melt(df_chance_plot,
                              id_vars = ['model_name','noise_level','x','x_id',],
                              value_vars = ['cnn_score','first_score_mean'],
                              value_name = 'ROC AUC',)
df_chance_melt_plot['Type'] = df_chance_melt_plot['variable'].apply(lambda x: x.split('_')[0].upper())
df_chance_melt_plot['Type'] = df_chance_melt_plot['Type'].map({'CNN':'CNN',
                                       'FIRST':'Decode first layer'})
df_chance_melt_plot['model_name'] = df_chance_melt_plot['model_name'].map({val:key for key,val in dict_folder.items()})

xargs = dict(x           = 'x',
             y           = 'ROC AUC',
             hue         = 'Type',
             palette     = sns.xkcd_palette(['black','blue']),
             alpha       = alpha_level,
             s           = 200)
fig,axes = plt.subplots(figsize = (25,16),
                        nrows = 2,
                        ncols = 2,
                        sharex = False,
                        sharey = True,
                        )
for ii,(axes_row,model_picked) in enumerate(zip(axes,['VGG19','Resnet50'])):
    df_plot_row = df_plot[df_plot['model_name'] == model_picked]
    df_chance_plot_row = df_chance_melt_plot[df_chance_melt_plot['model_name'] == model_picked]
    # left
    ax = axes_row[0]
    ax = sns.scatterplot(data = df_plot_row,
                         ax = ax,
                         **xargs)
    handles,labels = ax.get_legend_handles_labels()
    ax.legend([],[],frameon = False)
    if ii > 0:
        ax.set(xlabel = 'Noise level',)
    else:
        ax.set(xlabel = '')
    xticks = ax.get_xticks()
    ax.set(xticks = [0,np.max(xticks)-8],
           xticklabels = [0,df_plot['noise_level'].max(),],
           title = model_picked)
    # right
    ax = axes_row[1]
    ax = sns.scatterplot(data = df_chance_plot_row,
                         ax = ax,
                         **xargs,)
    handles,labels = ax.get_legend_handles_labels()
    ax.legend([],[],frameon = False)
    if ii > 0:
        ax.set(xlabel = 'Noise level',)
    else:
        ax.set(xlabel = '')
    xticks = ax.get_xticks()
    ax.set(xticks = [0,np.max(xticks)],
           xticklabels = [0,df_chance_melt_plot['noise_level'].max().round(3)],
           title = model_picked)
    
[ax.axhline(0.5,
            linestyle       = '--',
            color           = 'black',
            alpha           = 1.,
            lw              = 1,
            ) for ax in axes.flatten()]
fig.legend(handles,labels,loc = (0.65,0.8),)


fig.savefig(os.path.join(paper_dir,'decoding first layer and chance level.jpg'),
          dpi = 300,
          bbox_inches = 'tight')



























