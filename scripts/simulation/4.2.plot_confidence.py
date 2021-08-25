#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 11:33:55 2020
@author: nmei
"""

import os
from glob import glob

import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt

working_dir = '../confidence_results'
figure_dir = '../figures'
if not os.path.exists(figure_dir):
    os.mkdir(figure_dir)
working_data = glob(os.path.join(working_dir,'*','*.csv'))

df = pd.concat([pd.read_csv(f) for f in working_data if ('inception' not in f)])
x_map = {item:ii for ii,item in enumerate(pd.unique(df['noise_level']))}
df['x'] = df['noise_level'].map(x_map)

sns.barplot(x = 'x',
                y = 'confidence',
                data = df,
                )

# compute dprime


# compute meta-dprime






fig,axes = plt.subplots(figsize = (20,40),nrows = pd.unique(df['activations']).shape[0],
                        sharex = True,)
for ax,(activation,df_sub) in zip(axes.flatten(),df_CNN.groupby(['activations'])):
    for kk,(marker_size,(hidden_units,df_sub_sub)) in enumerate(zip(pd.unique(df['hidden_units']),df_sub.groupby(['hidden_units']))):
        ax = sns.stripplot(x = 'x',
                           y = 'confidence_mean',
                           hue = 'model',
                           size = np.log(marker_size * 3),
                           data = df_sub_sub,
                           ax = ax,
                           dodge = 0.1,
                           )
        ax.get_legend().remove()
        ax.axvline(33.5,linestyle = '--',color = 'black',alpha = 1.)
    _ = ax.set(ylabel = 'ROC AUC',xlabel = '',xticklabels = [],title = activation)
_ = ax.set(xlabel = 'Noise Level')
_ = ax.set_xticklabels([f'{key:.3f}' for key in x_map.keys()],
                        rotation = 45,
                        ha = 'center')
fig.savefig(os.path.join(figure_dir,'confidence.jpeg'),
            dpi = 300,
            bbox_inches = 'tight')
fig.savefig(os.path.join(figure_dir,'confidence (light).jpeg'),
#           dpi = 300,
            bbox_inches = 'tight')
