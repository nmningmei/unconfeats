#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 05:39:24 2021

@author: nmei
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns

from utils_deep import hidden_activation_functions
from matplotlib import pyplot as plt

sns.set_style('whitegrid')
sns.set_context('poster')

working_dir = '../results/first_layer'
pretrain_model_name     = 'resnet50'
hidden_units            = 300
hidden_func_name        = 'selu'
hidden_activation       = hidden_activation_functions(hidden_func_name)
hidden_dropout          = 0.
patience                = 5
output_activation       = 'softmax'
model_saving_name       = f'{pretrain_model_name}_{hidden_units}_{hidden_func_name}_{hidden_dropout}_{output_activation}'
n_noise_levels          = 50

noise_levels    = np.concatenate([[0],[item for item in np.logspace(-1,3,n_noise_levels)]])
x_map           = {round(item,9):ii for ii,item in enumerate(noise_levels)}
inverse_x_map   = {round(value,9):key for key,value in x_map.items()}

df = pd.read_csv(os.path.join(working_dir,
                              model_saving_name,
                              'decodings.csv'))

df_cnn = df[['noise_level','cnn_score','cnn_pval']]
df_svm_cnn = df[['noise_level','svm_score_mean','svm_cnn_pval']]
df_svm_first = df[['noise_level','first_score_mean','svm_first_pval']]

def rename_col(df,new_col_name = '',jitter = 0):
    df.columns = ['noise_level','score','pval']
    df['type'] = new_col_name
    df['x'] = df['noise_level'].map(x_map) + jitter
    return df
df_cnn = rename_col(df_cnn,'FCNN')
df_svm_cnn = rename_col(df_svm_cnn,'SVM-hidden',0.25,)
df_svm_first = rename_col(df_svm_first,'SVM-first-layer',0.5)

df_plot = pd.concat([df_cnn,df_svm_cnn,df_svm_first])
df_plot['better than chance'] = 0.05 > df_plot['pval'] .values

fig,ax = plt.subplots(figsize = (12,8),
                      )
ax = sns.scatterplot(x = 'x',
                     y = 'score',
                     hue = 'type',
                     style = 'better than chance',
                     style_order = [True,False],
                     data = df_plot,
                     ax = ax,
                     )
ax.set(xticks = np.arange(0,51,10),
       xticklabels = noise_levels[::10].round(2),
       xlabel = 'Noise level',
       ylabel = 'ROC AUC',
       )
ax.axhline(0.5,linestyle = '--',color = 'black',alpha = 0.5,)



