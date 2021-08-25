#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 02:16:27 2020

@author: nmei
"""

import os
import re
from glob import glob
from tqdm import tqdm

import pandas  as pd
import numpy   as np
import seaborn as sns

import matplotlib
# matplotlib.pyplot.switch_backend('agg')

from matplotlib import pyplot as plt
from matplotlib.patches import Patch

sns.set_style('white')
sns.set_context('poster',font_scale = 1.5,)
from matplotlib import rc
rc('font',weight = 'bold')
plt.rcParams['axes.labelsize'] = 45
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 45
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['ytick.labelsize'] = 32
plt.rcParams['xtick.labelsize'] = 32
paper_dir = '/export/home/nmei/nmei/properties_of_unconscious_processing/figures'
pretrain_model_name     = 'mobilenet'
hidden_units            = 2
hidden_func_name        = 'relu'
hidden_dropout          = 0.
output_activation       = 'softmax'
working_dir     = f'../stability/{pretrain_model_name}_{hidden_units}_{hidden_func_name}_{hidden_dropout}_{output_activation}'
figure_dir      = '../figures'
marker_factor   = 10
marker_type     = ['8','s','p','*','+','D','o']
alpha_level     = .75

scores = np.sort(glob(os.path.join(working_dir,
                           'score*.npy')))
RDMs = np.sort(glob(os.path.join(working_dir,
                         'stability*.npy')))
features = np.sort(glob(os.path.join(working_dir,
                                     'feature*.npy')))
labels = np.sort(glob(os.path.join(working_dir,
                                   'label*.npy')))


df = dict(noise_level = [],
          score_mean = [],
          score_std = [],
          RDM_mean = [],
          RDM_std = [],
          RDM_max = [],
          RDM_min = [],
          )
for ii,(_score,_RDM,_feature,_label) in tqdm(enumerate(zip(scores,RDMs,features,labels))):
    noise_level = float(_RDM.split('/')[-1].split('_')[-1].replace('.npy',''))
    
    score = np.load(_score)
    RDM = np.load(_RDM)
    feature = np.load(_feature)
    label = np.load(_label)
    
    
    df['noise_level'].append(noise_level)
    df['score_mean'].append(score.mean())
    df['score_std'].append(score.std())
    df['RDM_mean'].append(RDM.mean())
    df['RDM_std'].append(RDM.std())
    df['RDM_max'].append(RDM.max())
    df['RDM_min'].append(RDM.min())
df = pd.DataFrame(df)
df = df.sort_values('noise_level')
df['x'] = np.arange(df.shape[0])

fig,ax = plt.subplots(figsize = (16,9))
ax.plot(df['x'].values,
        df['RDM_mean'].values,
        color = 'black',
        alpha = 1.,
        linestyle = '--',
        )
ax.fill_between(df['x'].values,
                df['RDM_mean'].values + df['RDM_std'].values,
                df['RDM_mean'].values - df['RDM_std'].values,
                color = 'red',
                alpha = alpha_level,
                )
ax2 = ax.twinx()
ax2.plot(df['x'].values,
         df['score_mean'].values,
         color = 'black',
         alpha = 1.,
         linestyle = '-',
         )
ax2.fill_between(df['x'].values,
                 df['score_mean'].values + df['score_std'].values,
                 df['score_mean'].values - df['score_std'].values,
                 color = 'blue',
                 alpha = alpha_level,
                 )
ax.set(ylim = (0.4,1.2),
       xticks = df['x'].values[::10],
       xticklabels = df['noise_level'].values[::10],
       xlabel = 'Noise level',
       ylabel = '1 - cosine')
ax2.set(ylim = (0.4,1.2),
        ylabel = 'ROC AUC')
fig.savefig(os.path.join(paper_dir,
                         'CNN_stability.jpeg'),
            dpi = 400,
            bbox_inches = 'tight')


































