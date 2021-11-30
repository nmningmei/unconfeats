#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 16:28:26 2021

@author: nmei
"""
import os
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
sns.set_context('paper',font_scale=2)
# from matplotlib import rc
# rc('font',weight = 'bold')
# plt.rcParams['axes.labelsize'] = 45
# plt.rcParams['axes.labelweight'] = 'bold'
# plt.rcParams['axes.titlesize'] = 45
# plt.rcParams['axes.titleweight'] = 'bold'
# plt.rcParams['ytick.labelsize'] = 32
# plt.rcParams['xtick.labelsize'] = 32

working_dir     = '../../another_git/*/results'
figure_dir      = '../figures'
collect_dir     = '/export/home/nmei/nmei/properties_of_unconscious_processing/all_figures'
marker_factor   = 10
marker_type     = ['8','s','p','*','+','D','o']
alpha_level     = .75

if not os.path.exists(figure_dir):
    os.mkdir(figure_dir)

working_data = glob(os.path.join(working_dir,'*','*.csv'))

paper_dir = '/export/home/nmei/nmei/properties_of_unconscious_processing/stats'

df              = []
for f in working_data:
    if ('inception' not in f):
        temp                                    = pd.read_csv(f)
        k                                       = f.split('/')[-2]
        model,hidde_unit,hidden_ac,drop,out_ac  = k.split('_')
        if 'drop' not in temp.columns:
            temp['drop']                        = float(drop)
        df.append(temp)
df              = pd.concat(df)

n_noise_levels  = 50
noise_levels    = np.concatenate([[0],[item for item in np.logspace(-1,3,n_noise_levels)]])
x_map           = {round(item,9):ii for ii,item in enumerate(noise_levels)}
inverse_x_map   = {round(value,9):key for key,value in x_map.items()}
print(x_map,inverse_x_map)

df['x']         = df['noise_level'].round(9).map(x_map)
print(df['x'].values)

df['x']             = df['x'].apply(lambda x: [x + np.random.normal(0,0.1,size = 1)][0][0])
df                  = df.sort_values(['hidden_activation','output_activation'])
df['activations']   = df['hidden_activation'] + '-' +  df['output_activation']

idxs            = np.logical_or(df['model'] == 'CNN',df['model'] == 'linear-SVM')
df_plot         = df.loc[idxs,:]
df_plot         = df_plot.drop_duplicates(['model_name','hidden_units','hidden_activation',
                                          'output_activation','noise_level','dropout'])

df_plot['pooled_std'] = np.sqrt(df_plot['score_std'].values ** 2 + df_plot['chance_std'].values ** 2)
df_plot['effect_size'] = (df_plot['score_mean']  - df_plot['chance_mean']) / df_plot['pooled_std']
