#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 06:04:20 2021

@author: nmei
"""
import os
import gc
from glob import glob

import numpy   as np
import pandas  as pd
import seaborn as sns

from sklearn         import metrics
from sklearn.utils   import shuffle
from matplotlib      import pyplot as plt
from joblib          import Parallel,delayed
from shutil          import copyfile
from collections     import defaultdict

sns.set_style('whitegrid')
sns.set_context('poster',font_scale = 1.3)
copyfile('../../utils.py','utils.py')
import utils

from matplotlib import rc
rc('font',weight = 'bold')
plt.rcParams['axes.labelsize'] = 40
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 40
#plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['ytick.labelsize'] = 32
plt.rcParams['xtick.labelsize'] = 32

re_run = False

paper_dir = '/export/home/nmei/nmei/properties_of_unconscious_processing/figures'

figure_dir      = '../../../figures/MRI/nilearn/behavioral'
if not os.path.exists(figure_dir):
    os.mkdir(figure_dir)
working_dir     = '../../../data/behavioral'
working_data    = glob(os.path.join(working_dir,'sub-0*','*','*.csv'))
saving_dir = '../../../results/MRI/nilearn/behavioral'
if not os.path.exists(saving_dir):
    os.mkdir(saving_dir)
    
df              = []
for f in working_data:
    df_temp         = pd.read_csv(f).iloc[:32,:]
    df_temp['sub']  = f.split('/')[-3]
    numerical_columns   = ['probe_Frames_raw',
                           'response.keys_raw',
                           'visible.keys_raw',]
    for col_name in numerical_columns:
        try:
            df_temp[col_name]    = df_temp[col_name].apply(utils.extract)
        except:
            df_temp[col_name]    = df_temp['probeFrames_raw'].apply(utils.extract)
    df.append(df_temp)
df                  = pd.concat(df)

res = dict(sub = [],
           visibility = [],
           probe_frame_mean = [],
           probe_frame_std = [],
           )
for (sub,visibility),df_sub in df.groupby(['sub','visible.keys_raw']):
    if visibility != 99:
        res['sub'].append(sub)
        res['visibility'].append(visibility)
        res['probe_frame_mean'].append(np.mean(df_sub['probe_Frames_raw'].values*10))
        res['probe_frame_std'].append(np.std(df_sub['probe_Frames_raw'].values*10))
res = pd.DataFrame(res)
res['subject'] = res['sub'].map(utils.subj_map())
res= res.sort_values(['subject','visibility'])
res['report'] = [f'{x["probe_frame_mean"]:.2f}+/-{x["probe_frame_std"]:.2f}' for ii,x in res.iterrows()]
res.to_csv(os.path.join(paper_dir.replace('figures','stats'),'probe frames.csv'),index = False)
