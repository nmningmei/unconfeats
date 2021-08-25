#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 12:19:44 2021

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

def A(y_true,y_pred):
    fpr,tpr,thres = metrics.roc_curve(y_true, y_pred)
    fpr = fpr[1]
    tpr = tpr[1]
    if  fpr > tpr:
        A = 1/2 + ((fpr - tpr)*(1 + fpr - tpr))/((4 * fpr)*(1 - tpr))
    elif fpr <= tpr:
        A = 1/2 + ((tpr - fpr)*(1 + tpr - fpr))/((4 * tpr)*(1 - fpr))
    return A
def compute_A(h,f):
    if (.5 >= f) and (h >= .5):
        a = .75 + (h - f) / 4 - f * (1 - h)
    elif (h >= f) and (.5 >= h):
        a = .75 + (h - f) / 4 - f / (4 * h)
    else:
        a = .75 + (h - f) / 4 - (1 - h) / (4 * (1 - f))
    return a
def process(y_true, y_pred):
    fpr,tpr,thresholds = metrics.roc_curve(y_true, y_pred)
    tpr = check_nan(tpr)
    fpr = check_nan(fpr)
    a = compute_A(tpr,fpr)
    return a
def check_nan(temp):
    if np.isnan(temp[1]):
        return 0
    else:
        return temp[1]
def score_func(y_true,y_pred):
    if y_true.max() > 1:
        y_true_ = np.array(y_true == 1,dtype = 'int')
        y_pred_ = np.array(y_pred == 1,dtype = 'int')
    else:
        y_true_ = y_true.copy()
        y_pred_ = y_pred.copy()
    return process(y_true_, y_pred_)
ylabel = "A'"

df              = []
for f in working_data:
    df_temp         = pd.read_csv(f).dropna()
    infos           = pd.read_csv(f).iloc[32:,:2]
    infos.index = infos['category']
    infos = infos['index'].T
    df_temp['sub']  = f.split('/')[-3]
    df_temp['session'] = infos['session']
    df_temp['block'] = infos['block']
    df.append(df_temp)
df                  = pd.concat(df)
for col in ['probe_Frames_raw',
            'response.keys_raw',
            'visible.keys_raw']:
    df[col] = df[col].apply(utils.str2int)


for ((sub),df_sub) in df.groupby(['sub','session','block']):#'visible.keys_raw']):
    df_sub = df_sub.sort_values(['session','block','order'])
    





















