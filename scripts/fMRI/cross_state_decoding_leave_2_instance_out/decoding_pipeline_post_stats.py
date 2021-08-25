#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 15:44:59 2021

@author: nmei
"""

import os
import gc
import utils

import pandas as pd
import numpy as np

from glob import glob
from tqdm import tqdm

from sklearn import metrics


def _gen(y_true):
    y_pred = [np.random.beta(2,2,item.shape) for item in y_true]
    return np.mean([metrics.roc_auc_score(a,b) for a,b in zip(y_true,y_pred)])

if __name__ == "__main__":
    folder_name = 'decoding_replicate'
    working_dir = f'../../../../results/MRI/nilearn/{folder_name}'
    working_data = glob(os.path.join(working_dir,'*','*_None_*.csv'))
    working_data = [item for item in working_data if 'glimpse' not in item]
    stats_dir = f'../../../../results/MRI/nilearn/decoding_stats/{folder_name}'
    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)
    
    n_permutations = int(1e4)
    metric_col = 'roc_auc'
    
    df = pd.concat([pd.read_csv(f) for f in tqdm(working_data)])
    missing_col_names = dict(roi_name='roi',
                             conscious_state_source='condition_source',
                             conscious_state_target='condition_target',)
    for key,val in missing_col_names.items():
        if key not in df.columns:
            df[key] = df[val]
    df['region'] = df['roi_name'].map(utils.define_roi_category())
    df['ROI'] = df['roi_name'].map(utils.rename_ROI_for_plotting())
    df['sub'] = df['sub'].map(utils.subj_map())
    
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
    if "new" in metric_col:
        groupby = ['sub','roi_name','conscious_state_source','conscious_state_target']
        temp = []
        for _attrs,df_sub in df.groupby(groupby):
            df_sub['n'] = df_sub['tn'] + df_sub['fp']+ df_sub['fn'] + df_sub['tp']
            N = df_sub['n'].sum()
            df_sub[metric_col] = df_sub['roc_auc'] * df_sub['n'] / N * df_sub.shape[0]
            temp.append(df_sub)
        df = pd.concat(temp)
    sort_by = ['sub',
               'conscious_state_source',
               'conscious_state_target',
               'x_order',
               ]
    df_plot = df.sort_values(sort_by)
    
    df_stat = dict(sub = [],
                   conscious_state_source = [],
                   conscious_state_target = [],
                   roi_name = [],
                   score_mean = [],
                   pval = [],)
    
    for _temp,df_sub in df_plot.groupby([
            'sub',
            'conscious_state_source',
            'conscious_state_target',
            'roi_name']):
        sub_name,conscious_state_source,conscious_state_target,roi_name = _temp
        df_experi = df_sub[df_sub['model'] == 'None + Linear-SVM']
        df_chance = df_sub[df_sub['model'] == 'None + Dummy']
        pval = utils.resample_ttest_2sample(df_experi[metric_col].values,
                                            df_chance[metric_col].values,
                                            n_permutation = n_permutations,
                                            n_ps = 10,
                                            n_jobs = -1,
                                            one_tail = True,
                                            )
        for ii,(key,_) in enumerate(df_stat.items()):
            if ii > 3:
                break
            else:
                df_stat[key].append(_temp[ii])
            
        df_stat['score_mean'].append(df_experi[metric_col].values)
        df_stat['pval'].append(np.mean(pval))
    
    df_stat = pd.DataFrame(df_stat)
    temp = []
    for _temp,df_sub in df_stat.groupby([
            'sub',
            'conscious_state_source',
            'conscious_state_target',]):
        df_sub = df_sub.sort_values(['pval'])
        converter = utils.MCPConverter(pvals = df_sub['pval'].values)
        d = converter.adjust_many()
        df_sub['p_corrected'] = d['bonferroni'].values
        temp.append(df_sub)
    df_stat = pd.concat(temp)
    df_stat.to_csv(os.path.join(stats_dir,'stats.csv'),index = False)




















