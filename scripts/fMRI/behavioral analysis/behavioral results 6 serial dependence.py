#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 11:58:29 2022

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

sns.set_style('white')
sns.set_context('paper',font_scale = 2)
copyfile('../../utils.py','utils.py')
import utils

re_run = False

paper_dir = '/export/home/nmei/nmei/properties_of_unconscious_processing/figures'
collect_dir = '/export/home/nmei/nmei/properties_of_unconscious_processing/all_figures'
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
    p_temp          = pd.read_csv(f).iloc[32:,:2]
    category        = p_temp['category'].values
    index           = p_temp['index'].values
    
    df_temp['sub']  = f.split('/')[-3]
    df_temp['session']  = int(index[np.where(category == 'session')[0]][0])
    df_temp['block']    = int(index[np.where(category == 'block')[0]][0])
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

df_povit = df[['sub','category','subcategory','label','session','block','order',
               'correctAns_raw','response.corr_raw','response.keys_raw','response.rt_raw',
               'visible.keys_raw','visible.rt_raw',
               'probe_Frames_raw',]]
df_povit.to_csv(os.path.join(saving_dir,'concat_df.csv'),index = False)

df_res = dict(response                  = [],
              rt_response               = [],
              visibility                = [],
              rt_visibility             = [],
              correct_answer            = [],
              category                  = [],
              subcategory               = [],
              label                     = [],
              # info of previous trial
              response_previous         = [],
              rt_response_previous      = [],
              visibility_previous       = [],
              rt_visibility_previous    = [],
              correct_answer_previous   = [],
              category_previous         = [],
              subcategory_previous      = [],
              label_previous            = [],
              # other info
              sub                       = [],
              session                   = [],
              block                     = [],
              )
for (sub,session,block),df_sub in df_povit.groupby(['sub','session','block']):
    df_sub = df_sub.sort_values(['order']).reset_index(drop = True)
    for ii,row in df_sub.iterrows():
        condition1 = ii != 0
        condition2 = False if ii == 0 else np.logical_not(np.isnan(df_sub.loc[ii - 1,'response.corr_raw']))
        condition3 = np.logical_not(np.isnan(row['response.corr_raw']))
        condition4 = False if ii == 0 else df_sub.loc[ii - 1,'visible.keys_raw'] != 99
        condition5 = row['visible.keys_raw'] != 99
        if np.logical_and.reduce((condition1,condition2,condition3,condition4,condition5)):
            previous_row = df_sub.iloc[ii - 1,:]
            df_res['response'].append(row['response.keys_raw'])
            df_res['rt_response'].append(row['response.rt_raw'])
            df_res['visibility'].append(row['visible.keys_raw'])
            df_res['rt_visibility'].append(row['visible.rt_raw'])
            df_res['correct_answer'].append(row['response.corr_raw'])
            df_res['category'].append(row['category'])
            df_res['subcategory'].append(row['subcategory'])
            df_res['label'].append(row['label'])
            
            df_res['response_previous'].append(previous_row['response.keys_raw'])
            df_res['rt_response_previous'].append(previous_row['response.rt_raw'])
            df_res['visibility_previous'].append(previous_row['visible.keys_raw'])
            df_res['rt_visibility_previous'].append(previous_row['visible.rt_raw'])
            df_res['correct_answer_previous'].append(previous_row['response.corr_raw'])
            df_res['category_previous'].append(previous_row['category'])
            df_res['subcategory_previous'].append(previous_row['subcategory'])
            df_res['label_previous'].append(previous_row['label'])
            
            df_res['sub'].append(sub)
            df_res['session'].append(session)
            df_res['block'].append(block)
df_res = pd.DataFrame(df_res)
df_res.to_csv(os.path.join(saving_dir,'previous_now.csv'),index = False)

# df_now                  = df_res[[item for item in df_res.columns if ('previous' not in item)]]
# df_now['trial']         = 'current'
# df_previous             = df_res[[item for item in df_res.columns if ('previous' in item)]]
# df_previous.columns     = [item[:-9] for item in df_previous.columns]
# df_previous['sub']      = df_now['sub'].copy()
# df_previous['session']  = df_now['session'].copy()
# df_previous['block']    = df_now['block'].copy()
# df_previous['trial']    = 'previous'
# df_res                  = pd.concat([df_now,df_previous]).reset_index(drop = True)

df_res['Category: current == previous'] = df_res['category'] == df_res['category_previous']
df_res['Response: current == previous'] = df_res['response'] == df_res['response_previous']
df_res['visibility_match']              = df_res['visibility'] == df_res['visibility_previous']
df_res['RT response']                   = df_res['rt_response'].copy()
df_res['RT visibility']                 = df_res['rt_visibility'].copy()

g = sns.catplot(x           = 'Response: current == previous',
                y           = 'RT response',
                hue         = 'Category: current == previous',
                row         = 'sub',
                col         = 'visibility_match',
                order       = [True,False],
                hue_order   = [True,False],
                col_order   = [True,False],
                data        = df_res,
                kind        = 'violin',
                aspect      = 2,
                sharex      = True,
                **dict(cut      = 0,
                       inner    = 'quartile',
                       split    = True,
                       )
                )
(g.set_titles("{row_name} | Visibility: current==previous? {col_name}"))
g.savefig(os.path.join(figure_dir,'RT as a function between 2 trials among many other parameters.jpg'),
          dpi           = 200,
          bbox_inches   = 'tight',
          )

g = sns.catplot(x           = 'Response: current == previous',
                y           = 'RT visibility',
                hue         = 'Category: current == previous',
                row         = 'sub',
                col         = 'visibility_match',
                order       = [True,False],
                hue_order   = [True,False],
                col_order   = [True,False],
                data        = df_res,
                kind        = 'violin',
                aspect      = 2,
                sharex      = True,
                **dict(cut      = 0,
                       inner    = 'quartile',
                       split    = True,
                       )
                )
(g.set_titles("{row_name} | Visibility: current==previous? {col_name}"))
g.savefig(os.path.join(figure_dir,'RT of visibility as a function between 2 trials among many other parameters.jpg'),
          dpi           = 200,
          bbox_inches   = 'tight',
          )













