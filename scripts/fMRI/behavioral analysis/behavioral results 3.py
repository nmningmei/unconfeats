#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 12:19:44 2021

@author: nmei

This script is to calculate d' and many other intermediate variables

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
from scipy           import stats

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


ylabel = "d'"
def dprime(y_true,y_pred,epsilon = 1e-12,return_rate = False):
    fpr,tpr,_ = metrics.roc_curve(y_true,y_pred)
    z_H,z_FA = stats.norm.ppf([tpr[1]+epsilon,fpr[1]+epsilon],)
    d_prime = z_H - z_FA
    if return_rate:
        return d_prime,fpr[1],tpr[1]
    return d_prime
def _chance(responses,correct_ans,return_rate = False):
    idx_                = np.random.choice(np.arange(len(responses)),
                                           len(responses),
                                           replace = True)
    random_responses    = shuffle(responses)
    return dprime(correct_ans[idx_],random_responses[idx_],return_rate = return_rate)

def _scoring(responses,correct_ans,return_rate = False):
    idx_                = np.random.choice(np.arange(len(responses)),
                                           len(responses),
                                           replace = True)
    return dprime(correct_ans[idx_],responses[idx_],return_rate = return_rate)


n_sim = int(1e4)
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

res = dict(sub = [],
           visibility = [],
           fpr = [],
           tpr = [],
           dprime = [],
           pval = [],
           chance = [],
           N = [],
           probe_mean = [],
           probe_std = [],
           )
for ((sub,vis),df_sub) in df.groupby(['sub','visible.keys_raw']):
    df_sub = df_sub.sort_values(['session','block','order'])
    correct_ans,responses = df_sub['correctAns_raw'].values-1, df_sub['response.keys_raw'].values-1
    fpr,tpr,_ = metrics.roc_curve(correct_ans,responses)
    np.random.seed(12345)
    gc.collect()
    temp                    = Parallel(n_jobs = -1,verbose = 1)(delayed(_scoring)(**{
                            'responses':responses,
                            'correct_ans':correct_ans,
                            'return_rate':True}) for i in range(n_sim))
    temp                    = np.array(temp)
    scores = temp[:,0]
    fpr = temp[:,1]
    tpr = temp[:,2]
    experiment              = dprime(correct_ans,responses)
    chance_level            = Parallel(n_jobs = -1,verbose = 1)(delayed(_chance)(**{
                            'responses':responses,
                            'correct_ans':correct_ans}) for i in range(n_sim))
    chance_level            = np.array(chance_level)
    pval                    = (np.sum(chance_level > np.nanmean(scores)) + 1) / (n_sim + 1)
    
    res['sub'].append(sub)
    res['visibility'].append(vis)
    res['fpr'].append(np.nanmean(fpr))
    res['tpr'].append(np.nanmean(tpr))
    res['dprime'].append(np.nanmean(scores))
    res['pval'].append(pval)
    res['chance'].append(np.nanmean(chance_level))
    res['N'].append(df_sub.shape[0])
    res['probe_mean'].append(np.nanmean(df_sub['probe_Frames_raw'].values*10))
    res['probe_std'].append(np.nanstd(df_sub['probe_Frames_raw'].values*10))

results = pd.DataFrame(res)
total = results['N'].values
results['total'] = np.repeat(total.reshape(-1,3).sum(1),3)
results['proportion'] = results['N'] / results['total']
results['report'] = [f'{x["probe_mean"]:.2f}+/-{x["probe_std"]:.2f}' for ii,x in results.iterrows()]

df_plot = results.copy()
temp = []
for vis,df_sub in df_plot.groupby(['sub']):
    df_sub     = df_sub.sort_values(['pval'])
    converter   = utils.MCPConverter(pvals = df_sub['pval'].values)
    d           = converter.adjust_many()
    df_sub['p_corrected'] = d['bonferroni'].values
    temp.append(df_sub)
df_plot = pd.concat(temp)
df_plot['Behavioral' ] = df_plot['pval'] < 0.05
df_plot['Behavioral' ] = df_plot['Behavioral'].map({True:'Above Chance',False:'At Chance'})
df_plot['awareness' ] = df_plot['visibility'].map({
        1:'Unconscious',
        2:'Glimpse',
        3:'Conscious',})
df_plot = df_plot.sort_values(['sub','visibility',])
df_plot['subject'] = df_plot['sub'].map(utils.subj_map())
df_plot = df_plot.sort_values(['subject','visibility',])
df_plot.to_csv(os.path.join(paper_dir.replace('figures','stats'),'dprime behavioral.csv'),index = False)

# metadprime
df_meta_dprime = pd.read_csv(os.path.join(paper_dir.replace('figures','stats'),'behavioral metad.csv'))

temp = []
for sub,df_sub in df_plot.groupby(['sub']):
    df_meta_sub = df_meta_dprime[df_meta_dprime['sub'] == sub]
    df_sub['meta dprime'] = df_meta_sub['metad'].values[0]
    df_sub['overall dprime'] = df_meta_sub['dprime'].values[0]
    df_sub['m_ratio'] = df_meta_sub['m_ratio'].values[0]
    df_sub['m_diff'] = df_meta_sub['m_diff'].values[0]
    temp.append(df_sub)
df_meta_d_added = pd.concat(temp)

df_meta_d_added.to_csv(os.path.join(paper_dir.replace('figures','stats'),
                                    'supplemental behavioral report.csv'),
          index = False)

fig,ax = plt.subplots(figsize = (16,12))
ax = sns.swarmplot(
                 x      = 'awareness',
                 y      = 'dprime',
                 hue    = 'Behavioral',
                 size   = 18,
                 data   = df_plot,
                 ax     = ax,
                 )
ax.set_xlabel('Conscious State')
ax.set_ylabel(ylabel)
ax.get_legend().set_title("")

fig.savefig(os.path.join(paper_dir,'dprime behavioral.jpg'),
            dpi = 300,
            bbox_inches = 'tight')















