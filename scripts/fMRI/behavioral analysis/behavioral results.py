#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 12:36:38 2019

@author: nmei

this script is to access the behavioral statistics of the uncon_feat
fMRI experiment for each of the subjects
The statistical significance is estimated by a permutation test


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

if re_run:
    n_sim       = int(1e5)
    n_sample    = int(2e3)
    df_kplot    = []
    df_perceptual_learning = []
    results     = dict(pval         = [],
                       sub          = [],
                       accuracy     = [],
                       chance_mean  = [],
                       chance_std   = [],
                       visibility   = [],
                       pp1          = [],
                       pp2          = [],
                       frame_mean   = [],
                       frame_std    = [],
                       RT_mean      = [],
                       RT_std       = [],)
    for sub,df_sub in df.groupby(['sub']):
        frames,_,df_stat,df_concat = utils.get_frames(directory = os.path.join(
            working_dir,sub),
            new = True,EEG = False)
        
        for col in ['probe_Frames_raw',
                    'response.keys_raw',
                    'visible.keys_raw']:
            df_sub[col]     = df_sub[col].apply(utils.str2int)
        for visibility,df_sub_vis in df_sub.groupby(['visible.keys_raw']):
            correct_ans     = df_sub_vis['correctAns_raw'].values.astype('int32')
            responses       = df_sub_vis['response.keys_raw'].values.astype('int32')
            score           = score_func(correct_ans,responses)
            
            experiment      = score
            np.random.seed(12345)
            gc.collect()
            def _chance(responses,correct_ans):
                idx_                = np.random.choice(np.arange(len(responses)),
                                                       len(responses),
                                                       replace = True)
                random_responses    = shuffle(responses)
                return score_func(correct_ans[idx_],random_responses[idx_])
            
            chance_level            = Parallel(n_jobs = -1,verbose = 1)(delayed(_chance)(**{
                                    'responses':responses,
                                    'correct_ans':correct_ans}) for i in range(n_sim))
            chance_level            = np.array(chance_level)
            pval                    = (np.sum(chance_level > experiment) + 1) / (n_sim + 1)
            results['sub'           ].append(sub)
            results['pval'          ].append(pval)
            results['accuracy'      ].append(experiment)
            results['chance_mean'   ].append(chance_level.mean())
            results['chance_std'    ].append(chance_level.std() / np.sqrt(n_sim))
            results['visibility'    ].append(visibility)
            
            row = df_stat[df_stat['conscious_state'] == visibility]
            results['pp1'           ].append(row['prob_press_1'].values[0])
            results['pp2'           ].append(row['prob_press_2'].values[0])
            results['frame_mean'    ].append(row['frame_mean'].values[0])
            results['frame_std'     ].append(row['frame_std'].values[0])
            results['RT_mean'       ].append(row['RT_mean'].values[0])
            results['RT_std'        ].append(row['RT_std'].values[0])
            
            temp = defaultdict()
            gc.collect()
            def _scoring(responses,correct_ans):
                idx_                = np.random.choice(np.arange(len(responses)),
                                                       len(responses),
                                                       replace = True)
                return score_func(correct_ans[idx_],responses[idx_])
            
            scores                  = Parallel(n_jobs = -1,verbose = 1)(delayed(_scoring)(**{
                                    'responses':responses,
                                    'correct_ans':correct_ans}) for i in range(n_sim))
            scores                  = np.array(scores)
            temp['scores']          = scores
            temp['chance']          = chance_level
            tep                     = pd.DataFrame(temp)
            temp['sub']             = sub
            temp['visibility']      = visibility
            temp = pd.DataFrame(temp)
            df_kplot.append(temp)
            df_perceptual_learning.append(df_concat)
    results     = pd.DataFrame(results)
    results.to_csv(os.path.join(saving_dir,'accuracy as a function of visibility.csv'),index = False)
    results.to_csv(os.path.join('/bcbl/home/home_n-z/nmei/properties_of_unconscious_processing/stats',
                                'behavioral accuracy as a function of visibility.csv'))
    df_kplot    = pd.concat(df_kplot)
    df_kplot.to_csv(os.path.join('/bcbl/home/home_n-z/nmei/properties_of_unconscious_processing/stats',
                                'behavioral_accuracy_kplot.csv'))
    df_perceptual_learning = pd.concat(df_perceptual_learning)
    df_perceptual_learning.to_csv(os.path.join('/bcbl/home/home_n-z/nmei/properties_of_unconscious_processing/stats',
                                'perceptual learning concat.csv'))
else:
    results = pd.read_csv(os.path.join('/bcbl/home/home_n-z/nmei/properties_of_unconscious_processing/stats',
                                'behavioral accuracy as a function of visibility.csv'))
    df_kplot = pd.read_csv(os.path.join('/bcbl/home/home_n-z/nmei/properties_of_unconscious_processing/stats',
                                'behavioral_accuracy_kplot.csv'))
    df_perceptual_learning = pd.read_csv(os.path.join('/bcbl/home/home_n-z/nmei/properties_of_unconscious_processing/stats',
                                'perceptual learning concat.csv'))

results['sub'] = results['sub'].map(utils.subj_map())
df_kplot['sub'] = df_kplot['sub'].map(utils.subj_map())
df_perceptual_learning['sub'] = df_perceptual_learning['sub'].map(utils.subj_map())

df_kplot['visibility' ] = df_kplot['visibility'].map({
        1:'Unconscious',
        2:'Glimpse',
        3:'Conscious',})

print('plotting')
df_plot     = results.copy()
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
df_plot['visibility' ] = df_plot['visibility'].map({
        1:'Unconscious',
        2:'Glimpse',
        3:'Conscious',})
df_plot = df_plot.sort_values(['visibility','sub'],ascending = False)
df_plot.to_csv(os.path.join('/bcbl/home/home_n-z/nmei/properties_of_unconscious_processing/stats',
                            'behavioral_results.csv'),
               index = False)

fig,ax = plt.subplots(figsize = (16,12))
ax = sns.swarmplot(
                 x      = 'visibility',
                 y      = 'accuracy',
                 hue    = 'Behavioral',
                 size   = 18,
                 data   = df_plot,
                 ax     = ax,
                 )
ax.set_xlabel('Conscious State')
ax.set_ylabel(ylabel)
ax.get_legend().set_title("")
fig.savefig(os.path.join(figure_dir,
                         'behaviroal.jpeg'),
            dpi = 400,
            bbox_inches = 'tight')
fig.savefig(os.path.join('/bcbl/home/home_n-z/nmei/properties_of_unconscious_processing/figures',
                         'performance_conscious.png'),
            dpi = 400,
            bbox_inches = 'tight')

df_melt = pd.melt(df_kplot,
                  id_vars = ['sub','visibility'],
                  value_vars = ['scores','chance'])

fig,axes = plt.subplots(figsize = (15,15),
                        nrows = 3)
title_dict = {'Unconscious':'Trials rated as unaware',
              'Glimpse':'Trials rated as glimpse',
              'Conscious':'Trials rated as aware'}
for (visibility,df_sub),ax in zip(df_melt.groupby(['visibility'],sort = False),
                                  axes.flatten()):
    print(visibility)
    df_temp = df_plot[df_plot['visibility'] == visibility].reset_index(drop=True)
    df_temp = df_temp.sort_values(['sub']).reset_index(drop = True)
    df_sub = df_sub.sort_values(['sub']).reset_index(drop = True)
    df_temp['stars'] = df_temp['pval'].apply(utils.stars)
    ax = sns.violinplot(x = 'sub',
                        y = 'value',
                        hue = 'variable',
#                        order = np.sort(pd.unique(df_temp['sub'])),
                        data = df_sub,
                        ax = ax,
                        split = True,
                        inner = 'quartile',
                        cut = 0,
                        scale = 'width',
                        )
    for ii,row in df_temp.iterrows():
        text = row['stars']
        if text != "n.s.":
            ax.annotate(text,
                        xy = (ii,1.05),
                        ha = 'center',)
    handles,labels = ax.get_legend_handles_labels()
    if visibility == 'Unconscious':
        ax.legend(loc = 'upper left',)
    else:
        ax.get_legend().remove()
    ax.set(xticklabels = [],ylabel = ylabel,xlabel = '',
           ylim = (0,1.3),title = title_dict[visibility])
ax.set(xticklabels = np.arange(1,8),xlabel = 'Observer')
#fig.legend(handles,
#           labels,
#           loc = 'center right',
#           bbox_to_anchor = (1.11,.45),
#           )
fig.savefig(os.path.join(figure_dir,
                         'behaviroal_new.jpeg'),
            dpi = 400,
            bbox_inches = 'tight')
fig.savefig(os.path.join('/bcbl/home/home_n-z/nmei/properties_of_unconscious_processing/figures',
                         'performance_by_conscious.jpeg'),
            dpi = 400,
            bbox_inches = 'tight')

for column_name in ['visible.keys_raw','response.keys_raw']:
    df_perceptual_learning[column_name] = df_perceptual_learning[column_name].apply(utils.str2int)

df_for_plot = dict(
        session = [],
        sub = [],
        vis = [],
        score = [],
        duration = [],
        )
for (session,sub,vis),df_sub in df_perceptual_learning.groupby(['session','sub','visible.keys_raw']):
    correct_ans     = df_sub['correctAns_raw'].values.astype('int32')
    responses       = df_sub['response.keys_raw'].values.astype('int32')
    score = score_func(correct_ans,responses)
    df_for_plot['session'].append(session)
    df_for_plot['sub'].append(sub)
    df_for_plot['vis'].append(vis)
    df_for_plot['score'].append(score)
    df_for_plot['duration'].append(df_sub['probeFrames'].values.mean())
df_for_plot = pd.DataFrame(df_for_plot)
df_for_plot['visibility' ] = df_for_plot['vis'].map({
        1:'Unconscious',
        2:'Glimpse',
        3:'Conscious',})

observer_map = {f'sub-0{ii+1}':f'{ii+1}' for ii in range(7)}
df_for_plot['Observer'] = df_for_plot['sub'].map(observer_map)
g = sns.catplot(x = 'session',
                y = 'score',
                hue = 'Observer',
                hue_order = [f'{ii+1}' for ii in range(7)],
                row = 'visibility',
                data = df_for_plot,
                kind = 'point',
                aspect = 2,
                sharey = False,
                markers = '.',
                )
(g.set_axis_labels('Day',"A'")
  .set_titles('{row_name}'))
g.axes.flatten()[-1].set_xticklabels([f'{ii + 1}' for ii in range(7)],)
g.savefig(os.path.join(figure_dir,
                         'performance_by_day_conscious.jpeg'),
            dpi = 400,
            bbox_inches = 'tight')
g.savefig(os.path.join('/bcbl/home/home_n-z/nmei/properties_of_unconscious_processing/figures',
                         'performance_by_day_conscious.png'),
            dpi = 400,
            bbox_inches = 'tight')

g = sns.catplot(x = 'session',
                y = 'duration',
                hue = 'sub',
                hue_order = [f'sub-0{ii+1}' for ii in range(7)],
                row = 'visibility',
                data = df_for_plot,
                kind = 'point',
                aspect = 2,
                sharey = True,
                markers = '.',
                )
(g.set_axis_labels('Day',"Duration")
  .set_titles('{row_name}'))
g.axes.flatten()[-1].set_xticklabels([f'{ii + 1}' for ii in range(7)],)
g.savefig(os.path.join(figure_dir,
                         'duration_by_day_conscious.jpeg'),
            dpi = 400,
            bbox_inches = 'tight')
g.savefig(os.path.join('/bcbl/home/home_n-z/nmei/properties_of_unconscious_processing/figures',
                         'duration_by_day_conscious.png'),
            dpi = 400,
            bbox_inches = 'tight')

"""
# parsing RT based on correct incorrect
df['visibility'] = df['visible.keys_raw'].apply(utils.str2int).map({
        1:'Unconscious',
        2:'Glimpse',
        3:'Conscious',})
df['x'] = df['visible.keys_raw'].apply(utils.str2int).apply(
        lambda x: x + np.random.normal(0,0.1))

df['rt'] = df['response.rt_raw']
df['correct'] = df['response.corr_raw']

df.to_csv(os.path.join(saving_dir,'score.csv'),index = False)


from statsmodels.stats.anova import AnovaRM
res = AnovaRM(data = df, depvar = 'rt',subject = 'sub',
              within = ['visibility','correct'],
              aggregate_func = 'mean').fit()
print(res.anova_table)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit,permutation_test_score
from sklearn.base import clone
logistic = LogisticRegression(class_weight = 'balanced',
                              random_state = 12345)
# logistic = SVC(class_weight = 'balanced',random_state = 12345)
cv = StratifiedShuffleSplit(n_splits = 300,test_size = 0.2, random_state = 12345)
results = dict(sub = [],
               visibility = [],
               score_mean = [],
               pval = [],
               )
for (sub,visibility), df_sub in df.groupby(['sub','visibility']):
    df_sub
    clf = clone(logistic)
    X = df_sub['rt'].values.reshape(-1,1)
    y = df_sub['correct'].values.reshape(-1,1).ravel()
    res = permutation_test_score(clf, X,y,
                                 cv = cv,
                                 n_permutations = int(5e2),
                                 scoring = 'roc_auc',
                                 n_jobs = -1,
                                 verbose = 1,
                                 )
    results['sub'].append(sub)
    results['score_mean'].append(res[0])
    results['pval'].append(res[-1])
    results['visibility'].append(visibility)
results = pd.DataFrame(results)
results = results.sort_values('pval')
converter = utils.MCPConverter(pvals = results['pval'].values)
d = converter.adjust_many()
for col in d.columns[1:]:
    results[col] = d[col].values
results = results.sort_values(['sub','visibility'])
results.to_csv(os.path.join(saving_dir,'correct as RT.csv'),index = False)

fig = plt.figure(figsize = (12,12),)
corrs = [int(f'33{ii+1}') for ii in range(7)]
ax1 = fig.add_subplot(331)
for corr,(sub,df_sub) in zip(corrs,df.groupby('sub')):
    ax = fig.add_subplot(corr,sharex = ax1)
    ax = sns.violinplot(x = 'visibility',
                        y = 'response.rt_raw',
                        hue = 'response.corr_raw',
                        hue_order = [1,0],
                        data = df_sub,
                        palette = ['blue','red'],
                        ax = ax,
                        cut = 0,
                        split = True,
                        inner = 'quartile',
                        )
    ax.set(xticks = [0,1,2],
           xticklabels = ['Unconscious',
                          'Glimpse',
                          'Conscious'],
           xlabel = '',
           ylabel = 'Raction Time (sec)',
           title = f'{sub}'
           )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend().remove()
    ax.set(xlabel = 'Conscious State')
fig.tight_layout()
fig.legend(handles,
           ['Correct','Incorrect'],
           loc = (0.4,0.2),
           borderaxespad = 0.1,
           )
fig.savefig(os.path.join(figure_dir,
                         'Reaction as function of conscious split by correction.jpeg'),
        dpi = 300,
        bbox_inches = 'tight')



results = dict(sub = [],
               visibility = [],
               correct = [],
               rt = [],
               )
for (sub,correct,visibility), df_sub in df.groupby(['sub','response.corr_raw','visibility']):
    temp = df_sub.mean()
    results['sub'].append(sub)
    results['visibility'].append(visibility)
    results['correct'].append(correct)
    results['rt'].append(temp['rt'])
results = pd.DataFrame(results)
results.to_csv(os.path.join(saving_dir,'RT as function of subject, correct, and visibility.csv'),index=False)
"""





















