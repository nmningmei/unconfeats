#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 12 13:13:12 2019

@author: nmei
"""

import os
import gc

from glob  import glob
from tqdm  import tqdm
from scipy import stats

import pandas  as pd
import numpy   as np
import seaborn as sns

sns.set_style('whitegrid')
sns.set_context('poster',font_scale = 1.5)

from shutil import copyfile
copyfile('../../../utils.py','utils.py')
import utils

sub                 = 'sub-07'
target_folder       = 'decoding'
target_file         = '*csv'
working_dir         = '../../../../results/MRI/nilearn/{}/{}'.format(target_folder,sub)
working_data        = glob(os.path.join(working_dir,target_file))
beha_dir            = '../../../../data/behavioral/{}'.format(sub)
results,summary_ ,_ = utils.get_frames(beha_dir,EEG = False)
saving_dir          = '../../../../results/MRI/nilearn/{}/{}_stats'.format(sub,target_folder)
figure_dir          = '../../../../figures/MRI/nilearn/{}/{}'.format(sub,target_folder)

"""
# check missing
df = dict(
    roi = [],
    source = [],
    target = [],
    model = [],
    )
for a in working_data:
    a = a.replace('(','').replace(')','')
    a = a.split('/')[-1]
    sub_name,roi,source,target,feature_selector,estimator = a.split('_')
    model = f'{feature_selector}_{estimator}'
    df['roi'].append(roi)
    df['source'].append(source)
    df['target'].append(target)
    df['model'].append(model)
df = pd.DataFrame(df)
df = df.sort_values(['roi','source','target','model'])
for roi,df_sub in df.groupby(['roi']):
    print(roi,df_sub.shape)
"""

for d in [saving_dir,figure_dir]:
    if not os.path.exists(d):
        os.makedirs(d)

with open(os.path.join(saving_dir,'report.txt'),'w') as f:
    f.write(summary_)
    f.close()

df      = pd.concat([pd.read_csv(f) for f in working_data])
n_folds = df['fold'].max()

if 'model_name' not in df.columns:
    df['model_name']    = df['model']
if 'score'          in df.columns:
    df['roc_auc']       = df['score']
if 'roi_name'   not in df.columns:
    df['roi_name']      = df['roi']

df['feature_selector']  = df['model_name'].apply(utils.get_fs)
df['estimator']         = df['model_name'].apply(utils.get_clf)

sort_by                 = ['sub', 
                           'roi', 
                           'model', 
                           'condition_target', 
                           'condition_source', 
                           'flip',
                           'language', 
                           'transfer', 
                           'model_name',
                           'feature_selector', 
                           'estimator', 
                           'roi_name',]
df                      = df.sort_values(sort_by)

if not os.path.exists(os.path.join(saving_dir,'decoding cross stats.csv')):
    results = dict(
            roi_name = [],
            feature_selector = [],
            condition_target = [],
            condition_source = [],
            ps = [],
            diff = [],
            p = [],
            t = [],)
    for (roi,feature_selector,condition_target,condition_source),df_sub in tqdm(
            df.copy().groupby([
            'roi_name','feature_selector','condition_target','condition_source'])):
        
        grouping = ['roi','feature_selector',]
        df_baseline = df_sub[df_sub['estimator'] == 'Dummy'].sort_values(grouping)
        df_estimate = df_sub[df_sub['estimator'] != 'Dummy'].sort_values(grouping)
        
        a = df_estimate['roc_auc'].values
        b = df_baseline['roc_auc'].values
        
        t,p = stats.ttest_ind(a,b,)
        if a.mean() <= b.mean():
            p = 1-p/2
        else:
            p = p/2
        
        n_permutation = int(1e6)
        diff = a - b
        null = diff - np.mean(diff) + 0.
        null_dist = np.random.choice(null,size = (diff.shape[0],n_permutation),replace = True)
        ps = (np.sum(null_dist.mean(0) >= diff.mean()) + 1) / (n_permutation + 1)
        gc.collect()
        del null_dist
        
        results['roi_name'].append(roi)
        results['feature_selector'].append(feature_selector)
        results['condition_source'].append(condition_source)
        results['condition_target'].append(condition_target)
        results['ps'].append(ps)
        results['diff'].append(np.mean(a - b))
        results['t'].append(t)
        results['p'].append(p)
    
    results = pd.DataFrame(results)
    
    col = 'ps'
    results = results.sort_values([col])
    converter = utils.MCPConverter(pvals = results[col].values)
    d = converter.adjust_many()
    results['ps_corrected'] = d['bonferroni'].values
    
    
    results = results.sort_values(['condition_source','condition_target','roi_name','feature_selector','ps_corrected'])
    results.to_csv(os.path.join(saving_dir,'decoding cross stats.csv'),index=False)
else:
    results = pd.read_csv(os.path.join(saving_dir,'decoding cross stats.csv'))

results['stars'] = results['ps_corrected'].apply(utils.stars)

results_trim = results[results['ps_corrected'] < 0.05]
for cs,df_sub in results_trim.groupby(['condition_source','condition_target']):
    from collections import Counter
    print(cs,Counter(df_sub['roi_name']))
    print()

df['region'] = df['roi_name'].map(utils.define_roi_category())
df = pd.concat([df_sub for ii,df_sub in df.groupby([
            'region','roi_name','condition_target','condition_source'])])
violin = dict(split = True,cut = 0, inner = 'quartile')

model_name = 'PCA + Linear-SVM'
df_plot = df[df['model_name'] != model_name]
results['model_name'] = results['feature_selector'].apply(lambda x: x + ' + Linear-SVM')
results_plot = results[results['model_name'] == model_name]


g = sns.catplot(x = 'roi_name',
                y = 'roc_auc',
                row = 'condition_source',
                row_order = ['unconscious','glimpse','conscious'],
                col = 'condition_target',
                col_order = ['unconscious','glimpse','conscious'],
                data = df_plot,
                kind = 'bar',
                aspect = 3,
#                **violin
                )
(g.set_axis_labels('ROIs','ROC AUC')
  .set_titles('{row_name} --> {col_name}')
  .set(ylim=(0.35,1.)))
[ax.set_xticklabels(ax.xaxis.get_majorticklabels(),rotation = 90, ha = 'center') for ax in g.axes[-1]]
xtick_order = list(g.axes[-1][-1].xaxis.get_majorticklabels())
[ax.axhline(0.5,linestyle='--',color='k',alpha=0.5) for ax in g.axes.flatten()]
[ax.axvline(6.5,linestyle='-' ,color='k',alpha=0.5) for ax in g.axes.flatten()]

# add stars
for n_row, condition_source in enumerate(['unconscious','glimpse','conscious']):
    for n_col, condition_target in enumerate(['unconscious','glimpse','conscious']):
        ax = g.axes[n_row][n_col]
        
        for ii,text_obj in enumerate(xtick_order):
            position = text_obj.get_position()
            xtick_label = text_obj.get_text()
            row = results_plot[np.logical_and(results_plot['condition_source'] == condition_source,
                                              results_plot['condition_target'] == condition_target)]
            row = row[row['roi_name'] == xtick_label]
            if '*' in row['stars'].values[0]:
                print(xtick_label,condition_source,condition_target,row['stars'].values[0])
                ax.annotate(row['stars'].values[0],
                            ha = 'center',
                            fontsize = 16,
                            xy = (position[0],0.9))
            else:
                print(xtick_label,condition_source,condition_target,row['stars'].values[0])

title = 'within subject - out-of-sample, fold varies\n nilearn pipeline, estimator = {}\nBoferroni corrected\n*:<0.05,**<0.01,***<0.001'.format(
    n_folds,model_name)
# g.fig.suptitle(title,y = 1.1)
g.savefig(os.path.join(figure_dir,f'cross state decoding (light).png'),
#          dpi = 400,
          bbox_inches = 'tight')
g.savefig(os.path.join(figure_dir,f'cross state decoding.png'),
          dpi = 400,
          bbox_inches = 'tight')
g.savefig(os.path.join(f'/export/home/nmei/nmei/properties_of_unconscious_processing/figures',
                        f'supfig{sub[-1]}.png'),
        dpi = 450,
        bbox_inches = 'tight')






























