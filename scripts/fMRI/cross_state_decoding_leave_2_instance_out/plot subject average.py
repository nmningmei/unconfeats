#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 12:34:25 2019

@author: nmei
"""

import os
from glob import glob

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
sns.set_context('poster',font_scale = 1.5)

working_dir = "../../../../results/MRI/nilearn/"
figure_dir = "../../../../figures/MRI/nilearn/cross-subjects"
if not os.path.exists(figure_dir):
    os.mkdir(figure_dir)

df = []
for idx_sub in range(7):
    working_data = glob(os.path.join(working_dir,
                                     "sub-0{}".format(idx_sub + 1),
                                     "LOO",
                                     "*.csv"))
    working_data = np.sort(working_data)
    
    df_sub = pd.concat([pd.read_csv(f) for f in working_data if\
                        ("None" in f)])
    df_sub['Comparison'] = df_sub['model'].apply(lambda x:"Chance" if "Dummy" in x else "Empirical")
    temp = np.array([item.split('-') for item in df_sub['roi'].values])
    df_sub['roi_name'] = temp[:,1]
    df_sub['Side'] = temp[:,0]
    
    results = pd.read_csv(os.path.join(f'../../../../results/MRI/nilearn/sub-0{idx_sub+1}/LOO_stats','LOO 4 models.csv'))
    results_trim = results[results['ps_corrected'] < 0.05]
    
    df
    
    g = sns.catplot(x = 'roi_name',
                    y = 'roc_auc',
                    hue = 'Side',
                    row = 'condition_source',
                    row_order = ['unconscious','glimpse','conscious'],
                    kind = 'bar',
                    aspect = 3,
                    data = df_sub[df_sub['Comparison'] == 'Empirical'],
                    )
    [ax.set_xticklabels(ax.xaxis.get_majorticklabels(),rotation = 90, ha = 'center') for ax in g.axes[-1]]
    xtick_order = list(g.axes[-1][-1].xaxis.get_majorticklabels())
    [ax.axhline(0.5,linestyle='--',color='k',alpha=0.5) for ax in g.axes.flatten()]
    [ax.axvline(6.5,linestyle='-' ,color='k',alpha=0.5) for ax in g.axes.flatten()]
    for n_row, conscious_state in enumerate(['unconscious','glimpse','conscious']):
        ax = g.axes[n_row][0]
        
        for ii,text_obj in enumerate(xtick_order):
            position = text_obj.get_position()
            xtick_label = text_obj.get_text()
            rows = results[results['conscious_state'] == conscious_state]
            rows = rows[rows['roi_name'] == xtick_label]
            for (ii,temp_row),adjustment in zip(rows.iterrows(),[-0.2,0.2]):
                if '*' in temp_row['stars']:
                    ax.annotate(temp_row['stars'],
                                xy = (position[0] + adjustment,0.9),
                                ha = 'center',
                                fontsize = 16,)
    (g.set_axis_labels("ROIs","ROC AUC")
      .set_titles("{row_name}")
      .set(ylim = (0.05,0.95))
      )
    title = f'''
Within subject decoding by out-of-sample generalization, folds = {48 * 48}
estimator = Linear-SVM
Boferroni corrected within conscious state
*: <0.05, **: <0.01, ***: <0.001
'''
    # g.fig.suptitle(title,y = 1.15)
    g.savefig(os.path.join(f'../../../../figures/MRI/nilearn/sub-0{idx_sub+1}/LOO',
                           'decoding (Linear-SVM) as a function of roi, conscious state.png'),
            dpi = 450,
            bbox_inches = 'tight')

    g.savefig(os.path.join(f'../../../../figures/MRI/nilearn/sub-0{idx_sub+1}/LOO',
                           'decoding (Linear-SVM) as a function of roi, conscious state (light).png'),
#            dpi = 450,
            bbox_inches = 'tight')
    plt.close('all')
    df_sub = df_sub.groupby(['sub','model','condition_source','roi']).median().reset_index()
    df_sub['sub'] = f'sub-0{idx_sub + 1}'
    df.append(df_sub)

df = pd.concat(df)
df['Comparison'] = df['model'].apply(lambda x:"Chance" if "Dummy" in x else "Empirical")
temp = np.array([item.split('-') for item in df['roi'].values])
df['roi_name'] = temp[:,1]
df['Side'] = temp[:,0]
"""
g = sns.catplot(x = 'roi_name',
                y = 'roc_auc',
                hue = 'Side',
                row = 'condition_source',
                row_order = ['unconscious','glimpse','conscious'],
#                col = 'side',
                kind = 'violin',
                data = df,
                aspect = 3,
                **{'cut':0,'split':True,'inner':None},
#                **{'jitter':True,'dodge':True},
                sharey = False,
                alpha = 0.5)
for ax,conscious_state in zip(g.axes.flatten(),['unconscious','glimpse','conscious']):
    ax = sns.stripplot(x = 'roi_name',
                       y = 'roc_auc',
                       hue = 'Side',
                       data = df[df['condition_source'] == conscious_state],
                       jitter = True,
                       dodge = True,
                       alpha = 1.0,
                       ax = ax)
    ax.get_legend().remove()
[ax.set_xticklabels(ax.xaxis.get_majorticklabels(),rotation = 90, ha = 'center') for ax in g.axes[-1]]
xtick_order = list(g.axes[-1][-1].xaxis.get_majorticklabels())
[ax.axhline(0.5,linestyle='--',color='k',alpha=0.5) for ax in g.axes.flatten()]
[ax.axvline(6.5,linestyle='-' ,color='k',alpha=0.5) for ax in g.axes.flatten()]
(g.set_axis_labels("ROIs","ROC AUC")
  .set_titles("{row_name}")
#  .set(ylim = (0.4,None))
  )
title = f'''
Cross-subject Decoding Performance
Out-of-sample decoding, folds = {48 * 48}
decoder: Linear-SVM, y-axis range are NOT equal
N = {len(pd.unique(df["sub"]))} thus standard error might not be accurate'''
g.fig.suptitle(title,y = 1.15)
g.fig.savefig(os.path.join(figure_dir,
                           'decoding (Linear-SVM) as a function of roi, conscious state.png'
                           ),
                dpi = 450,
                bbox_inches = 'tight')
g.fig.savefig(os.path.join(figure_dir,
                           'decoding (Linear-SVM) as a function of roi, conscious state (light).png'
                           ),
#                dpi = 450,
                bbox_inches = 'tight')
"""































