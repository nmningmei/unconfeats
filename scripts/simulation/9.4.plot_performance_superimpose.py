#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 10:33:27 2021

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

sns.set_style('whitegrid')
sns.set_context('poster',font_scale = 1.5)
from matplotlib import rc
rc('font',weight = 'bold')
plt.rcParams['axes.labelsize'] = 45
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 45
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['ytick.labelsize'] = 32
plt.rcParams['xtick.labelsize'] = 32

working_dir     = '../results/first_layer'
figure_dir      = '../figures'
marker_factor   = 10
marker_type     = ['8','s','p','*','+','D','o']
alpha_level     = .75

paper_dir       = '/export/home/nmei/nmei/properties_of_unconscious_processing/figures'
working_dir1    = '../../another_git/agent_models/results'
working_dir2    = '../results/first_layer'
n_noise_levels  = 50
noise_levels    = np.concatenate([[0],[item for item in np.logspace(-1,3,n_noise_levels)]])
def load_data(working_data,):
    df              = []
    for f in working_data:
        if ('inception' not in f):
            temp                                    = pd.read_csv(f)
            k                                       = f.split('/')[-2]
            try:
                model,hidde_unit,hidden_ac,drop,out_ac  = k.split('_')
            except:
                _model,_model_,hidde_unit,hidden_ac,drop,out_ac  = k.split('_')
                model = f'{_model}_{_model_}'
                
            if 'drop' not in temp.columns:
                temp['drop']                        = float(drop)
            df.append(temp)
    df              = pd.concat(df)
    
    
    x_map           = {round(item,9):ii for ii,item in enumerate(noise_levels)}
    inverse_x_map   = {round(value,9):key for key,value in x_map.items()}
    print(x_map,inverse_x_map)
    
    df['x']         = df['noise_level'].round(9).map(x_map)
    df['x_id']      = df['noise_level'].round(9).map(x_map)
    print(df['x'].values)
    
    df['x']             = df['x'].apply(lambda x: [x + np.random.normal(0,0.1,size = 1)][0][0])
    df                  = df.sort_values(['hidden_activation','output_activation'])
    df['activations']   = df['hidden_activation'] + '->' +  df['output_activation']
    return df

working_data1 = glob(os.path.join(working_dir1,'*','*.csv'))
working_data2 = glob(os.path.join(working_dir2,'*','decodings.csv'))

df1 = load_data(working_data1)
df2 = load_data(working_data2)
df1 = df1[np.logical_or(df1['model_name'] == 'vgg19',
                        df1['model_name'] == 'resnet')]
df2 = df2[df2['model_name'] != 'densenet169']
"""
temp operation
"""
values = df2['cnn_score'].values
temp = []
for item in values:
    item_temp = []
    for item_item in item[1:-2].replace('\n','').split(' '):
        try:
            item_item = float(item_item)
            item_temp.append(item_item)
        except:
            pass
    temp.append(np.mean(item_temp))
df2['cnn_score'] = np.array(temp)

name_mapper = dict(vgg19_bn='vgg19',
                   resnet50='resnet')
model_for_plot = dict(vgg19='VGG19',resnet='Resnet50')

col_for_comparison = ['x_id',
                      'model_name',
                      'hidden_units',
                      'hidden_activation',
                      'drop',
                      'output_activation',]
iterator = tqdm(df2.groupby(col_for_comparison))
df_include = []
for attributes,df_sub in iterator:
    x_id,model_name,hidden_units,hidden_activation,drop_rate,output_activation =\
        attributes
    model_name = name_mapper[model_name]
    row_picked = np.array([df1[col_name] == value for col_name,value in zip(
                                      col_for_comparison,
                                      [x_id,
                                       model_name,
                                       hidden_units,
                                       hidden_activation,
                                       drop_rate,
                                       output_activation])]).sum(0) == len(col_for_comparison)
    row = df1[row_picked]
    row = row[row['model'] == 'CNN']
    df_sub['model_name'] = model_for_plot[model_name]
    iterator.set_description(f'size = {len(row)}')
    if len(row) == 1:
        df_sub['cnn_score'] = row['score_mean'].values
        df_sub['cnn_pval'] = row['pval'].values
        df_include.append(df_sub)
df_include = pd.concat(df_include)

df_include['Model name'] = df_include['model_name']
df_include['Dropout rate'] = df_include['drop']
df_include['# of hidden units'] = df_include['hidden_units']

# plot cnn and svm on hidden layer
df_plot = pd.melt(df_include,id_vars = ['Model name',
                                '# of hidden units',
                                'hidden_activation',
                                'output_activation',
                                'Dropout rate',
                                'noise_level',
                                'drop',
                                'x',
                                'activations',],
                  value_vars = ['cnn_score',
                                'svm_score_mean',])
temp = pd.melt(df_include,id_vars = ['Model name',
                                '# of hidden units',
                                'hidden_activation',
                                'output_activation',
                                'Dropout rate',
                                'noise_level',
                                'drop',
                                'x',
                                'activations',],
                  value_vars = ['cnn_pval',
                                'svm_cnn_pval',])
df_plot['pvals'] = temp['value'].values.copy()
df_plot['Type'] = df_plot['variable'].apply(lambda x: x.split('_')[0].upper())
df_plot['Type'] = df_plot['Type'].map({'CNN':'CNN',
                                       'SVM':'Decode hidden layer'})

g               = sns.relplot(
                x           = 'x',
                y           = 'value',
                hue         = 'Type',
                size        = 'Dropout rate',
                style       = '# of hidden units',
                col         = 'Model name',
                col_order   = list(model_for_plot.values()),
                row         = 'activations',
                palette     = sns.xkcd_palette(['black','blue']),
                alpha       = alpha_level,
                data        = df_plot,
                facet_kws   = {'gridspec_kws':{"wspace":0.2}},
                aspect      = 3,
                )
[ax.axhline(0.5,
            linestyle       = '--',
            color           = 'black',
            alpha           = 1.,
            lw              = 1,
            ) for ax in g.axes.flatten()]
[ax.set(xticks = [0,n_noise_levels],
        xticklabels = [0,noise_levels.max()]
        ) for ax in g.axes.flatten()]

(g.set_axis_labels('Noise Level','ROC AUC')
  .set_titles('{col_name} {row_name}'))
handles, labels             = g.axes[0][0].get_legend_handles_labels()
# convert the circle to irrelevant patches
handles[1]                  = Patch(facecolor = 'black')
handles[2]                  = Patch(facecolor = 'blue',)
g._legend.remove()
g.fig.legend(handles,
             labels,
             loc            = "center right",
             borderaxespad  = 0.1)

g.savefig(os.path.join(paper_dir,'supplymental cnn hidden layer decoding vgg+resnet.jpg'),
          dpi = 300,
          bbox_inches = 'tight')
g.savefig(os.path.join(paper_dir,'supplymental cnn hidden layer decoding vgg+resnet (light).jpg'),
          # dpi = 300,
          bbox_inches = 'tight')


# direct comparison between cnn and hidden layer
df_include['Decoding hidden layer > CNN'] = df_include['svm_score_mean'].values - df_include['cnn_score'].values

g               = sns.relplot(
                x           = 'x',
                y           = 'Decoding hidden layer > CNN',
                hue         = 'Model name',
                hue_order   = list(model_for_plot.values()),
                size        = 'Dropout rate',
                style       = '# of hidden units',
                row         = 'hidden_activation',
                col         = 'output_activation',
                alpha       = alpha_level,
                data        = df_include,
                facet_kws   = {'gridspec_kws':{"wspace":0.2}},
                aspect      = 3,
                )
[ax.axhline(0.5,
            linestyle       = '--',
            color           = 'black',
            alpha           = 1.,
            lw              = 1,
            ) for ax in g.axes.flatten()]
[ax.set(xticks = [0,n_noise_levels],
        xticklabels = [0,noise_levels.max()]
        ) for ax in g.axes.flatten()]

(g.set_axis_labels('Noise Level','Difference')
  .set_titles('{row_name}->{col_name}'))
g.savefig(os.path.join(paper_dir,'hidden better than cnn.jpg'),
          dpi = 300,
          bbox_inches = 'tight')
g.savefig(os.path.join(paper_dir,'hidden better than cnn (light).jpg'),
          # dpi = 300,
          bbox_inches = 'tight')































