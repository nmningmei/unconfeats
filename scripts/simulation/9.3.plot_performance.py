#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 14:51:57 2021

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

if not os.path.exists(figure_dir):
    os.mkdir(figure_dir)

working_data = glob(os.path.join(working_dir,'*','decodings.csv'))

paper_dir = '/export/home/nmei/nmei/properties_of_unconscious_processing/figures'


df              = []
for f in working_data:
    if ('densenet' not in f):
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

model_names     = ['VGG19','Resnet50']
n_noise_levels  = 50
noise_levels    = np.concatenate([[0],[item for item in np.logspace(-1,3,n_noise_levels)]])
x_map           = {round(item,9):ii for ii,item in enumerate(noise_levels)}
inverse_x_map   = {round(value,9):key for key,value in x_map.items()}
print(x_map,inverse_x_map)

df['x']         = df['noise_level'].round(9).map(x_map)
print(df['x'].values)

df['x']             = df['x'].apply(lambda x: [x + np.random.normal(0,0.1,size = 1)][0][0])
df                  = df.sort_values(['hidden_activation','output_activation'])
df['activations']   = df['hidden_activation'] + '_' +  df['output_activation']
df['Model name']    = df['model_name'].map({'vgg19_bn':'VGG19','resnet50':'Resnet50'})


# plot cnn and svm on hidden layer
df_plot = pd.melt(df,id_vars = ['Model name',
                                'hidden_units',
                                'hidden_activation',
                                'output_activation',
                                'dropout',
                                'noise_level',
                                'drop',
                                'x',
                                'activations',],
                  value_vars = ['cnn_score_mean',
                                'svm_score_mean',])
temp = pd.melt(df,id_vars = ['Model name',
                                'hidden_units',
                                'hidden_activation',
                                'output_activation',
                                'dropout',
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
                size        = 'drop',
                style       = 'hidden_units',
                col         = 'Model name',
                col_order   = model_names,
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
asdf
g.savefig(os.path.join(paper_dir,'supplymental cnn hidden layer decoding vgg+resnet.jpg'),
          dpi = 300,
          bbox_inches = 'tight')
g.savefig(os.path.join(paper_dir,'supplymental cnn hidden layer decoding vgg+resnet (light).jpg'),
          # dpi = 300,
          bbox_inches = 'tight')


# plot cnn and svm on first layer
df_plot = pd.melt(df,id_vars = ['Model name',
                                'hidden_units',
                                'hidden_activation',
                                'output_activation',
                                'dropout',
                                'noise_level',
                                'drop',
                                'x',
                                'activations',],
                  value_vars = ['cnn_score',
                                'first_score_mean',])
temp = pd.melt(df,id_vars = ['Model name',
                                'hidden_units',
                                'hidden_activation',
                                'output_activation',
                                'dropout',
                                'noise_level',
                                'drop',
                                'x',
                                'activations',],
                  value_vars = ['cnn_pval',
                                'svm_first_pval',])
df_plot['pvals'] = temp['value'].values.copy()
df_plot['Type'] = df_plot['variable'].apply(lambda x: x.split('_')[0].upper())
df_plot['Type'] = df_plot['Type'].map({'CNN':'CNN',
                                       'FIRST':'Decode first layer'})

from utils_deep import resample_ttest
"""
If we decode from the frist layer, the configurations like the hidden units does
not matter because the training does not matter
so we should average these points?
"""
df_plot_first = {col_name:[] for col_name in df_plot.columns}
for (noise_level,_type,model_name),df_sub in df_plot.groupby(['noise_level','Type','Model name']):
    df_plot_first['hidden_units'].append(pd.unique(df_sub['hidden_units'])[0])
    df_plot_first['hidden_activation'].append(pd.unique(df_sub['hidden_activation'])[0])
    df_plot_first['output_activation'].append(pd.unique(df_sub['output_activation'])[0])
    df_plot_first['dropout'].append(pd.unique(df_sub['dropout'])[0])
    df_plot_first['drop'].append(pd.unique(df_sub['drop'])[0])
    df_plot_first['noise_level'].append(noise_level)
    df_plot_first['activations'].append(pd.unique(df_sub['activations'])[0])
    df_plot_first['variable'].append(pd.unique(df_sub['variable'])[0])
    df_plot_first['Type'].append(_type)
    
    df_plot_first['Model name'].append(model_name)
    df_plot_first['x'].append(x_map[noise_level.round(9)])
    df_plot_first['pvals'].append(resample_ttest(df_sub['value'].values,
                                                  0.5,
                                                  n_permutation=int(1e4),
                                                  ))
    df_plot_first['value'].append(df_sub['value'].values.mean())
df_plot_first = pd.DataFrame(df_plot_first)

g               = sns.relplot(
                x           = 'x',
                y           = 'value',
                hue         = 'Type',
                row         = 'Model name',
                row_order   = model_names,
                palette     = sns.xkcd_palette(['black','blue']),
                alpha       = alpha_level,
                data        = df_plot_first,
                aspect      = 2,
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
  .set_titles('{row_name}'))
handles, labels             = g.axes[0][0].get_legend_handles_labels()
# convert the circle to irrelevant patches
handles[0]                  = Patch(facecolor = 'black')
handles[1]                  = Patch(facecolor = 'blue',)
g._legend.remove()
g.fig.legend(handles,
             labels,
             loc            = "center right",
             borderaxespad  = 0.1)

g.savefig(os.path.join(paper_dir,'supplymental cnn first layer decoding vgg+resnet.jpg'),
          dpi = 300,
          bbox_inches = 'tight')
g.savefig(os.path.join(paper_dir,'supplymental cnn first layer decoding vgg+resnet (light).jpg'),
          # dpi = 300,
          bbox_inches = 'tight')

# direct comparison between cnn and hidden layer
df['Decoding hidden layer > CNN'] = df['svm_score_mean'].values - df['cnn_score'].values
# direct comparison between cnn and first layer
df['Decoding first layer > CNN'] = df['first_score_mean'].values - df['cnn_score'].values

g               = sns.relplot(
                x           = 'x',
                y           = 'Decoding hidden layer > CNN',
                hue         = 'Model name',
                hue_order   = model_names,
                size        = 'drop',
                style       = 'hidden_units',
                row         = 'hidden_activation',
                col         = 'output_activation',
                alpha       = alpha_level,
                data        = df,
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

#g               = sns.relplot(
#                x           = 'x',
#                y           = 'Decoding first layer > CNN',
#                hue         = 'Model name',
#                hue_order   = model_names,
##                size        = 'drop',
##                style       = 'hidden_units',
##                col         = 'hidden_activation',
##                row         = 'output_activation',
#                alpha       = alpha_level,
#                data        = df,
#                facet_kws   = {'gridspec_kws':{"wspace":0.2}},
#                aspect      = 3,
#                )
#[ax.axhline(0.5,
#            linestyle       = '--',
#            color           = 'black',
#            alpha           = 1.,
#            lw              = 1,
#            ) for ax in g.axes.flatten()]
#[ax.set(xticks = [0,n_noise_levels],
#        xticklabels = [0,noise_levels.max()]
#        ) for ax in g.axes.flatten()]
#
#(g.set_axis_labels('Noise Level','Difference')
##  .set_titles('{row_name} {col_name}')
#  )
#g.savefig(os.path.join(paper_dir,'first better than cnn.jpg'),
#          dpi = 300,
#          bbox_inches = 'tight')
#g.savefig(os.path.join(paper_dir,'first better than cnn (light).jpg'),
#          # dpi = 300,first
#          bbox_inches = 'tight')

# chance level cnn
df_chance = df[df['cnn_pval' ] > 0.05]
g               = sns.relplot(
                x           = 'x',
                y           = 'svm_score_mean',
                hue         = 'Model name',
                hue_order   = model_names,
                size        = 'drop',
                style       = 'hidden_units',
                row         = 'hidden_activation',
                col         = 'output_activation',
                alpha       = alpha_level,
                palette     = sns.xkcd_palette(['blue','orange']),
                data        = df_chance,
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
  .set_titles('{row_name}->{col_name}'))
handles, labels             = g.axes[0][0].get_legend_handles_labels()
# convert the circle to irrelevant patches
handles[1]                  = Patch(facecolor = 'blue')
handles[2]                  = Patch(facecolor = 'orange',)
g._legend.remove()
g.fig.legend(handles,
             labels,
             loc            = "center right",
             borderaxespad  = 0.1)
g.savefig(os.path.join(paper_dir,'supplemental cnn chance hidden layer.jpg'),
          dpi = 300,
          bbox_inches = 'tight')
g.savefig(os.path.join(paper_dir,'supplemental cnn chance hidden layer(light).jpg'),
#          dpi = 300,
          bbox_inches = 'tight')

df_chance_first = {
    'noise_level': [],
    'cnn_score': [],
    'cnn_pval': [],
    'first_score_mean': [],
    'svm_first_pval':[],
    'Model name': [],
    }
from utils_deep import resample_ttest
for (noise_level,model_name),df_sub in df.groupby(['noise_level','Model name']):
    df_chance_first['noise_level'].append(noise_level)
    df_chance_first['Model name'].append(model_name)
    df_chance_first['cnn_score'].append(np.mean(df_sub['cnn_score'].values))
    df_chance_first['cnn_pval'].append(resample_ttest(df_sub['cnn_score'].values,
                                                      n_permutation=int(1e4),
                                                      one_tail=True,
                                                      metric_func=np.median))
    df_chance_first['first_score_mean'].append(df_sub['first_score_mean'].values.mean())
    df_chance_first['svm_first_pval'].append(resample_ttest(df_sub['first_score_mean'].values,
                                                        n_permutation=int(1e4),
                                                        one_tail=True,
                                                        metric_func=np.median))
df_chance_first = pd.DataFrame(df_chance_first)
df_chance_first['x'] = df_chance_first['noise_level'].round(9).map(x_map)

df_chance_first = df_chance_first[df_chance_first['cnn_pval'] > 0.05]
df_chance_first['p value < 0.05'] = df_chance_first['svm_first_pval'].values < 0.05

g               = sns.relplot(
                x           = 'x',
                y           = 'first_score_mean',
                hue         = 'Model name',
                hue_order   = model_names,
                style       = 'p value < 0.05',
                style_order = [True,False],
                alpha       = alpha_level,
                palette     = sns.xkcd_palette(['blue','orange']),
                data        = df_chance_first,
                facet_kws   = {'gridspec_kws':{"wspace":0.2}},
                aspect      = 2,
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
  # .set_titles('{row_name}')
  )
handles, labels             = g.axes[0][0].get_legend_handles_labels()
# convert the circle to irrelevant patches
handles[1]                  = Patch(facecolor = 'blue')
handles[2]                  = Patch(facecolor = 'orange',)
g._legend.remove()
g.fig.legend(handles,
            labels,
            loc            = (0.65,0.25),
            borderaxespad  = 0.1)
g.savefig(os.path.join(paper_dir,'supplemental cnn chance first layer.jpg'),
          dpi = 300,
          bbox_inches = 'tight')
g.savefig(os.path.join(paper_dir,'supplemental cnn chance first layer(light).jpg'),
#          dpi = 300,
          bbox_inches = 'tight')


















