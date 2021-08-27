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

working_dir     = '../results/trained_with_noise'
figure_dir      = '../figures'
marker_factor   = 10
marker_type     = ['8','s','p','*','+','D','o']
alpha_level     = .75

if not os.path.exists(figure_dir):
    os.mkdir(figure_dir)

working_data = glob(os.path.join(working_dir,'*','decodings.csv'))

paper_dir = '/export/home/nmei/nmei/properties_of_unconscious_processing/figures'


idx_noise_applied = 13.25 # fit and predict by a linear regression, don't forget to apply log10

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
df['Dropout rate']  = df['drop']
df['# of hidden units']  = df['hidden_units']

# plot cnn and svm on hidden layer
df_plot = pd.melt(df,id_vars = ['Model name',
                                '# of hidden units',
                                'hidden_activation',
                                'output_activation',
                                'dropout',
                                'noise_level',
                                'Dropout rate',
                                'x',
                                'activations',],
                  value_vars = ['cnn_score',
                                'svm_score_mean',])
temp = pd.melt(df,id_vars = ['Model name',
                                '# of hidden units',
                                'hidden_activation',
                                'output_activation',
                                'dropout',
                                'noise_level',
                                'Dropout rate',
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
[ax.axvline(idx_noise_applied,
            linestyle       = '--',
            color           = 'red',
            alpha           = 1.,
            lw              = 3,
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
g.savefig(os.path.join(paper_dir,
                       'trained with noise performance.jpg'),
          bbox_inches = 'tight')

# direct comparison between cnn and hidden layer
df['Decoding hidden layer > CNN'] = df['svm_score_mean'].values - df['cnn_score'].values

g               = sns.relplot(
                x           = 'x',
                y           = 'Decoding hidden layer > CNN',
                hue         = 'Model name',
                hue_order   = model_names,
                size        = 'Dropout rate',
                style       = '# of hidden units',
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
[ax.axvline(idx_noise_applied,
            linestyle       = '--',
            color           = 'red',
            alpha           = 1.,
            lw              = 3,
            ) for ax in g.axes.flatten()]
[ax.set(xticks = [0,n_noise_levels],
        xticklabels = [0,noise_levels.max()]
        ) for ax in g.axes.flatten()]

(g.set_axis_labels('Noise Level','Difference')
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
g.savefig(os.path.join(paper_dir,
                       'trained with noise difference between cnn and svm.jpg'),
          dpi = 100,
          bbox_inches = 'tight')

# chance level cnn
df_chance = df[df['cnn_pval' ] > 0.05]

df_plot                         = df_chance.copy()
df_plot['Decode Above Chance']  = df_plot['svm_cnn_pval'] < 0.05
df_plot = df_plot.sort_values(['# of hidden units','Dropout rate','Model name'])
df_plot['# of hidden units'] = df_plot['# of hidden units'].astype('category')
k                               = len(pd.unique(df_plot['# of hidden units']))
df_plot['x']                    = df_plot['noise_level'].round(9).map(x_map)
df_plot['x']                    = df_plot['x'].apply(lambda x: [x + np.random.normal(0,0.1,size = 1)][0][0])

bins = np.array([-0.5,idx_noise_applied,n_noise_levels + 1.5],dtype = 'float')
def cut_bins(x):
    if bins[0] <= x < bins[1]:
        return 'low'
    # elif bins[1] <= x < bins[2]:
    #     return 'medium'
    else:
        return 'high'
    
g                               = sns.relplot(
                x               = 'x',
                y               = 'svm_score_mean',
                size            = 'Dropout rate',
                hue             = '# of hidden units',
                hue_order       = pd.unique(df_plot['# of hidden units']),
                style           = 'Decode Above Chance',
                style_order     = [True, False],
                row             = 'Model name',
                row_order       = model_names,
                alpha           = alpha_level,
                data            = df_plot,
                palette         = sns.color_palette("bright")[:k],
                height          = 8,
                aspect          = 4,
                )
(g.set_axis_labels('Noise Level','ROC AUC')
  .set_titles('{row_name}')
  .set(xlim = (-0.1,50.5)))

[ax.axhline(0.5,
            linestyle           = '--',
            color               = 'black',
            alpha               = 1.,
            lw                  = 1,
            ) for ax in g.axes.flatten()]
[ax.axvline(idx_noise_applied,
            linestyle       = '--',
            color           = 'red',
            alpha           = 1.,
            lw              = 3,
            ) for ax in g.axes.flatten()]
[ax.text(idx_noise_applied - 0.7,
         0.8,
         'Low noise',
         rotation = 90, 
         va = 'center',
         ) for ax in g.axes.flatten()]
[ax.text(idx_noise_applied + 0.1,
         0.8,
         'High noise',
         rotation = 270, 
         va = 'center',
         ) for ax in g.axes.flatten()]
[ax.set(xticks = [0,n_noise_levels],
        xticklabels = [0,noise_levels.max()]
        ) for ax in g.axes.flatten()]

temp = []
for model_name,ax in zip(model_names,g.axes.flatten()):
    df_sub = df_plot[df_plot['Model name'] == model_name]
    df_sub['groups'] = df_sub['x'].apply(cut_bins)
    counter = df_sub.groupby(['groups','Decode Above Chance']).count().reset_index()[['groups','Decode Above Chance','x']]
    sum_of_group = counter['x'].values[::2] + counter['x'].values[1::2]
    counter['proportion'] = counter['x'].values / np.repeat(sum_of_group,2)
    counter['Model name'] = model_name
    temp.append(counter)
    
    
    tiny_ax = ax.inset_axes([.6,.6,.3,.3])
    tiny_ax = sns.barplot(x = 'groups',
                          order = ['low','high'],
                          y = 'proportion',
                          hue = 'Decode Above Chance',
                          hue_order = [True,False],
                          data = counter,
                          ax = tiny_ax,
                          palette = ['green','red'],
                          )
    # tiny_ax.set(xticklabels = ['low','medium','high'],
    tiny_ax.set_xlabel('Noise level',fontsize = 18)
    tiny_ax.set_ylabel('Decoding rate',fontsize = 18)
    tiny_handles,tiny_labels = tiny_ax.get_legend_handles_labels()
    tiny_ax.get_legend().remove()
    
df_proportion = pd.concat(temp)
df_proportion.to_csv(os.path.join(paper_dir.replace('figures','stats'),
                                  'trained with noise CNN_chance_decode_proportion.csv'),
                     index = False)
handles, labels                 = g.axes[0][0].get_legend_handles_labels()
[handles.append(item) for item in tiny_handles]
[labels.append(item) for item in ['Decode Above Chance','Decode At Chance']]
g._legend.remove()
for ii,color in enumerate(sns.color_palette("bright")[:k]):
    handles[ii + 1]             = Patch(facecolor = color)
g.fig.legend(handles,
             labels,
             loc = "center right",
             borderaxespad = 0.1)
g.savefig(os.path.join(paper_dir,
                       'trained with noise chance cnn.jpg'),
          dpi = 100,
          bbox_inches = 'tight')

















