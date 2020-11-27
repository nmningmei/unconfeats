#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 14:26:03 2020

@author: nmei

InceptionNet is excluded from analysis

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

working_dir     = '../../another_git/agent_models/results'
figure_dir      = '../figures'
marker_factor   = 10
marker_type     = ['8','s','p','*','+','D','o']
alpha_level     = .75

if not os.path.exists(figure_dir):
    os.mkdir(figure_dir)

working_data = glob(os.path.join(working_dir,'*','*.csv'))

paper_dir = '/export/home/nmei/nmei/properties_of_unconscious_processing/figures'


df              = []
for f in working_data:
    if ('inception' not in f):
        temp                                    = pd.read_csv(f)
        k                                       = f.split('/')[-2]
        model,hidde_unit,hidden_ac,drop,out_ac  = k.split('_')
        if 'drop' not in temp.columns:
            temp['drop']                        = float(drop)
        df.append(temp)
df              = pd.concat(df)

n_noise_levels  = 50
noise_levels    = np.concatenate([[0],[item for item in np.logspace(-1,3,n_noise_levels)]])
x_map           = {round(item,9):ii for ii,item in enumerate(noise_levels)}
inverse_x_map   = {round(value,9):key for key,value in x_map.items()}
print(x_map,inverse_x_map)

_,bins          = pd.cut(np.arange(n_noise_levels),2,retbins = True)
def cut_bins(x):
    if bins[0] <= x < bins[1]:
        return 'low'
    # elif bins[1] <= x < bins[2]:
    #     return 'medium'
    else:
        return 'high'

df['x']         = df['noise_level'].round(9).map(x_map)
print(df['x'].values)

df['x']             = df['x'].apply(lambda x: [x + np.random.normal(0,0.1,size = 1)][0][0])
df                  = df.sort_values(['hidden_activation','output_activation'])
df['activations']   = df['hidden_activation'] + '_' +  df['output_activation']

idxs            = np.logical_or(df['model'] == 'CNN',df['model'] == 'linear-SVM')
df_plot         = df.loc[idxs,:]

g               = sns.relplot(
                x           = 'x',
                y           = 'score_mean',
                hue         = 'model',
                size        = 'drop',
                style       = 'hidden_units',
                col         = 'model_name',
                col_order   = ['alexnet','vgg19','mobilenet','densenet','resnet',],
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
g.savefig(os.path.join(figure_dir,'CNN_performance.jpeg'),
          dpi               = 300,
          bbox_inches       = 'tight')
g.savefig(os.path.join(figure_dir,'CNN_performance (light).jpeg'),
#          dpi = 300,
          bbox_inches       = 'tight')
g.savefig(os.path.join(paper_dir,'CNN_performance.jpeg'),
          dpi               = 300,
          bbox_inches       = 'tight')
g.savefig(os.path.join(paper_dir,'CNN_performance_light.jpeg'),
#          dpi               = 300,
          bbox_inches       = 'tight')

# plot the decoding when CNN failed
idxs            = np.logical_or(df['model'] == 'CNN',df['model'] == 'linear-SVM')
df_picked       = df.loc[idxs,:]
col_indp = ['hidden_units','hidden_activation','output_activation','noise_level','drop','model_name']
for col in col_indp:
    print(col, pd.unique(df_picked[col]))
df_stat = {name:[] for name in col_indp}
df_stat['CNN_performance'] = []
df_stat['SVM_performance'] = []
df_stat['CNN_chance_performance']= []
df_stat['CNN_pval'] = []
df_stat['SVM_pval'] = []
for attri,df_sub in tqdm(df_picked.groupby(col_indp),desc='generate features'):
    if df_sub.shape[0] == 1:
        # [df_stat[col].append(df_sub[col].values[0]) for col in col_indp]
        # df_stat['CNN_performance'].append(df_sub['score_mean'].values[0])
        # df_stat['CNN_pval'].append(df_sub['pval'].values[0])
        # df_stat['SVM_performance'].append(0)
        # df_stat['SVM_pval'].append(1)
        pass
    elif df_sub.shape[0] > 1:
        for model,df_sub_sub in df_sub.groupby('model'):
            if model == 'CNN':
                [df_stat[col].append(df_sub[col].values[0]) for col in col_indp]
                df_stat['CNN_performance'].append(df_sub_sub['score_mean'].values[0])
                df_stat['CNN_pval'].append(df_sub_sub['pval'].values[0])
                df_stat['CNN_chance_performance'].append(df_sub_sub['chance_mean'].values[0])
                
            elif model == 'linear-SVM':
                df_stat['SVM_performance'].append(df_sub_sub['score_mean'].values[0])
                df_stat['SVM_pval'].append(df_sub_sub['pval'].values[0])
    else:
        print('what?')
df_stat = pd.DataFrame(df_stat)
df_stat.to_csv(os.path.join(paper_dir,
                            'CNN_SVM_stats.csv'),index = False)

df_chance = df_stat[np.logical_or(
                        df_stat['CNN_pval'] > 0.05,
                        df_stat['CNN_performance'] < df_stat['CNN_chance_performance'])
                    ]


df_plot                         = df_chance.copy()#loc[idxs,:]
df_plot['Decode Above Chance']  = df_plot['SVM_pval'] < 0.05
df_plot = df_plot.sort_values(['hidden_units','drop','model_name'])
df_plot['hidden_units'] = df_plot['hidden_units'].astype('category')
print(pd.unique(df_plot['hidden_units']))
k                               = len(pd.unique(df_plot['hidden_units']))
df_plot['x']                    = df_plot['noise_level'].round(9).map(x_map)
df_plot['x']                    = df_plot['x'].apply(lambda x: [x + np.random.normal(0,0.1,size = 1)][0][0])

g                               = sns.relplot(
                x               = 'x',
                y               = 'SVM_performance',
                size            = 'drop',
                hue             = 'hidden_units',
                hue_order       = pd.unique(df_plot['hidden_units']),
                style           = 'Decode Above Chance',
                style_order     = [True, False],
                row             = 'model_name',
                row_order       = ['alexnet','vgg19','mobilenet','densenet','resnet',],
                alpha           = alpha_level,
                data            = df_plot,
                palette         = sns.color_palette("bright")[:k],
                height          = 8,
                aspect          = 4,
                )
(g.set_axis_labels('Noise Level','ROC AUC')
  .set_titles('{row_name}')
  .set(xlim = (-0.1,50.5)))
[ax.set(xticks = [0,n_noise_levels],
        xticklabels = [0,noise_levels.max()]
        ) for ax in g.axes.flatten()]
[ax.axhline(0.5,
            linestyle           = '--',
            color               = 'black',
            alpha               = 1.,
            lw                  = 1,
            ) for ax in g.axes.flatten()]

temp = []
for model_name,ax in zip(['alexnet','vgg19','mobilenet','densenet','resnet',],g.axes.flatten()):
    df_sub = df_plot[df_plot['model_name'] == model_name]
    df_sub['groups'] = df_sub['x'].apply(cut_bins)
    counter = df_sub.groupby(['groups','Decode Above Chance']).count().reset_index()[['groups','Decode Above Chance','x']]
    sum_of_group = counter['x'].values[::2] + counter['x'].values[1::2]
    counter['proportion'] = counter['x'].values / np.repeat(sum_of_group,2)
    counter['model_name'] = model_name
    temp.append(counter)
    
    ax.axvline(bins[1],linestyle = '--' ,color = 'black', alpha = 0.6)
    
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
                                  'CNN_chance_decode_proportion.csv'),
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

# g.fig.suptitle('Linear SVM decoding the hidden layers of CNNs that failed to descriminate living vs. nonliving',
#                y = 1.02)
g.savefig(os.path.join(figure_dir,'decoding_performance.jpeg'),
          dpi = 300,
          bbox_inches = 'tight')
g.savefig(os.path.join(figure_dir,'decoding_performance (light).jpeg'),
#          dpi = 300,
          bbox_inches = 'tight')
g.savefig(os.path.join(paper_dir,'decoding_performance.jpeg'),
          dpi = 300,
          bbox_inches = 'tight')
g.savefig(os.path.join(paper_dir,'decoding_performance_light.jpeg'),
#          dpi = 300,
          bbox_inches = 'tight')

#fig,axes = plt.subplots(figsize = (70,40),
#                        nrows = pd.unique(df['model_name']).shape[0],
#                        ncols = pd.unique(df['activations']).shape[0],
#                        sharex = True, sharey = True,)
#for ii,(ax,((model_name,activations),df_sub)) in enumerate(zip(axes.flatten(),df_plot.groupby(['model_name','activations']))):
#    k = len(pd.unique(df_sub['model']))
#    print(model_name,activations,k)
#    ax = sns.scatterplot(x = 'x',
#                         y = 'score_mean',
#                         hue = 'model',
#                         style = 'hidden_units',
#                         size = 'drop',
#                         data = df_sub,
#                         alpha = alpha_level,
#                         ax = ax,
#                         palette = sns.xkcd_palette(['black','blue'][:k]),#,'yellow','green','red']),
#                         )
#    ax.axhline(0.5,linestyle = '--',color = 'black',alpha = 1.,lw = 1)
#    _ = ax.set(ylabel = 'ROC AUC',xlabel = '',title = f'{model_name} {activations}',
#               xticks = np.arange(0,51,10),
#               )
#    if ii == len(axes.flatten()) - 1:
#        handles, labels = ax.get_legend_handles_labels()
#    ax.legend().remove()
#    xticklabels = [round(inverse_x_map[item],3) for item in np.arange(0,51,10)]
#    _ = ax.set_xticklabels(xticklabels,
#                            rotation = 45,
#                            ha = 'center')
#_ = ax.set(xlabel = 'Noise Level')
#fig.legend(handles,labels,loc="center right",borderaxespad=0.1)
#plt.subplots_adjust(right=.97)
#fig.savefig(os.path.join(figure_dir,'performance.jpeg'),
#          dpi = 300,
#          bbox_inches = 'tight')
#fig.savefig(os.path.join(figure_dir,'performance (light).jpeg'),
##          dpi = 300,
#          bbox_inches = 'tight')

#fig,axes = plt.subplots(figsize = (20,40),
#                        nrows = pd.unique(df_plot['model_name']).shape[0],
#                        sharex = True, sharey = True,)
#for ax,(model_name,df_sub) in zip(axes.flatten(),df_plot.groupby(['model_name'])):
#    temp = df_sub[df_sub['model'] == 'linear-SVM']
#    k = len(pd.unique(temp['hidden_units']))
#    all_cnn_models = df[df['model_name']==model_name].shape[0]
#    failed_cnn_models = df_sub.shape[0]
#    print(model_name,k)
#    ax = sns.scatterplot(x = 'x',
#                         y = 'score_mean',
#                         hue = 'hidden_units',
#                         size = 'drop',
#                         style = 'Decode Above Chance',
#                         style_order = [True,False],
#                         data = temp,
#                         alpha = alpha_level,
#                         ax = ax,
#                         palette = sns.xkcd_palette(np.array(['black','blue','red','green','yellow'])[:k]),
#                         )
#    ax.axhline(0.5,linestyle = '--',color = 'black',alpha = 1.,lw = 1)
#    _ = ax.set(ylabel = 'ROC AUC',
#               xlabel = '',
#               xticks = np.arange(51),
#               title = f'{model_name}, {failed_cnn_models}/{all_cnn_models}',
#               xlim = (df_chance['x'].values.min() - 1,51),
#               )
#    handles, labels = ax.get_legend_handles_labels()
#    ax.legend().remove()
#xticklabels = [round(inverse_x_map[item],3) for item in np.arange(51)]
#_ = ax.set_xticklabels(xticklabels,
#                    rotation = 45,
#                    ha = 'center')
#_ = ax.set(xlabel = 'Noise Level')
## convert the circle to irrelevant patches
#from matplotlib.patches import Patch
#handles[1] = Patch(facecolor = 'black')
#handles[2] = Patch(facecolor = 'blue',)
#handles[3] = Patch(facecolor = 'red',)
#handles[4] = Patch(facecolor = 'green',)
#fig.legend(handles,labels,loc = "center right",borderaxespad=0.1)
#plt.subplots_adjust(right=.87)
#fig.suptitle('Linear SVM decoding the hidden layers of CNNs that failed to descriminate living vs. nonliving',
#             y = 0.9)
#fig.savefig(os.path.join(figure_dir,'decoding performance.jpeg'),
#          dpi = 300,
#          bbox_inches = 'tight')
#fig.savefig(os.path.join(figure_dir,'decoding performance (light).jpeg'),
##          dpi = 300,
#          bbox_inches = 'tight')

























