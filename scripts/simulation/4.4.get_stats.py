"""
Created on Wed Jul  8 21:03:54 2020

@author: ning
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

from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit,cross_validate
from sklearn.utils import shuffle as sk_shuffle
from sklearn.inspection import permutation_importance

sns.set_style('white')
sns.set_context('poster',font_scale = 1.5,)
from matplotlib import rc
rc('font',weight = 'bold')
plt.rcParams['axes.labelsize'] = 45
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 45
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['ytick.labelsize'] = 32
plt.rcParams['xtick.labelsize'] = 32
paper_dir = '/export/home/nmei/nmei/properties_of_unconscious_processing/figures'
working_dir     = '../../another_git/agent_models/results'
figure_dir      = '../figures'
marker_factor   = 10
marker_type     = ['8','s','p','*','+','D','o']
alpha_level     = .75

if not os.path.exists(figure_dir):
    os.mkdir(figure_dir)

working_data = glob(os.path.join(working_dir,'*','*.csv'))

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

df['x']         = df['noise_level'].round(9).map(x_map)
print(df['x'].values)

df['x']             = df['x'].apply(lambda x: [x + np.random.normal(0,0.1,size = 1)][0][0])
df                  = df.sort_values(['hidden_activation','output_activation'])
df['activations']   = df['hidden_activation'] + '_' +  df['output_activation']

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
