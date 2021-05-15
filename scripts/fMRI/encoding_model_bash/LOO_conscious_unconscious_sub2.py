#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 06:53:45 2020

@author: nmei

An RSA on ROI-based format

"""

import os
import gc

import pandas as pd
import numpy  as np
import multiprocessing

print(f'availabel cpus = {multiprocessing.cpu_count()}')
from glob                    import glob
from tqdm                    import tqdm
from collections             import defaultdict
from shutil                  import copyfile
copyfile('../../../utils.py','utils.py')
import utils

from nilearn.input_data      import NiftiMasker
from sklearn.linear_model    import Ridge,RidgeCV
from sklearn.preprocessing   import StandardScaler,MinMaxScaler
from sklearn.pipeline        import make_pipeline
from sklearn.model_selection import cross_validate,GridSearchCV
from sklearn.metrics         import make_scorer,r2_score

def score_func(y, y_pred,tol = 1e-2,func = np.sum):
    temp        = r2_score(y,y_pred,multioutput = 'raw_values')
    if np.sum(temp > tol) > 0:
        return func(temp[temp > tol])
    else:
        return 0

scorer = make_scorer(score_func, greater_is_better = True,)

# interchangable part:
sub                     = 'sub-02'
conscious_state_source  = 'conscious'
conscious_state_target  = 'unconscious'
tol                     = 1e-2
fine_tune_at            = 'caltech'
model_name              = 'AlexNet'

stacked_data_dir        = '../../../../data/BOLD_whole_brain_averaged/{}/'.format(sub)
feature_dir             = '../../../../data/computer_vision_features_no_background_{}'.format(fine_tune_at)
mask_dir                = '../../../../data/MRI/{}/func/mask.nii.gz'.format(sub)
function_brain          = '../../../../data/MRI/{}/func/example_func.nii.gz'.format(sub)
output_dir              = '../../../../results/MRI/nilearn/encoding/{}/{}'.format(sub,model_name)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
BOLD_data               = np.sort(glob(os.path.join(stacked_data_dir,'*nii.gz')))
event_data              = np.sort(glob(os.path.join(stacked_data_dir,'*.csv')))


label_map               = {'Nonliving_Things':[0,1],
                           'Living_Things':   [1,0]}
average                 = True
n_jobs                  = 16
n_splits                = -1
alpha_max               = 20



np.random.seed(12345)
masker_source           = NiftiMasker(mask_dir)
masker_source.fit()
data_source             = masker_source.transform([item for item in BOLD_data if (f'_{conscious_state_source}' in item)][0])
df_data_source          = pd.read_csv([item for item in event_data if (f'_{conscious_state_source}' in item)][0])
df_data_source['id']    = df_data_source['session'] * 1000 + df_data_source['run'] * 100 + df_data_source['trials']
targets_source          = np.array([label_map[item] for item in df_data_source['targets'].values])[:,-1]
images                  = df_data_source['paths'].apply(lambda x: x.split('.')[0] + '.npy').values
CNN_feature_source      = np.array([np.load(os.path.join(feature_dir,
                                                 model_name,
                                                 item)) for item in images])
groups_source           = df_data_source['labels'].values

masker_target           = NiftiMasker(mask_dir,)
masker_target.fit()
data_target             = masker_target.transform([item for item in BOLD_data if (f'_{conscious_state_target}' in item)][0])
df_data_target          = pd.read_csv([item for item in event_data if (f'_{conscious_state_target}' in item)][0])
df_data_target['id']    = df_data_target['session'] * 1000 + df_data_target['run'] * 100 + df_data_target['trials']
targets_target          = np.array([label_map[item] for item in df_data_target['targets'].values])[:,-1]
images                  = df_data_target['paths'].apply(lambda x: x.split('.')[0] + '.npy').values
CNN_feature_target      = np.array([np.load(os.path.join(feature_dir,
                                                 model_name,
                                                 item)) for item in images])
groups_target           = df_data_target['labels'].values

print(f'partitioning target: {conscious_state_target}')
idxs_train_target,idxs_test_target  = utils.LOO_partition(df_data_target)
n_splits                            = len(idxs_test_target)
print(f'{n_splits} folds of testing')

print(f'partitioning source: {conscious_state_source}')
# for each fold of the train-test, we throw away the subcategories that exist in the target
cv_warning                  = False
idxs_train_source           = []
idxs_test_source            = []
for idx_test_target in tqdm(idxs_test_target):
    df_data_target_sub      = df_data_target.iloc[idx_test_target]
    unique_subcategories    = pd.unique(df_data_target_sub['labels'])
    # category check:
    # print(Counter(df_data_target_sub['targets']))
    idx_train_source        = []
    idx_test_source         = []
    for subcategory,df_data_source_sub in df_data_source.groupby(['labels']):
        if subcategory not in unique_subcategories:
            idx_train_source.append(list(df_data_source_sub.index))
        else:
            idx_test_source.append(list(df_data_source_sub.index))
    idx_train_source        = np.concatenate(idx_train_source)
    idx_test_source         = idx_train_source.copy()
    
    # check if the training and testing have subcategory overlapping
    target_set              = set(pd.unique(df_data_target.iloc[idx_test_target]['labels']))
    source_set              = set(pd.unique(df_data_source.iloc[idx_train_source]['labels']))
    overlapping             = target_set.intersection(source_set)
    # print(f'overlapped subcategories: {overlapping}')
    if len(overlapping) > 0:
        cv_warning          = True
    idxs_train_source.append(idx_train_source)
    # the testing set for the source does NOT matter since we don't care its performance
    idxs_test_source.append(idx_test_source)
    
if not cv_warning:
    csv_saving_name = os.path.join(output_dir,f'{sub}_{conscious_state_source}_{conscious_state_target}.csv')
    if not os.path.exists(csv_saving_name):
        results = dict(scores = [],
                       alpha = [],
                       fold = [],
                       positive_voxels = [],
                       sub_name = [],
                       condition_source = [],
                       condition_target = [],
                       )
    else:
        temp = pd.read_csv(csv_saving_name)
        results = {name:[] for name in temp.columns}
        for _,row in temp.iterrows():
            [results[name].append(row[name]) for name in temp.columns]
        done_fold = row['fold'] - 1
    raw_scores = []
    for fold,(idx_train,idx_test,idx_test_target) in enumerate(zip(
            idxs_train_source,idxs_test_source,idxs_test_target)):
        if 'done_fold' in globals() and done_fold >= fold:
            print('you have done fold {}'.format(fold + 1))
        else:
            scaler_x                = StandardScaler()
            scaler_y                = MinMaxScaler((-1,1)) # seems like an important step
            reg                     = Ridge(alpha           = 1,
                                            normalize       = False,
                                            fit_intercept   = False,
                                            random_state    = 12345,
                                            )
            grid_search             = GridSearchCV(reg,
                                                   param_grid = dict(alpha = np.logspace(2,alpha_max,alpha_max-1)),
                                                   scoring = scorer,
                                                   cv = 10,
                                                   n_jobs = n_jobs,
                                                   )
            
            X                       = scaler_x.fit_transform(CNN_feature_source[idx_train])
            Y                       = scaler_y.fit_transform(data_source[idx_train])
            grid_search.fit(X,Y)
        
            gc.collect()
            y_true                  = scaler_y.transform(data_target[idx_test_target])
            y_pred                  = grid_search.predict(scaler_x.transform(CNN_feature_target[idx_test_target]))
            score                   = score_func(y_true,y_pred,func = np.mean)
            
            
            results['scores'            ].append(score)
            results['alpha'             ].append(grid_search.best_params_['alpha'])
            results['positive_voxels'   ].append(np.sum(y_pred >= tol))
            results['sub_name'          ].append(sub)
            results['condition_source'  ].append(conscious_state_source)
            results['condition_target'  ].append(conscious_state_target)
            results['fold'              ].append(fold + 1)
            
            raw_score = r2_score(y_true, y_pred,multioutput = 'raw_values')
            
            raw_score_nii           = masker_target.inverse_transform(raw_score)
            results_to_save         = pd.DataFrame(results)
            results_to_save.to_csv(os.path.join(output_dir,f'{sub}_{conscious_state_source}_{conscious_state_target}.csv'),index = False)
            
            raw_score_nii.to_filename(os.path.join(output_dir,
                                                    f'{sub}_{conscious_state_source}_{conscious_state_target}_{fold+1}.nii.gz'))
            gc.collect()




