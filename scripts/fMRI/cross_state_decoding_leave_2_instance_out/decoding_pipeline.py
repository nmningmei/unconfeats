#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 05:10:53 2021

@author: nmei
This script is decoding with LOO methods to fit in one of the conscious state, 
and then test in the other consciousness states

"""
def warn(*args,**kwargs):
    pass

import os
import gc
import warnings
warnings.warm = warn
# warnings.filterwarnings('ignore') 
import pandas as pd
import numpy  as np
import multiprocessing

print(f'availabel cpus = {multiprocessing.cpu_count()}')
from glob                    import glob
from shutil                  import copyfile
copyfile('../../../utils.py','utils.py')
from utils                   import (
                                     LOO_partition,
                                     check_LOO_cv,
                                     check_train_balance,
                                     build_model_dictionary,
                                     Find_Optimal_Cutoff
                                     )
from sklearn.model_selection import (cross_validate,
                                     StratifiedShuffleSplit,
                                     LeaveOneGroupOut)
from sklearn                 import metrics
from sklearn.exceptions      import ConvergenceWarning,UndefinedMetricWarning
from joblib                  import Parallel,delayed
from collections             import OrderedDict

# interchangable part:
sub                     = 'sub-05'
conscious_state_source  = 'conscious'
conscious_state_target  = 'unconscious'

folder_name             = 'decoding_LOOC_10' # decoding_LOO_balanced,decoding_stratified
if 'glimpse' in folder_name:
    add_glimpse         = True
else:
    add_glimpse         = False

stacked_data_dir        = '../../../../data/BOLD_average_BOLD_average_lr/{}/'.format(sub)
output_dir              = '../../../../results/MRI/nilearn/{}/{}'.format(folder_name,sub)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
BOLD_data               = np.sort(glob(os.path.join(stacked_data_dir,'*BOLD.npy')))
event_data              = np.sort(glob(os.path.join(stacked_data_dir,'*.csv')))


model_names             = [
        'None + Linear-SVM',
        'None + Dummy',
        ]

label_map               = {'Nonliving_Things':[0,1],
                           'Living_Things':   [1,0]}
average                 = True
n_jobs                  = -1
if 'LOO' in folder_name:
    n_splits            = -1
elif 'loocv' in folder_name:
    n_splits            = 96
else:
    n_splits            = int(1e3)

class_weight = None if "unweighted" in folder_name else 'balanced'
C = float(folder_name.split('_')[-1]) if 'LOOC' in folder_name else 1

df_event                = pd.read_csv(event_data[0])
idx_conscious_source    = df_event['visibility'] == conscious_state_source
df_data_source          = df_event[idx_conscious_source].reset_index(drop=True)
df_data_source['id']    = df_data_source['session'] * 1000 + df_data_source['run'] * 100 + df_data_source['trials']
targets_source          = np.array([label_map[item] for item in df_data_source['targets'].values])

idx_conscious_target    = df_event['visibility'] == conscious_state_target
df_data_target          = df_event[idx_conscious_target].reset_index(drop=True)
df_data_target['id']    = df_data_target['session'] * 1000 + df_data_target['run'] * 100 + df_data_target['trials']
targets_target          = np.array([label_map[item] for item in df_data_target['targets'].values])

if add_glimpse and conscious_state_source == 'conscious':
    _idx_glimpse_source     = df_event['visibility'] == 'glimpse'
    _df_data_source         = df_event[_idx_glimpse_source].reset_index(drop=True)
    _df_data_source['id']   = _df_data_source['session'] * 1000 + _df_data_source['run'] * 100 + _df_data_source['trials']
    _targets_source         = np.array([label_map[item] for item in _df_data_source['targets'].values])
    df_data_source          = pd.concat([df_data_source,_df_data_source]).reset_index(drop=True)
    targets_source          = np.concatenate([targets_source,_targets_source])
if add_glimpse and conscious_state_target == 'conscious':
    _idx_glimpse_target     = df_event['visibility'] == 'glimpse'
    _df_data_target         = df_event[_idx_glimpse_target].reset_index(drop=True)
    _df_data_target['id']   = _df_data_target['session'] * 1000 + _df_data_target['run'] * 100 + _df_data_target['trials']
    _targets_target         = np.array([label_map[item] for item in _df_data_target['targets'].values])
    df_data_target          = pd.concat([df_data_target,_df_data_target]).reset_index(drop=True)
    targets_target          = np.concatenate([targets_target,_targets_target])

# because for the same subject and the same condition (consciousness states), all
# ROIs will use the same event file, so I will create the same cross-validation
# folds for all to save some time
if n_splits == -1: # this is to do the out-of-sample generallization
    print(f'partitioning target: {conscious_state_target}')
    idxs_train_target,idxs_test_target  = LOO_partition(df_data_target)
    n_splits                            = len(idxs_test_target)
    print(f'{n_splits} folds of testing')
    
    # we partition the source training data according to the target dataset
    print(f'partitioning source: {conscious_state_source}')
    # for each fold of the train-test, we throw away the subcategories that exist in the target
    cv_warning,idxs_train_source,idxs_test_source = check_LOO_cv(
            idxs_test_target,df_data_target,df_data_source)
#    if conscious_state_source != conscious_state_target:
#        idxs_train_source = [idx_train_source.append(idx_train_target + df_data_source.shape[0]) for idx_train_source,idx_train_target in zip(idxs_train_source,idxs_train_target)]
elif n_splits == 96:
    cv = LeaveOneGroupOut()
    groups_target = df_data_target['labels'].values
    idxs_train_target,idxs_test_target  = [],[]
    for train,test in cv.split(df_data_target,targets_target[:,-1],groups_target):
        idxs_train_target.append(train)
        idxs_test_target.append(test)
    cv_warning,idxs_train_source,idxs_test_source = check_LOO_cv(
            idxs_test_target,df_data_target,df_data_source)
    
else: # this is to do a stratified shuffle cross-validation
    cv = StratifiedShuffleSplit(n_splits = n_splits,test_size = 0.05,random_state = 12345)
    print(f'partitioning target: {conscious_state_target}')
    idxs_train_target,idxs_test_target  = [],[]
    for train,test in cv.split(df_data_target,targets_target[:,-1]):
        idxs_train_target.append(train)
        if conscious_state_source == conscious_state_target:
            idxs_test_target.append(test)
        else:
            idxs_test_target.append(np.arange(df_data_target.shape[0]))
    print(f'partitioning source: {conscious_state_source}')
    # for each fold of the train-test, we throw away the subcategories that exist in the target
    idxs_train_source,idxs_test_source = [],[]
    for train,test in cv.split(df_data_source,targets_source[:,-1]):
        idxs_train_source.append(train)
        idxs_test_source.append(test)

# balance the training instances
if ("balance" in folder_name) or (conscious_state_source != conscious_state_target):
    gc.collect()
    # this is a function to randomly drop instances of the major class until the
    # classes are balanced (difference is less than 2)
    np.random.seed(12345)
    idxs_train_source = Parallel(n_jobs = -1,verbose = 1)(delayed(check_train_balance)(**{
            'df':df_data_source,
            'idx_train':idx_train,
            'keys':['Living_Things','Nonliving_Things'],
            'tol':20,
            }) for idx_train in idxs_train_source)
    gc.collect()

np.random.seed(12345)
for BOLD_name,df_name in zip(BOLD_data,event_data):
    BOLD                    = np.load(BOLD_name)
    roi_name                = df_name.split('/')[-1].split('_events')[0]
    
    idx_conscious_source    = df_event['visibility'] == conscious_state_source
    data_source             = BOLD[idx_conscious_source]
    idx_conscious_target    = df_event['visibility'] == conscious_state_target
    data_target             = BOLD[idx_conscious_target]
    
    if add_glimpse and conscious_state_source == 'conscious':
        idx_glimpse_source  = df_event['visibility'] == 'glimpse'
        _data_source        = BOLD[idx_glimpse_source]
        data_source         = np.concatenate([data_source,_data_source])
    if add_glimpse and conscious_state_target == 'conscious':
        idx_glimpse_target  = df_event['visibility'] == 'glimpse'
        _data_target        = BOLD[idx_glimpse_target]
        data_target         = np.concatenate([data_target,_data_target])
#    if conscious_state_source != conscious_state_target:
#        data_source = np.concatenate([data_source,data_target])
#        targets_source = np.concatenate([targets_source,targets_target])
    
    for model_name in model_names:
        file_name           = f'{sub}_{roi_name}_{conscious_state_source}_{conscious_state_target}_{model_name}.csv'.replace(' + ','_')
#        print(f'{roi_name} {model_name} {conscious_state_source} --> {conscious_state_target}')
        if not os.path.exists(os.path.join(output_dir,file_name)):
            np.random.seed(12345)
            features        = data_source.copy()
            targets         = targets_source.copy()[:,-1]
            
            
            pipeline        = build_model_dictionary(n_jobs            = 4,
                                                     remove_invariant  = True,
                                                     l1                = True,
                                                     C                 = C,
#                                                     class_weight      = class_weight,
#                                                     tol               = 1e-3,
                                                     )[model_name]
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", 
                                        category = ConvergenceWarning,
                                        module = "sklearn")
                gc.collect()
                res = cross_validate(pipeline,
                             features,
                             targets,
                             scoring            = 'accuracy',
                             cv                 = zip(idxs_train_source,idxs_test_source),
                             return_estimator   = True,
                             n_jobs             = n_jobs,
                             verbose            = 1,
                             )
                gc.collect()
                
                warnings.filterwarnings("ignore", 
                                        category = UndefinedMetricWarning)
                regs                = res['estimator']
                y_true              = [targets_target[idx_test] for idx_test in idxs_test_target]
                y_pred              = [estimator.predict_proba(data_target[idx_test]) for idx_test,estimator in zip(idxs_test_target,regs)]
                
                if 'loocv' not in folder_name:
                    roc_auc             = [metrics.roc_auc_score(y_true_,y_pred_,average = 'micro') for y_true_,y_pred_ in zip(y_true,y_pred)]
                    threshold_          = [Find_Optimal_Cutoff(y_true_[:,-1],y_pred_[:,-1]) for y_true_,y_pred_ in zip(y_true,y_pred)]
                    mattews_correcoef   = [metrics.matthews_corrcoef(y_true_[:,-1],y_pred_[:,-1]>thres_) for y_true_,y_pred_,thres_ in zip(y_true,y_pred,threshold_)]
                    f1_score            = [metrics.f1_score(y_true_[:,-1],y_pred_[:,-1]>thres_) for y_true_,y_pred_,thres_ in zip(y_true,y_pred,threshold_)]
                    log_loss            = [metrics.log_loss(y_true_,y_pred_) for y_true_,y_pred_ in zip(y_true,y_pred)]
                    
                    
                    temp                = np.array([metrics.confusion_matrix(y_true_[:,-1],y_pred_[:,-1]>thres_).ravel() for y_true_,y_pred_,thres_ in zip(y_true,y_pred,threshold_)])
                    tn, fp, fn, tp      = temp[:,0],temp[:,1],temp[:,2],temp[:,3]
                    
                    results                         = OrderedDict()
                    results['fold']                 = np.arange(n_splits) + 1
                    results['sub']                  = [sub] * n_splits
                    results['roi']                  = [roi_name] * n_splits
                    results['roc_auc']              = roc_auc
                    results['mattews_correcoef']    = mattews_correcoef
                    results['f1_score']             = f1_score
                    results['log_loss']             = log_loss
                    results['model']                = [model_name] * n_splits
                    results['condition_target']     = [conscious_state_target] * n_splits
                    results['condition_source']     = [conscious_state_source] * n_splits
                    results['tn']                   = tn
                    results['tp']                   = tp
                    results['fn']                   = fn
                    results['fp']                   = fp
                    results['y_true']               = [','.join(y_true_[:,-1].astype(int).astype(str)) for y_true_ in y_true]
                    gc.collect()
                    print(f'{conscious_state_source}-->{conscious_state_target}, {roi_name}, {model_name}, roc_auc = {np.mean(roc_auc):.4f}+/-{np.std(roc_auc):.4f}')
                    results_to_save                 = pd.DataFrame(results)
                    results_to_save.to_csv(os.path.join(output_dir,file_name),index = False)
                    print(f'saving {os.path.join(output_dir,file_name)}')
                elif 'loocv' in folder_name:
                    y_true = np.vstack(y_true)
                    y_pred = np.vstack(y_pred)
                    roc_auc = metrics.roc_auc_score(y_true,y_pred,average = 'micro')
                    threshold = Find_Optimal_Cutoff(y_true[:,-1],y_pred[:,-1])
                    mattews_correcoef = metrics.matthews_corrcoef(y_true[:,-1],y_pred[:,-1]>threshold)
                    f1_score = metrics.f1_score(y_true[:,-1],y_pred[:,-1]>threshold)
                    log_loss = metrics.log_loss(y_true,y_pred)
                    temp = metrics.confusion_matrix(y_true[:,-1],y_pred[:,-1]>threshold).ravel()
                    tn,fp,fn,tp = temp
                    results                         = OrderedDict()
                    results['fold']                 = [n_splits]
                    results['sub']                  = [sub]
                    results['roi']                  = [roi_name]
                    results['roc_auc']              = [roc_auc]
                    results['mattews_correcoef']    = [mattews_correcoef]
                    results['f1_score']             = [f1_score]
                    results['log_loss']             = [log_loss]
                    results['model']                = [model_name]
                    results['condition_target']     = [conscious_state_target]
                    results['condition_source']     = [conscious_state_source]
                    results['tn']                   = [tn]
                    results['tp']                   = [tp]
                    results['fn']                   = [fn]
                    results['fp']                   = [fp]
                    results['y_true']               = [','.join(y_true[:,-1].astype(int).astype(str))]
                    gc.collect()
                    print(f'{conscious_state_source}-->{conscious_state_target}, {roi_name}, {model_name}, roc_auc = {roc_auc:.4f}')
                    results_to_save                 = pd.DataFrame(results)
                    results_to_save.to_csv(os.path.join(output_dir,file_name),index = False)
                    print(f'saving {os.path.join(output_dir,file_name)}')
