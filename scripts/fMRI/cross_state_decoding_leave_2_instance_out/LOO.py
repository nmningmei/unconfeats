#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 13:49:33 2019

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
from tqdm                    import tqdm
#from sklearn.utils           import shuffle
from shutil                  import copyfile
copyfile('../../../utils.py','utils.py')
from utils                   import (
#                                     customized_partition,
#                                     check_train_test_splits,
#                                     check_train_balance,
                                     build_model_dictionary,
                                     Find_Optimal_Cutoff,
                                     LOO_partition
                                     )
from sklearn.model_selection import cross_validate,GridSearchCV,StratifiedShuffleSplit
from sklearn                 import metrics
from sklearn.exceptions      import ConvergenceWarning,UndefinedMetricWarning
#from sklearn.utils.testing   import ignore_warnings
from collections             import OrderedDict,Counter
from joblib                  import Parallel,delayed

# interchangable part:
sub                     = 'sub-04'
idx                     = 8
conscious_state_source  = 'unconscious'
conscious_state_target  = 'unconscious'

stacked_data_dir        = '../../../../data/BOLD_average_BOLD_average_lr/{}/'.format(sub)
output_dir              = '../../../../results/MRI/nilearn/decoding_Gridsearch/{}'.format(sub)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
BOLD_data               = np.sort(glob(os.path.join(stacked_data_dir,'*BOLD.npy')))
event_data              = np.sort(glob(os.path.join(stacked_data_dir,'*.csv')))

model_names             = [
        'None + Linear-SVM','None + Dummy',
#        'None + Ensemble-SVMs',
#        'None + KNN',
#        'None + Tree',
#        'PCA + Dummy',
#        'PCA + Linear-SVM',
#        'PCA + Ensemble-SVMs',
#        'PCA + KNN',
#        'PCA + Tree',
#        'Mutual + Dummy',
#        'Mutual + Linear-SVM',
#        'Mutual + Ensemble-SVMs',
#        'Mutual + KNN',
#        'Mutual + Tree',
#        'RandomForest + Dummy',
#        'RandomForest + Linear-SVM',
#        'RandomForest + Ensemble-SVMs',
#        'RandomForest + KNN',
#        'RandomForest + Tree',
        ]
param_grid = {
#               'calibratedclassifiercv__base_estimator__penalty':['l1','l2'],
              'calibratedclassifiercv__base_estimator__C':np.logspace(-10,0,11)}
scorer = metrics.make_scorer(metrics.log_loss,needs_proba = True,greater_is_better = False)
#build_model_dictionary().keys()
label_map               = {'Nonliving_Things':[0,1],
                           'Living_Things':   [1,0]}
average                 = True
n_jobs                  = -1


np.random.seed(12345)
BOLD_name,df_name       = BOLD_data[idx],event_data[idx]
BOLD                    = np.load(BOLD_name)
df_event                = pd.read_csv(df_name)
roi_name                = df_name.split('/')[-1].split('_events')[0]
print(roi_name)


idx_unconscious_source  = df_event['visibility'] == conscious_state_source
data_source             = BOLD[idx_unconscious_source]
df_data_source          = df_event[idx_unconscious_source].reset_index(drop=True)
df_data_source['id']    = df_data_source['session'] * 1000 + df_data_source['run'] * 100 + df_data_source['trials']
targets_source          = np.array([label_map[item] for item in df_data_source['targets'].values])[:,-1]

idx_unconscious_target  = df_event['visibility'] == conscious_state_target
data_target             = BOLD[idx_unconscious_target]
df_data_target          = df_event[idx_unconscious_target].reset_index(drop=True)
df_data_target['id']    = df_data_target['session'] * 1000 + df_data_target['run'] * 100 + df_data_target['trials']
targets_target          = np.array([label_map[item] for item in df_data_target['targets'].values])[:,-1]

print(f'partitioning target: {conscious_state_target}')
idxs_train_target,idxs_test_target  = LOO_partition(df_data_target)
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
    for model_name in model_names:
        file_name           = f'({sub}_{roi_name}_{conscious_state_source}_{conscious_state_target}_{model_name}).csv'.replace(' + ','_')
        print(f'{model_name} {conscious_state_source} --> {conscious_state_target}')
        if not os.path.exists(os.path.join(output_dir,file_name)):
            np.random.seed(12345)
            features        = data_source.copy()
            targets         = targets_source.copy()
            
            pipeline        = build_model_dictionary(n_jobs            = 4,
                                                     remove_invariant  = True,
                                                     l1                = True,
                                                     )[model_name]
            if 'Dummy' not in model_name:
                pipeline = GridSearchCV(pipeline,
                                        param_grid = param_grid,
                                        scoring = scorer,
                                        n_jobs = 1,
                                        cv = 5,
                                        )
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", 
                                        category = ConvergenceWarning,
                                        module = "sklearn")
                gc.collect()
                res = cross_validate(pipeline,
                             features,
                             targets,
                             scoring            = 'roc_auc',
                             cv                 = zip(idxs_train_source[:18],idxs_test_source),
                             return_estimator   = True,
                             n_jobs             = n_jobs,
                             verbose            = 1,
                             )
                gc.collect()
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", 
                                        category = UndefinedMetricWarning)
                regs                = res['estimator']
                if 'Dummy' not in model_name:
                    params          = [list(estimator.best_params_.values())[0] for estimator in regs]
                else:
                    params          = [0 for estimator in regs]
                preds               = [estimator.predict_proba(data_target[idx_test])[:,-1] for idx_test,estimator in zip(idxs_test_target,regs)]
                roc_auc             = [metrics.roc_auc_score(targets_target[idx_test],y_pred,average = 'micro') for idx_test,y_pred in zip(idxs_test_target,preds)]
                threshold_          = [Find_Optimal_Cutoff(targets_target[idx_test],y_pred) for idx_test,y_pred in zip(idxs_test_target,preds)]
                mattews_correcoef   = [metrics.matthews_corrcoef(targets_target[idx_test],y_pred> thres_) for idx_test,y_pred,thres_ in zip(idxs_test_target,preds,threshold_)]
                f1_score            = [metrics.f1_score(targets_target[idx_test],y_pred > thres_) for idx_test,y_pred,thres_ in zip(idxs_test_target,preds,threshold_)]
                log_loss            = [metrics.log_loss(targets_target[idx_test],y_pred) for idx_test,y_pred in zip(idxs_test_target,preds)]
                
                
                temp                = np.array([metrics.confusion_matrix(targets_target[idx_test],y_pred > thres_).ravel() for idx_test,y_pred,thres_ in zip(idxs_test_target,preds,threshold_)])
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
            results['flip']                 = [False] * n_splits
            results['language']             = ['Image'] * n_splits
            results['transfer']             = [True] * n_splits
            results['best_C']               = params
            results['tn']                   = tn
            results['tp']                   = tp
            results['fn']                   = fn
            results['fp']                   = fp
            gc.collect()
            print(f'{conscious_state_source}-->{conscious_state_target}, {roi_name}, {model_name}, roc_auc = {np.mean(roc_auc):.4f}+/-{np.std(roc_auc):.4f}')
            asdf
            results_to_save                 = pd.DataFrame(results)
            results_to_save.to_csv(os.path.join(output_dir,file_name),index = False)
            print(f'saving {os.path.join(output_dir,file_name)}')
        else:
            print(file_name)
else:
    print('cross validation partition is wrong')

"""
idxs_target             = []
idxs_source_            = []
idxs_test_              = []
for idx_source,idx_test in zip(idxs_source,idxs_test):
    words_source = np.unique(df_data_source['labels'].values[idx_source])
    words_target = [word for word in np.unique(df_data_target['labels'].values) if (word not in words_source)]
    
    if len(words_target) > 1:
        idx_words_target, = np.where(df_data_target['labels'].apply(lambda x: x in words_target) == True)
        idxs_target.append(idx_words_target)
        idxs_source_.append(idx_source)
        idxs_test_.append(idx_test)
# class check
idxs_target_temp = []
idxs_source_temp = []
idxs_source_test_temp = []
for idx_,idx_source_train,idx_source_test in zip(
                        idxs_target,
                        idxs_source_,
                        idxs_test_):
    temp = df_data_target.iloc[idx_]
    if len(np.unique(targets_target[idx_])) < 2:
        print(pd.unique(temp['targets']),pd.unique(temp['labels']),targets_target[idx_],)#np.array([label_map[item] for item in temp['targets'].values])[:,-1])
    else:
        idxs_target_temp.append(idx_)
        idxs_source_temp.append(idx_source_train)
        idxs_source_test_temp.append(idx_source_test)

idxs_target = idxs_target_temp
idxs_source_ = idxs_source_temp
idxs_test_ = idxs_source_test_temp

n_splits = len(idxs_target)
print(f'perform {n_splits} cross validation')
"""