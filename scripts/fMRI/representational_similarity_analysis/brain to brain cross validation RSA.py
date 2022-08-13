#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 18:25:32 2021

@author: nmei
"""

import os,gc
from glob import glob
from tqdm import tqdm

import pandas as pd
import numpy  as np

from shutil import copyfile
copyfile('../../../utils.py','utils.py')
from utils                   import (
                                     groupby_average,
                                     partition_for_RSA
                                     )
from nilearn.input_data import NiftiMasker
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score,make_scorer

from scipy.stats               import spearmanr
from scipy.spatial             import distance
from nilearn.image             import new_img_like,load_img,index_img,smooth_img
from brainiak.searchlight.searchlight import Searchlight
from brainiak.searchlight.searchlight import Ball

def score_func(y, y_pred,tol = 1e-2,func = np.mean,is_train = True,**kwargs):
    """
    Customized scoring function for optimizing the ridge regression model
    """
    # compute the raw R2 score for each voxel
    temp        = r2_score(y,y_pred,multioutput = 'raw_values')
    # find how many voxel can be explained by the computational model
    n_positive  = np.sum(temp > tol)
    if n_positive > 0:
        score   = func(temp[temp > tol])
    else:
        score   = 0
    counter     = 0
    the_number  = n_positive
    while the_number > 0:
        the_number = the_number // 10
        counter += 1
    if is_train: # we want to find a balance between the variance explained and the positive voxels
        return 0.5 * score + 0.5 * (n_positive / 10 ** counter)
    else: # during testing, we don't care about the number of positive voxels
        return score

scorer = make_scorer(score_func, greater_is_better = True,)
average = False # average before correlation, should be false if LOO partition

def feature_normalize(data,axis = 1):
    return data - data.mean(axis).reshape(-1,1)

# Define voxel function
def searchlight_function_unit(sphere_bold_singals, mask, myrad, broadcast_variable):
    """
    sphere_bold_signals: BOLD data, [0]:source, [1]:target
    mask: mask array
    myrad: not used
    broadcast_variable: label -- CNN features: [0]:source,[1]:target
    """
#    BOLD_scaler = MinMaxScaler((-1,1))
    BOLD_source = sphere_bold_singals[0][mask,:].T.copy()
    BOLD_target = sphere_bold_singals[1][mask,:].T.copy()
    
#    BOLD_source = BOLD_scaler.fit_transform(BOLD_source)
    
    model_source = broadcast_variable[0].copy()
    model_target = broadcast_variable[1].copy()
    
    
    is_chance = broadcast_variable[2]
    if is_chance:
        BOLD_preds = np.random.normal(BOLD_source.mean(),BOLD_source.std(),size = BOLD_target.shape,)
    else:
        # train the ridge model on train data
#        ridge = Ridge(alpha = 1e3,random_state = 12345)
        ridge = RidgeCV(alphas = np.logspace(2,10,9),
                        normalize = False,
                        scoring = 'r2',
                        cv = 5,
                        )
        pipeline = make_pipeline(StandardScaler(),
                                 ridge,)
        pipeline.fit(model_source,BOLD_source)
        
        # convert feature space to BOLD space on test data
        BOLD_preds = pipeline.predict(model_target)
    
    
    # average predicted responses and the brain responeses for each word
    df_data_target = broadcast_variable[3].copy().reset_index(drop = True)
    temp,df_average = groupby_average([BOLD_target,
                                       BOLD_preds,
                                       ],
                                      df_data_target,
                                      groupby = ['trials'])
    # pearson correlation
    RDM_X   = distance.pdist(feature_normalize(temp[0]),'correlation')
    RDM_Y   = distance.pdist(feature_normalize(temp[1]),'correlation')
#    D1,p    = spearmanr(RDM_X, RDM_Y)
    D       = distance.cdist(RDM_X.reshape(1,-1),RDM_Y.reshape(1,-1),'correlation').flatten()
#    print(D)
    return D[0]

def _searchligh_RSA(BOLD_source,
                    BOLD_target,
                    feature_source,
                    feature_target,
                    is_chance,
                    df_event_target_test,
                    whole_brain_mask,
                    sl_rad                          = 9,
                    max_blk_edge                    = 9 - 1,
                    shape                           = Ball,
                    min_active_voxels_proportion    = 0,
                    ):
    """
    This is function is defined here is because the complex environment
    settings where Brainiak is installed
    
    Inputs
    ------------
    BOLD_source, ndarray, (n_samples, n_voxels)
    BOLD_target, ndarray, (n_samples, n_voxels)
    feature_source, ndarray, (n_samples,n_features)
    feature_target, ndarray, (n_samples,n_features)
    is_chance: bool,
    df_event_target_test, DataFrame,
    whole_brain_mask, 3D/4D Nifti object
        the standard whole brain mask
    sl_rad, int
        searchlight radius, in mm
    max_blk_edge, int
        unknown
    shape, Brainiak object
        the shape of the moving searchlight sphere used for extracting voxel values
    min_active_voxels_proportion, int or float
        unknown
    
    Output
    -------------
    global_outputs, ndarray
        X by Y by Z by 1
    """
    
    sl = Searchlight(sl_rad                         = sl_rad, 
                     max_blk_edge                   = max_blk_edge, 
                     shape                          = shape,
                     min_active_voxels_proportion   = min_active_voxels_proportion,
                     )
    # this is where the searchlight will be moving on
    print('distribute the variables')
    sl.distribute([np.asanyarray(BOLD_source.dataobj),
                   np.asanyarray(BOLD_target.dataobj),
                   ], 
                    np.asanyarray(load_img(whole_brain_mask).dataobj) == 1)
    # this is used by all the searchlights
    print('broadcast the variables')
    sl.broadcast([feature_source,feature_target,is_chance,df_event_target_test])
    # run searchlight algorithm
    print('run RSA')
    global_outputs = sl.run_searchlight(searchlight_function_unit,
                                        pool_size = -1, # use all the CPUs
                                        )
    return global_outputs

if __name__ == "__main__":
    [os.remove(item) for item in glob("core*")]
    sub                     = 'sub-01' # change subject
    radius                  = 9
    conscious_state_source  = 'conscious'# change source
    conscious_state_target  = 'unconscious'# change target
    model_name              = 'VGG19'# change model
    is_chance               = False # change chance
    working_dir             = '../../../../data/BOLD_whole_brain_averaged'
    mask_dir                = '../../../../data/MRI/{}/anat/ROI_BOLD'.format(sub)
    
    masks                   = glob(os.path.join(mask_dir,'*.nii.gz'))
    whole_brain_mask        = f'../../../../data/MRI/{sub}/anat/ROI_BOLD/combine_BOLD.nii.gz'
    func_brain              = f'../../../../data/MRI/{sub}/func/example_func.nii.gz'
    feature_dir             = '../../../../data/computer_vision_features_no_background_caltech'
        
    label_map               = {'Nonliving_Things':[0,1],'Living_Things':[1,0]}
    n_splits                = 24
    n_jobs                  = -1
    if is_chance:
        saving_name         = f'{sub}_{conscious_state_source}_{conscious_state_target}_{model_name}_{radius}mm_RSA_chance.nii.gz'
    else:
        saving_name         = f'{sub}_{conscious_state_source}_{conscious_state_target}_{model_name}_{radius}mm_RSA.nii.gz'
    output_dir              = '../../../../results/MRI/nilearn/RSA_searchlight_long_process_{}mm/{}/{}'.format(
                                radius,model_name,sub)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    BOLD_data_source        = os.path.join(working_dir,sub,f'whole_brain_{conscious_state_source}.nii.gz')
    event_data_source       = os.path.join(working_dir,sub,f'whole_brain_{conscious_state_source}.csv')
    BOLD_data_target        = os.path.join(working_dir,sub,f'whole_brain_{conscious_state_target}.nii.gz')
    event_data_target       = os.path.join(working_dir,sub,f'whole_brain_{conscious_state_target}.csv')
    
    df_data_source          = pd.read_csv(event_data_source)
    df_data_source['id']    = df_data_source['session'] * 1000 + df_data_source['run'] * 100 + df_data_source['trials']
    targets_source          = np.array([label_map[item] for item in df_data_source['targets'].values])
    
    df_data_target          = pd.read_csv(event_data_target)
    df_data_target['id']    = df_data_target['session'] * 1000 + df_data_target['run'] * 100 + df_data_target['trials']
    targets_target          = np.array([label_map[item] for item in df_data_target['targets'].values])
    
    
    gc.collect()
    # load the whole brain data
    masker              = NiftiMasker(mask_img = whole_brain_mask,).fit()
    BOLD_data_source    = masker.inverse_transform(masker.transform(BOLD_data_source))
    BOLD_data_target    = masker.inverse_transform(masker.transform(BOLD_data_target))
    
    # load features
    images_source       = df_data_source['paths'].apply(lambda x: x.split('.')[0] + '.npy').values
    feature_source      = np.array([np.load(os.path.join(feature_dir,
                                                         model_name,
                                                         item)) for item in images_source])
    images_target       = df_data_target['paths'].apply(lambda x: x.split('.')[0] + '.npy').values
    feature_target      = np.array([np.load(os.path.join(feature_dir,
                                                         model_name,
                                                         item)) for item in images_target])
    # partition the data into train and test sets
    idxs_train_source,idxs_test_source,idxs_train_target,idxs_test_target = partition_for_RSA(
                                                                            conscious_state_target,
                                                                            conscious_state_source,
                                                                            df_data_target,
                                                                            df_data_source,
                                                                            targets_target,
                                                                            targets_source,
                                                                            n_splits = n_splits,
                                                                            )
    if not os.path.exists(os.path.join(output_dir,saving_name)):
        # cross validate RSA
        results = []
        for idx_train,idx_test in tqdm(zip(idxs_train_source,idxs_test_target)):
            gc.collect()
            BOLD_source     = index_img(BOLD_data_source,idx_train)
            BOLD_target     = index_img(BOLD_data_target, idx_test)
            features_source = feature_source[idx_train]
            features_target = feature_target[idx_test]
            df_event_target_test = df_data_target.iloc[idx_test].reset_index(drop = True)
            res             = _searchligh_RSA(
                                  BOLD_source,
                                  BOLD_target,
                                  features_source,
                                  features_target,
                                  is_chance,
                                  df_event_target_test,
                                  whole_brain_mask,
                                  sl_rad                          = radius,
                                  max_blk_edge                    = radius - 1,
                                  shape                           = Ball,
                                  min_active_voxels_proportion    = 0,
                                  )
            gc.collect()
            res = new_img_like(func_brain,np.array(res, dtype = np.float64),)
            res = masker.transform(res)
            results.append(res[0])
            
            results_to_save = np.array(results)
            results_to_save = masker.inverse_transform(results_to_save)
        
            # save the results
            results_to_save.to_filename(os.path.join(output_dir,saving_name))
    else:# pick it up if not all the folds are done
        temp = masker.transform(os.path.join(output_dir,saving_name))
        done_folds = temp.shape[0]
        # cross validate RSA
        results = np.zeros((len(idxs_test_target),temp.shape[1]))
        for ii,row in enumerate(temp):
            results[ii] = row
        for fold,(idx_train,idx_test) in tqdm(enumerate(zip(idxs_train_source,idxs_test_target))):
            if fold >= done_folds - 1:
                gc.collect()
                BOLD_source     = index_img(BOLD_data_source,idx_train)
                BOLD_target     = index_img(BOLD_data_target, idx_test)
                features_source = feature_source[idx_train]
                features_target = feature_target[idx_test]
                df_event_target_test = df_data_target.iloc[idx_test].reset_index(drop = True)
                res             = _searchligh_RSA(
                                      BOLD_source,
                                      BOLD_target,
                                      features_source,
                                      features_target,
                                      is_chance,
                                      df_event_target_test,
                                      whole_brain_mask,
                                      sl_rad                          = radius,
                                      max_blk_edge                    = radius - 1,
                                      shape                           = Ball,
                                      min_active_voxels_proportion    = 0,
                                      )
                gc.collect()
                res = new_img_like(func_brain,np.array(res,  dtype = np.float64),)
                res = masker.transform(res)
                results[fold] = res[0]
                
                results_to_save = results[:fold].copy()
                results_to_save = masker.inverse_transform(results_to_save)
            
                # save the results
                results_to_save.to_filename(os.path.join(output_dir,saving_name))