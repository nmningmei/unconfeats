#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 12:04:11 2020

@author: nmei
"""

import os
import gc
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas  as pd
import numpy   as np
#import seaborn as sns

from glob                      import glob
#from tqdm                      import tqdm
from copy                      import copy
#from sklearn.utils             import shuffle
try:
    from shutil                    import copyfile
    copyfile('../../../utils.py','utils.py')
except:
    pass
#from utils                     import (LOO_partition,
#                                       get_label_category_mapping,
#                                       get_label_subcategory_mapping,
#                                       make_df_axis)
#from sklearn.model_selection   import cross_validate,LeavePGroupsOut
#from sklearn.preprocessing     import MinMaxScaler
#from sklearn                   import metrics
#from sklearn.exceptions        import ConvergenceWarning
from sklearn.utils             import shuffle as sk_shuffle
#from collections               import OrderedDict
#from matplotlib                import pyplot as plt
from scipy.spatial             import distance
from scipy.stats               import spearmanr
from joblib                    import Parallel,delayed
from nibabel                   import load as load_fmri
from nilearn.image             import new_img_like
from sklearn                   import preprocessing
from nilearn.input_data        import NiftiMasker
#from nilearn.decoding          import SearchLight
#from sklearn.metrics           import make_scorer
from brainiak.searchlight.searchlight import Searchlight
from brainiak.searchlight.searchlight import Ball

def normalize(data,axis = 1):
    return data - data.mean(axis).reshape(-1,1)
# Define voxel function
def sfn(l, msk, myrad, bcast_var):
    """
    l: BOLD
    msk: mask array
    myrad: not use
    bcast_var: label -- CNN features
    """
    BOLD = l[0][msk,:].T.copy()
    model = bcast_var.copy()
#    print(BOLD.shape)
    # pearson correlation
    RDM_X   = distance.pdist(normalize(BOLD),'correlation')
    RDM_y   = distance.pdist(normalize(model),'correlation')
    D,p     = spearmanr(RDM_X ,RDM_y)
#    print(D)
    return D
if __name__ == "__main__":

    sub                 = 'sub-01'
    radius              = 6
    stacked_data_dir    = '../../../../data/BOLD_whole_brain_averaged/{}/'.format(sub)
    mask_dir            = '../../../../data/MRI/{}/anat/ROI_BOLD'.format(sub)
    
    masks               = glob(os.path.join(mask_dir,'*.nii.gz'))
    whole_brain_mask    = f'../../../../data/MRI/{sub}/anat/ROI_BOLD/combine_BOLD.nii.gz'
    func_brain          = f'../../../../data/MRI/{sub}/func/example_func.nii.gz'
    feature_dir         = '../../../../data/computer_vision_features_no_background_caltech'
    model_name          = 'DenseNet169'
    label_map           = {'Nonliving_Things':[0,1],'Living_Things':[1,0]}
    average             = True
    n_splits            = 1000
    n_jobs              = 16
    output_dir          = '../../../../results/MRI/nilearn/RSA_searchlight_{}mm/{}/{}'.format(
                        radius,model_name,sub)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    conscious_state     = 'unconscious'
    output_name         = f'{conscious_state}.nii.gz'
    if 'chance' in conscious_state:
        conscious_state,chance_level = conscious_state.split('_')
    np.random.seed(12345)
    if not os.path.exists(os.path.join(output_dir,output_name)):
        df_data         = pd.read_csv(os.path.join(stacked_data_dir,
                                                   f'whole_brain_{conscious_state}.csv'))
        df_data['id']   = df_data['session'] * 1000 + df_data['run'] * 100 + df_data['trials']
        df_data         = df_data[df_data.columns[1:]]
        BOLD_file       = os.path.join(stacked_data_dir,
                                       f'whole_brain_{conscious_state}.nii.gz')
        BOLD_image      = load_fmri(BOLD_file)
        if 'chance_level' in globals():
            masker = NiftiMasker(whole_brain_mask).fit()
            temp_data = masker.transform(BOLD_file)
            temp_array = np.zeros(temp_data.shape)
            for ii,row in enumerate(temp_data):
                temp_array[ii] = sk_shuffle(row)
            BOLD_image = masker.inverse_transform(temp_array)
            del temp_data
        
        targets         = np.array([label_map[item] for item in df_data['targets']])[:,-1]
        images          = df_data['paths'].apply(lambda x: x.split('.')[0] + '.npy').values
        CNN_feature     = np.array([np.load(os.path.join(feature_dir,
                                                         model_name,
                                                         item)) for item in images])
        groups          = df_data['labels'].values
        
        def _proc(df_data):
            df_picked = df_data.groupby('labels').apply(lambda x: x.sample(n = 1).drop('labels',axis = 1)).reset_index()
            df_picked = df_picked.sort_values(['targets','subcategory','labels'])
            idx_test    = df_picked['level_1'].values
            return idx_test
        print(f'partitioning data for {n_splits} folds')
        np.random.seed(12345)
        idxs = Parallel(n_jobs = n_jobs, verbose = 1)(delayed(_proc)(**{
                    'df_data':df_data,}) for _ in range(n_splits))
        gc.collect()
        
        def _searchligh_RSA(idx,
                            sl_rad = radius, 
                            max_blk_edge = radius - 1, 
                            shape = Ball,
                            min_active_voxels_proportion = 0,
                            ):
            sl = Searchlight(sl_rad = sl_rad, 
                             max_blk_edge = max_blk_edge, 
                             shape = shape,
                             min_active_voxels_proportion = min_active_voxels_proportion,
                             )
            sl.distribute([np.asanyarray(BOLD_image.dataobj)[:,:,:,idx]], 
                           np.asanyarray(load_fmri(whole_brain_mask).dataobj) == 1)
            sl.broadcast(CNN_feature[idx])
            # run searchlight algorithm
            global_outputs = sl.run_searchlight(sfn,pool_size = 1)
            return global_outputs
        for _ in range(10):
            gc.collect()
#        a = _searchligh_RSA(idxs[9])
#        asdf
        print(f'working on {conscious_state}')
        res = Parallel(n_jobs = n_jobs,verbose = 1,)(delayed(_searchligh_RSA)(**{
                'idx':idx}) for idx in idxs)
        
        gc.collect()
        results_to_save = np.zeros(np.concatenate([BOLD_image.shape[:3],[n_splits]]))
        for ii,item in enumerate(res):
            results_to_save[:,:,:,ii] = np.array(item, dtype=np.float)
        results_to_save = new_img_like(BOLD_image,results_to_save,)
        
        results_to_save.to_filename(os.path.join(output_dir,output_name))























