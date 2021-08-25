#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 12:04:11 2020

@author: nmei


RSA based on the average of volumes

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
try:
    from shutil                    import copyfile
    copyfile('../../../utils.py','utils.py')
except:
    pass
from utils                     import (groupby_average,
                                       add_track
                                                )
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
    for ii_sub in  [1,2,3,4,5,6,7]:
        for conscious_state in ['unconscious','conscious']:
            for model_name in ['AlexNet', 'DenseNet169', 'MobileNetV2', 'ResNet50', 'VGG19']:
                sub                 = f'sub-0{ii_sub}'
                radius              = 9
                stacked_data_dir    = '../../../../data/BOLD_whole_brain_averaged/{}/'.format(sub)
                mask_dir            = '../../../../data/MRI/{}/anat/ROI_BOLD'.format(sub)
                
                masks               = glob(os.path.join(mask_dir,'*.nii.gz'))
                whole_brain_mask    = f'../../../../data/MRI/{sub}/anat/ROI_BOLD/combine_BOLD.nii.gz'
                func_brain          = f'../../../../data/MRI/{sub}/func/example_func.nii.gz'
                feature_dir         = '../../../../data/computer_vision_features_no_background_caltech'
                label_map           = {'Nonliving_Things':[0,1],'Living_Things':[1,0]}
                average             = True
                output_dir          = '../../../../results/MRI/nilearn/RSA_searchlight_{}mm_average/{}/{}'.format(
                                    radius,model_name,sub)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                output_name         = f'{conscious_state}.nii.gz'
                np.random.seed(12345)
                if not os.path.exists(os.path.join(output_dir,output_name)):
                    df_data         = pd.read_csv(os.path.join(stacked_data_dir,
                                                               f'whole_brain_{conscious_state}.csv'))
                    df_data['id']   = df_data['session'] * 1000 + df_data['run'] * 100 + df_data['trials']
                    df_data         = df_data[df_data.columns[1:]]
                    BOLD_file       = os.path.join(stacked_data_dir,
                                                   f'whole_brain_{conscious_state}.nii.gz')
                    masker          = NiftiMasker(mask_img = whole_brain_mask,).fit()
                    BOLD_image      = masker.transform(BOLD_file)
                    
                    targets         = np.array([label_map[item] for item in df_data['targets']])[:,-1]
                    images          = df_data['paths'].apply(lambda x: x.split('.')[0] + '.npy').values
                    CNN_feature     = np.array([np.load(os.path.join(feature_dir,
                                                                     model_name,
                                                                     item)) for item in images])
                    groups          = df_data['labels'].values
                    
                    BOLD_average,df_average = groupby_average(BOLD_image,df_data,['labels'],)
                    feature_average,_ = groupby_average(CNN_feature,df_data,['labels'],)
                    BOLD_average = masker.inverse_transform(BOLD_average)
                    def _searchligh_RSA(input_image,
                                        CNN_feature,
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
                        sl.distribute([np.asarray(input_image.dataobj)], 
                                       np.asanyarray(load_fmri(whole_brain_mask).dataobj) == 1)
                        sl.broadcast(CNN_feature)
                        # run searchlight algorithm
                        global_outputs = sl.run_searchlight(sfn,pool_size = -1)
                        return global_outputs
                    for _ in range(10):
                        gc.collect()
                    print(f'working on {sub} {conscious_state} {model_name}')
                    global_outputs = _searchligh_RSA(BOLD_average,feature_average,)
                    gc.collect()
                    results_to_save = new_img_like(load_fmri(func_brain),np.array(global_outputs, dtype=np.float),)
                    results_to_save.to_filename(os.path.join(output_dir,output_name))






















