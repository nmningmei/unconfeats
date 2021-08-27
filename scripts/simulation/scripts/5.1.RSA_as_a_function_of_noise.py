#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:08:12 2020
@author: nmei
"""

import os
import gc
import numpy as np
import pandas as pd

import torch
#from torch.utils.data import Dataset
from torchvision import transforms,datasets

from sklearn import metrics
from sklearn.utils import shuffle

from glob import glob
from tqdm import tqdm

from utils_deep import (data_loader,
                        createLossAndOptimizer,
                        behavioral_evaluate_with_path,
                        build_model,
                        hidden_activation_functions,
                        resample_ttest_2sample,
                        noise_fuc,
                        make_decoder,
                        decode_hidden_layer,
                        resample_ttest,
                        resample_behavioral_estimate
                        )
from matplotlib import pyplot as plt
#plt.switch_backend('agg')

from scipy.spatial import distance
from joblib import Parallel, delayed

print('set up random seeds')
torch.manual_seed(12345)


# experiment control
model_dir               = '../models'
train_folder            = 'greyscaled'
valid_folder            = 'experiment_images_grayscaled'
train_root              = f'../data/{train_folder}/'
valid_root              = f'../data/{valid_folder}'
print_train             = True #
image_resize            = 128
batch_size              = 8
lr                      = 1e-4
n_epochs                = int(1e3)
device                  = 'cpu'
pretrain_model_name     = 'mobilenet'
hidden_units            = 2
hidden_func_name        = 'relu'
hidden_activation       = hidden_activation_functions(hidden_func_name)
hidden_dropout          = 0.
patience                = 5
output_activation       = 'softmax'
model_saving_name       = f'{pretrain_model_name}_{hidden_units}_{hidden_func_name}_{hidden_dropout}_{output_activation}'
testing                 = True #
n_experiment_runs       = 1000

n_noise_levels          = 50
n_keep_going            = 32
start_decoding          = False
to_round                = 9

results_dir             = '../stability/'
if not os.path.exists(results_dir):
    os.mkdir(results_dir)
if not os.path.exists(os.path.join(results_dir,model_saving_name)):
    os.mkdir(os.path.join(results_dir,model_saving_name))

if output_activation   == 'softmax':
    output_units        = 2
    categorical         = True
elif output_activation == 'sigmoid':
    output_units        = 1
    categorical         = False

if not os.path.exists(os.path.join(model_dir,model_saving_name)):
    os.mkdir(os.path.join(model_dir,model_saving_name))

if torch.cuda.is_available():torch.cuda.empty_cache();torch.cuda.manual_seed(12345);
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device:{device}')

noise_levels    = np.concatenate([[0],[item for item in np.logspace(-1,3,n_noise_levels)]])

print(noise_levels)
df_to_save = dict( model_name           = [],
                   hidden_units         = [],
                   hidden_activation    = [],
                   output_activation    = [],
                   noise_level          = [],
                   score_mean           = [],
                   score_std            = [],
                   chance_mean          = [],
                   chance_std           = [],
                   pval                 = [],
                   dropout              = [],
                   )
for var in noise_levels:
    var = round(var,to_round)
    if True:#var not in np.array(df_to_save['noise_level']).round(to_round):
        print(f'\nworking on {var:1.1e}')
        

        # define augmentation function + noise function
        augmentations = {
                'visualize':transforms.Compose([
                transforms.Resize((image_resize,image_resize)),
                transforms.RandomHorizontalFlip(p = 0.5),
                transforms.RandomRotation(25,),
                transforms.RandomVerticalFlip(p = 0.5,),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: noise_fuc(x,var)),
                ]),
                'valid':transforms.Compose([
                transforms.Resize((image_resize,image_resize)),
                transforms.RandomHorizontalFlip(p = 0.5),
                transforms.RandomRotation(25,),
                transforms.RandomVerticalFlip(p = 0.5,),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: noise_fuc(x,var)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]),
            }

        valid_loader        = data_loader(
                valid_root,
                augmentations   = augmentations['valid'],
                batch_size      = batch_size,
                return_path     = True,
                # here I turn on the shuffle like it is in a real experiment
                )
        secondary_map = {}
        category_map = {}
        for item in glob(os.path.join(valid_root,"*","*","*.jpg")):
            k = item.split('/')
            _category,_subcategory,_temp = k[-3],k[-2],k[-1]
            secondary_map[_temp.split('.')[0]] = _subcategory
            category_map[_temp.split('.')[0]] = _category
        visualize_loader    = data_loader(
                valid_root,
                augmentations   = augmentations['visualize'],
                batch_size      = 2 * batch_size,
                )
        # load model architecture
        print('loading the trained model')
        model_to_train      = build_model(
                        pretrain_model_name,
                        hidden_units,
                        hidden_activation,
                        hidden_dropout,
                        output_units,
                        )
        model_to_train.to(device)
        for params in model_to_train.parameters():
            params.requires_grad = False
        
        f_name              = os.path.join(model_dir,model_saving_name,model_saving_name+'.pth')
        # load trained model
        model_to_train      = torch.load(f_name)
        loss_func,optimizer = createLossAndOptimizer(model_to_train,learning_rate = lr)
        
        # evaluate the model
        y_trues,y_preds,scores,features,labels,items = behavioral_evaluate_with_path(
                                                        model_to_train,
                                                        n_experiment_runs,
                                                        loss_func,
                                                        valid_loader,
                                                        device,
                                                        categorical         = categorical,
                                                        output_activation   = output_activation,
                                                        image_type          = f'{var:1.1e} noise',
                                                        )
        print('evaluate behaviroal performation')
        # estimate chance level scores
        np.random.seed(12345)
        yy_trues        = torch.cat(y_trues).detach().cpu().numpy()
        yy_preds        = torch.cat(y_preds).detach().cpu().numpy()
        chance_scores   = resample_behavioral_estimate(yy_trues,yy_preds,shuffle = True)

        pval            = resample_ttest_2sample(scores,chance_scores,
                                                 match_sample_size = False,
                                                 one_tail = False,
                                                 n_permutation = int(1e5),
                                                 n_jobs = -1,
                                                 verbose = 1,
                                                 )
        
        # save the features and labels from the hidden layer
        if device == 'cpu':
            gc.collect()
            def _unpack(pack):
                return torch.cat(pack).detach().cpu().numpy()
            decode_features = Parallel(n_jobs =  -1,verbose = 1,)(delayed(_unpack)(**{
                    'pack':run}) for run in features)
            decode_labels   = Parallel(n_jobs =  -1,verbose = 1,)(delayed(_unpack)(**{
                    'pack':run}) for run in labels)
            decode_items    = Parallel(n_jobs =  -1,verbose = 1,)(delayed(_unpack)(**{
                    'pack':run}) for run in items)
        else:
            decode_features = [torch.cat(run).detach().cpu().numpy() for run in features]
            decode_labels   = [torch.cat(run).detach().cpu().numpy() for run in labels]
            decode_items    = [np.concatenate(run) for run in items]
        
        gc.collect()
        def _process(_features,_labels,_items):
            if categorical:
                _labels = _labels[:,-1]
            df_for_sort = pd.DataFrame(_items.reshape(-1,1),columns = ['items'])
            df_for_sort['subcategory'] = df_for_sort['items'].map(secondary_map)
            df_for_sort['targets'] = df_for_sort['items'].map(category_map)
            
            idx_sort = df_for_sort.sort_values(['targets','subcategory','items']).index.values
            temp = _features[idx_sort]
            return distance.pdist(temp - temp.mean(1).reshape(-1,1),'cosine')
        gc.collect()
        RDMs = Parallel(n_jobs = -1,verbose = 1)(delayed(_process)(**{
                '_features':_features,
                '_labels':_labels,
                '_items':_items}) for _features,_labels,_items in zip(decode_features,decode_labels,decode_items))
        
        RDMs = np.array(RDMs)
        
        print('computing RDM of RDMs... ...')
        RDM_of_RDMs = distance.pdist(RDMs - RDMs.mean(1).reshape(-1,1),'cosine')
        
#        df_to_save = pd.DataFrame(RDM_of_RDMs.reshape(-1,1),columns = ['RDM'])
        gc.collect()
        df_to_save['model_name'        ].append(pretrain_model_name)
        df_to_save['hidden_units'      ].append(hidden_units)
        df_to_save['hidden_activation' ].append(hidden_func_name)
        df_to_save['output_activation' ].append(output_activation)
        df_to_save['noise_level'       ].append(round(var,to_round))
        df_to_save['score_mean'        ].append(np.mean(scores))
        df_to_save['score_std'         ].append(np.std(scores))
        df_to_save['chance_mean'       ].append(.5)
        df_to_save['chance_std'        ].append(0.)
        df_to_save['pval'              ].append(pval)
        df_to_save['dropout'           ].append(hidden_dropout)
        RDMs_name = f'stability_{var:1.3e}.npy'
        performance_name = f'score{var:1.3e}.npy'
        feature_name = f'feature_{var:1.3e}.npy'
        label_name = f'label_{var:1.3e}.npy'
        print(f'saving {RDMs_name}')
        np.save(os.path.join(results_dir,model_saving_name,RDMs_name),
                RDM_of_RDMs)
        np.save(os.path.join(results_dir,model_saving_name,performance_name),
                scores)
        np.save(os.path.join(results_dir,model_saving_name,feature_name),
                decode_features)
        np.save(os.path.join(results_dir,model_saving_name,label_name),
                decode_labels)
        gc.collect()
        df_to_csv = pd.DataFrame(df_to_save)
df_to_csv.to_csv(os.path.join(results_dir,model_saving_name,'other_info.csv,'),index = False)
print('done')
