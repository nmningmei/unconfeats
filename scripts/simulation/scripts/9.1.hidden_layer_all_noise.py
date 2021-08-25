#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:08:12 2020
@author: nmei

1. get output of the first layer
2. get output of the hidden layer
3. decode from both the first layer and the hidden layer
4. test through all noise levels
"""

import os
import gc
import numpy as np
import pandas as pd

import torch

from sklearn import metrics
from sklearn.decomposition import PCA

from joblib import Parallel,delayed

from utils_deep import (data_loader,
                        define_augmentations,
                        createLossAndOptimizer,
                        train_and_validation,
                        hidden_activation_functions,
                        behavioral_evaluate,
                        build_model,
                        resample_ttest_2sample,
                        noise_fuc,
                        make_decoder,
                        decode_hidden_layer,
                        resample_ttest,
                        resample_behavioral_estimate,
                        simple_augmentations
                        )

# experiment control
model_dir               = '../models'
train_folder            = 'greyscaled'
valid_folder            = 'experiment_images_greyscaled'
train_root              = f'../data/{train_folder}/'
valid_root              = f'../data/{valid_folder}'
print_train             = True # display the training process
image_resize            = 128
batch_size              = 8
lr                      = 1e-4
n_epochs                = int(1e3)
device                  = 'cpu'
pretrain_model_name     = 'vgg19_bn'
hidden_units            = 5
hidden_func_name        = 'selu'
hidden_activation       = hidden_activation_functions(hidden_func_name)
hidden_dropout          = 0.25
patience                = 5
output_activation       = 'softmax'
model_saving_name       = f'{pretrain_model_name}_{hidden_units}_{hidden_func_name}_{hidden_dropout}_{output_activation}'
testing                 = True #
n_experiment_runs       = 20
n_noise_levels          = 50
n_permutations          = int(1e4)
n_noise                 = 1

noise_levels    = np.concatenate([[0],[item for item in np.logspace(-1,3,n_noise_levels)]])

if output_activation   == 'softmax':
    output_units        = 2
    categorical         = True
elif output_activation == 'sigmoid':
    output_units        = 1
    categorical         = False

if not os.path.exists(os.path.join(model_dir,model_saving_name)):
    os.mkdir(os.path.join(model_dir,model_saving_name))

results_dir             = '../results/all_for_all'
if not os.path.exists(results_dir):
    os.mkdir(results_dir)
if not os.path.exists(os.path.join(results_dir,model_saving_name)):
    os.mkdir(os.path.join(results_dir,model_saving_name))

print('set up random seeds')
torch.manual_seed(12345)
if torch.cuda.is_available():torch.cuda.empty_cache();torch.cuda.manual_seed(12345);
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device:{device}')

# configure the model
model_to_train = build_model(
                pretrain_model_name,
                hidden_units,
                hidden_activation,
                hidden_dropout,
                output_units,
                )
model_to_train.to(device)
model_parameters                            = filter(lambda p: p.requires_grad, model_to_train.parameters())
params                                      = sum([np.prod(p.size()) for p in model_parameters])
print(pretrain_model_name,
#      model_to_train(next(iter(train_loader))[0]),
      f'total params = {params}')

f_name = os.path.join(model_dir,model_saving_name,model_saving_name+'.pth')

# train the model
loss_func,optimizer                         = createLossAndOptimizer(model_to_train,learning_rate = lr)
model_to_train                              = train_and_validation(
        model_to_train,
        f_name,
        output_activation,
        loss_func,
        optimizer,
        image_resize    = image_resize,
        device          = device,
        batch_size      = batch_size,
        n_epochs        = n_epochs,
        print_train     = print_train,
        patience        = patience,
        train_root      = train_root,
        valid_root      = valid_root,
        n_noise         = n_noise,
        noise_level     = None,
        )

model_to_train.to('cpu')
del model_to_train

np.random.seed(12345)
torch.manual_seed(12345)
to_round = 9

csv_saving_name     = os.path.join(results_dir,model_saving_name,'performance_results.csv')
results         = dict(model_name           = [],
                       hidden_units         = [],
                       hidden_activation    = [],
                       output_activation    = [],
                       dropout              = [],
                       noise_level          = [],
                       svm_score_mean       = [],
                       svm_score_std        = [],
                       svm_cnn_pval         = [],
                       cnn_score_mean       = [],
                       cnn_score_std        = [],
                       cnn_pval             = [],
                       )

# build the model from saved checkpoint
model_to_test = build_model(
                pretrain_model_name,
                hidden_units,
                hidden_activation,
                hidden_dropout,
                output_units,
                )
model_to_test = torch.load(f_name)
model_to_test.eval()
for param in model_to_test.parameters():
    param.requires_grad = False
loss_func,optimizer = createLossAndOptimizer(model_to_test,learning_rate = lr)
# start testing
for ii_var,var in enumerate(noise_levels):
    np.random.seed(12345)
    torch.manual_seed(12345)
    var = round(var,to_round)
    valid_loader        = data_loader(
            valid_root,
            augmentations   = simple_augmentations(image_resize,var),
            batch_size      = batch_size,
            # here I turn on the shuffle like it is in a real experiment
            )
    
    # evaluate the model
    y_trues,y_preds,features,labels = behavioral_evaluate(
                        model_to_test,
                        n_experiment_runs,
                        loss_func,
                        valid_loader,
                        device,
                        categorical = categorical,
                        output_activation = output_activation,
                        )
    
    behavioral_scores = resample_behavioral_estimate(y_trues,y_preds,int(1e3),shuffle = False)
    # print(var,np.mean(behavioral_scores))
    
    
    decoder = make_decoder('linear-SVM',n_jobs = 1)
    decode_features = torch.cat([torch.cat(item) for item in features]).detach().cpu().numpy()
    decode_labels   = torch.cat([torch.cat(item) for item in labels  ]).detach().cpu().numpy()
    if len(decode_labels.shape) > 1:
        decode_labels = decode_labels[:,-1]
    gc.collect()
    res,_,svm_cnn_pval = decode_hidden_layer(decoder,decode_features,decode_labels,
                              n_splits = 50,
                              test_size = 0.2,)
    gc.collect()
    svm_cnn_scores = res['test_score']
    print(var,np.mean(behavioral_scores),np.mean(svm_cnn_scores),)
    
    
    print(f'finished {ii_var}')
    
    results['model_name'].append(pretrain_model_name)
    results['hidden_units'].append(hidden_units)
    results['hidden_activation'].append(hidden_func_name)
    results['output_activation'].append(output_activation)
    results['dropout'].append(hidden_dropout)
    results['noise_level'].append(var)
    results['svm_score_mean'].append(np.mean(svm_cnn_scores))
    results['svm_score_std'].append(np.std(svm_cnn_scores))
    results['svm_cnn_pval'].append(svm_cnn_pval)
    results['cnn_score_mean'].append(np.mean(behavioral_scores))
    results['cnn_score_std'].append(np.std(behavioral_scores))
    
    gc.collect()
    chance_level = Parallel(n_jobs = -1,verbose = 1)(delayed(resample_behavioral_estimate)(**{
        'y_true':y_trues,
        'y_pred':y_preds,
        'n_sampling':int(1e1),
        'shuffle':True,}) for _ in range(n_permutations))
    gc.collect()
    cnn_pval = (np.sum(np.array(chance_level).mean(1) >= np.mean(behavioral_scores)) + 1) / (n_permutations + 1)
    
    results['cnn_pval'].append(cnn_pval)
    gc.collect()
    results_to_save = pd.DataFrame(results)
    results_to_save.to_csv(os.path.join(results_dir,model_saving_name,'decodings.csv'),index = False)




