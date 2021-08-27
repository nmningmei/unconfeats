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

from torchvision import transforms

from sklearn import metrics
from sklearn.utils import shuffle

from utils_deep import (data_loader,
                        createLossAndOptimizer,
                        behavioral_evaluate,
                        build_model,
                        hidden_activation_functions,
                        resample_ttest_2sample,
                        noise_fuc,
                        make_decoder,
                        decode_hidden_layer
                        )

print('set up random seeds')
torch.manual_seed(12345)
#if torch.cuda.is_available():torch.cuda.empty_cache();torch.cuda.manual_seed(12345);
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# experiment control
model_dir           = '../models'
train_folder        = 'grayscaled'
valid_folder        = 'experiment_images_grayscaled'
train_root          = f'../data/{train_folder}/'
valid_root          = f'../data/{valid_folder}'
print_train         = True #
image_resize        = 128
batch_size          = 8
lr                  = 1e-4
n_epochs            = int(1e3)
device              = 'cpu'
pretrain_model_name = 'alexnet'
hidden_units        = 100
hidden_func_name    = 'relu'
hidden_activation   = hidden_activation_functions(hidden_func_name)
hidden_dropout      = 0.5
patience            = 5
output_activation   = 'softmax'
model_saving_name   = f'{pretrain_model_name}_{hidden_units}_{hidden_func_name}_{hidden_dropout}_{output_activation}'
testing             = True #
n_experiment_runs   = 20

n_noise_levels      = 30
n_keep_going        = 8
CNN_dir             = '../results'

results_dir         = '../decode_results/'
if not os.path.exists(results_dir):
    os.mkdir(results_dir)
if not os.path.exists(os.path.join(results_dir,model_saving_name)):
    os.mkdir(os.path.join(results_dir,model_saving_name))

if output_activation == 'softmax':
    output_units    = 2
    categorical     = True
elif output_activation == 'sigmoid':
    output_units    = 1
    categorical     = False

if not os.path.exists(os.path.join(model_dir,model_saving_name)):
    os.mkdir(os.path.join(model_dir,model_saving_name))

print(f'device:{device}')

noise_levels        = np.concatenate([[0],[item for item in np.logspace(-1,5,n_noise_levels)]])

csv_saving_name     = os.path.join(CNN_dir,model_saving_name,'performance_results.csv')

df_CNN              = pd.read_csv(csv_saving_name)

for ii_row,row in df_CNN.iterrows():
    noise_level = row['noise_level']
    noise_folder  = os.path.join(CNN_dir,model_saving_name,f'{noise_level:1.1e}')
    if row['score_mean'] < 0.6:
        features = np.load(os.path.join(noise_folder,'features.npy'))
        labels = np.load(os.path.join(noise_folder,'labels.npy'))

        for decoder_name in ['linear-SVM','RBF-SVM','RF','logit']:
            decoder = make_decoder(decoder_name,n_jobs = 1)
            res = decode_hidden_layer(decoder,features,labels,n_splits = 50,test_size = 0.2,categorical = categorical,output_activation = output_activation,)

            print(f'{noise_level:11e},{decoder_name},CNN = {row["score_mean"]:.2f}+/-{row["score_std"]:.2f},{res["test_score"].mean():.3f}+/-{res["test_score"].std():.3f}')
    else:
        print(f'{noise_level:1.1e},CNN = {row["score_mean"]:.3f},why bother?')
