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
                        decode_hidden_layer,
                        train_loop,
                        validation_loop
                        )
from collections import OrderedDict
from matplotlib import pyplot as plt

# experiment control
model_dir               = '../models'
train_folder            = 'grayscaled'
valid_folder            = 'experiment_images_grayscaled'
train_root              = f'../data/{train_folder}/'
valid_root              = f'../data/{valid_folder}'
print_train             = True #
image_resize            = 128
batch_size              = 8
lr                      = 1e-4
n_epochs                = int(1e3)
device                  = 'cpu'
pretrain_model_name     = 'resnet'
hidden_units            = 2
hidden_func_name        = 'relu'
hidden_activation       = hidden_activation_functions(hidden_func_name)
hidden_dropout          = 0.
patience                = 5
output_activation       = 'softmax'
model_saving_name       = f'{pretrain_model_name}_{hidden_units}_{hidden_func_name}_{hidden_dropout}_{output_activation}'
testing                 = True #
n_experiment_runs       = 20

n_noise_levels          = 50
n_keep_going            = 32

results_dir             = '../confidence_results/'
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

loss_func,optimizer                         = createLossAndOptimizer(model_to_train,learning_rate = lr)
if (not os.path.exists(f_name)) or (testing):
    augmentations = {
            'train':transforms.Compose([
            transforms.Resize((image_resize,image_resize)),
            transforms.RandomHorizontalFlip(p = 0.5),
            transforms.RandomRotation(45,),
            transforms.RandomVerticalFlip(p = 0.5,),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
            'valid':transforms.Compose([
            transforms.Resize((image_resize,image_resize)),
            transforms.RandomHorizontalFlip(p = 0.5),
            transforms.RandomRotation(25,),
            transforms.RandomVerticalFlip(p = 0.5,),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
        }

    train_loader        = data_loader(
            train_root,
            augmentations   = augmentations['train'],
            batch_size      = batch_size,
            )
    valid_loader        = data_loader(
            valid_root,
            augmentations   = augmentations['valid'],
            batch_size      = batch_size,
            )
    best_valid_loss                             = torch.from_numpy(np.array(np.inf))
    losses = []
    for idx_epoch in range(n_epochs):
        # train
        print('training ...')
        train_loss                              = train_loop(
                                                    net                 = model_to_train,
                                                    loss_func           = loss_func,
                                                    optimizer           = optimizer,
                                                    dataloader          = train_loader,
                                                    device              = device,
                                                    categorical         = categorical,
                                                    idx_epoch           = idx_epoch,
                                                    print_train         = print_train,
                                                    output_activation   = output_activation,
                                                    )
        print('validating ...')
        valid_loss,y_pred,y_true,features,labels= validation_loop(
                                                    net                 = model_to_train,
                                                    loss_func           = loss_func,
                                                    dataloader          = valid_loader,
                                                    device              = device,
                                                    categorical         = categorical,
                                                    output_activation   = output_activation,
                                                    )
        y_pred = torch.cat(y_pred)
        y_true = torch.cat(y_true)
        score = metrics.roc_auc_score(y_true.detach().cpu(),y_pred.detach().cpu())
        print(f'\nepoch {idx_epoch + 1}, loss = {valid_loss:6f},score = {score:.4f}')
        if valid_loss.cpu().clone().detach().type(torch.float64) < best_valid_loss:
            best_valid_loss = valid_loss.cpu().clone().detach().type(torch.float64)
            torch.save(model_to_train,f_name)
        else:
            model_to_train = torch.load(f_name)
        losses.append(best_valid_loss)

        if (len(losses) > patience) and (len(set(losses[-patience:])) == 1):
            break

model_to_train = torch.load(f_name)

print('set up random seeds')
torch.manual_seed(12345)
#if torch.cuda.is_available():torch.cuda.empty_cache();torch.cuda.manual_seed(12345);
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'device:{device}')

noise_levels        = np.concatenate([[0],[item for item in np.logspace(-1,2,n_noise_levels)]])

saving_name         = os.path.join(results_dir,model_saving_name,'trial_by_trial ({}).csv')

for var in noise_levels:
    var = round(var,5)
    # define augmentation function + noise function
    augmentations = {
            'visualize':transforms.Compose([
            transforms.Resize((image_resize,image_resize)),
            transforms.RandomHorizontalFlip(p = 0.5),
            transforms.RandomRotation(45,),
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
            # here I turn on the shuffle like it is in a real experiment
            )
    visualize_loader = data_loader(
            valid_root,
            augmentations = augmentations['visualize'],
            batch_size = 2 * batch_size,
            )
    # load model architecture
    print('loading the trained model')
    model_to_train = build_model(
                    pretrain_model_name,
                    hidden_units,
                    hidden_activation,
                    hidden_dropout,
                    output_units,
                    )
    model_to_train.to(device)
    for params in model_to_train.parameters():
        params.requires_grad = False

    f_name = os.path.join(model_dir,model_saving_name,model_saving_name+'.pth')
    # load trained model
    model_to_train      = torch.load(f_name)
    loss_func,optimizer = createLossAndOptimizer(model_to_train,learning_rate = lr)
    # evaluate the model
    y_trues,y_preds,scores,features,labels = behavioral_evaluate(
                                                    model_to_train,
                                                    n_experiment_runs,
                                                    loss_func,
                                                    valid_loader,
                                                    device,
                                                    categorical         = categorical,
                                                    output_activation   = output_activation,
                                                    image_type          = f'{var:1.1e} noise',
                                                    )
    # estimate chance level scores
    np.random.seed(12345)
    chance_scores   = [metrics.roc_auc_score(y_true.detach().cpu().numpy(),
                                           shuffle(y_pred.detach().cpu().numpy())) for y_true,y_pred in zip(
                            y_trues,y_preds)]
    pval            = resample_ttest_2sample(np.array(scores),np.array(chance_scores))

    results             = OrderedDict()

    if categorical:
        confidence = torch.cat(y_preds).cpu().numpy().max(1)
        results['y_pred'] = torch.cat(y_preds).detach().cpu().numpy()[:,-1]
        results['y_true'] = torch.cat(y_trues).detach().cpu().numpy()[:,-1]
    else:
        temp = torch.cat(y_preds).cpu().numpy()
        temp[temp < 0.5] = 1- temp[temp < 0.5]
        confidence = temp.copy().flatten()
        results['y_pred'] = torch.cat(y_preds).detach().cpu().numpy().flatten()
        results['y_true'] = torch.cat(y_trues).detach().cpu().numpy().flatten()

    results['confidence'] = confidence
    results_to_save = pd.DataFrame(results)
    results_to_save['pretrain_model_name'] = pretrain_model_name
    results_to_save['hidden_units'] = hidden_units
    results_to_save['hidden_activation'] = hidden_func_name
    results_to_save['hidden_dropout'] = hidden_dropout
    results_to_save['output_activation'] = output_activation
    results_to_save['block'] = np.repeat(np.arange(n_experiment_runs),96)
    results_to_save['noise_level'] = round(var,5)
    results_to_save.to_csv(saving_name.format(round(var,5)),index = False)

print('done')
