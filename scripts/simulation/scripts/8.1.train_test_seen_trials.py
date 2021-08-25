#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:08:12 2020
@author: nmei
This is a template script for training a convolutional neural network to perform
a categorical task, discriminating living vs nonliving images in a fMRI experiment
The convolutional layers are from pre-trained networks that were trained on
imageNet dataset.
The goal of the script is to emulate the behavioral performance of human subjects
in discriminating the image without any noise or color.
The network we will use is:
    1. Convolutional layers -- pretrained
    2. Adaptive pooling -- pooling, making the convolutional outputs a 1-D vector
    3. hidden layer - K-units fully-connected layer
    4. hidden activation - varies
    5. output layer - 1 or 2 units full-connected layer
    6. output activation - varies
Experiment parameters:
    1. pretrain model
    2. hiddden layer units
    3. hidden activation
    4. output layer units
    5. output activation
Particularly:
    This is a Pytorch implementation, which is slightly different from a
    Tensorflow implementation. During training, the corresponding output activation
    function (softmax/sigmoid) are log-scaled (log-softmax/log-sigmoid) because
    that is what Pytorch implemented loss function asks for. But in end of predicting
    the probability of each class (living v.s. nonliving), softmax/sigmoid is used.
"""

import os
from glob import glob
from collections import OrderedDict
import numpy as np

import torch

from torchvision import transforms

from sklearn import metrics

from utils_deep import (data_loader,
                        createLossAndOptimizer,
                        train_loop,
                        validation_loop,
                        hidden_activation_functions,
                        behavioral_evaluate,
                        build_model
                        )

# experiment control
model_dir               = '../models'
train_folder            = 'greyscaled'
valid_folder            = 'experiment_images_greyscaled'
train_root              = f'../data/{train_folder}/'
valid_root              = f'../data/{valid_folder}'
print_train             = True #
image_resize            = 128
batch_size              = 8
lr                      = 1e-4
n_epochs                = int(1e3)
device                  = 'cpu'
pretrain_model_name     = 'vgg19_bn'
hidden_units            = 2
hidden_func_name        = 'relu'
hidden_activation       = hidden_activation_functions(hidden_func_name)
hidden_dropout          = 0.
patience                = 5
output_activation       = 'softmax'
model_saving_name       = f'{pretrain_model_name}_{hidden_units}_{hidden_func_name}_{hidden_dropout}_{output_activation}'
testing                 = True #
n_experiment_runs       = 20

if output_activation   == 'softmax':
    output_units        = 2
    categorical         = True
elif output_activation == 'sigmoid':
    output_units        = 1
    categorical         = False

if not os.path.exists(os.path.join(model_dir,model_saving_name)):
    os.mkdir(os.path.join(model_dir,model_saving_name))

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

print('set up random seeds')
torch.manual_seed(12345)
if torch.cuda.is_available():torch.cuda.empty_cache();torch.cuda.manual_seed(12345);
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device:{device}')

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

#model_to_train = torch.load(f_name)
#
#y_trues,y_preds,scores,features,labels = behavioral_evaluate(model_to_train,
#                                                             n_experiment_runs,
#                                                             loss_func,
#                                                             valid_loader,
#                                                             device,
#                                                             categorical = categorical,
#                                                             output_activation = output_activation,
#                                                             )
