#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 06:18:45 2021

@author: nmei
"""
import os
import gc
import numpy as np
import pandas as pd

import torch
from torchvision import transforms

from sklearn import metrics
from sklearn.utils import shuffle as sk_shuffle
from sklearn.model_selection import StratifiedShuffleSplit,cross_validate

from glob import glob
from tqdm import tqdm

from utils_deep import (hidden_activation_functions,
                        build_model,
                        simple_augmentations,
                        createLossAndOptimizer,
                        train_and_validation,
                        data_loader,
                        validation_loop,
                        decode_and_visualize_hidden_representations,
                        noise_fuc,
                        modified_model
                        )

from joblib import Parallel,delayed

from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style('white')
sns.set_context('poster')

print('set up random seeds')
torch.manual_seed(12345)

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
device                  = 'cpu'#torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pretrain_model_name     = 'vgg19_bn'
hidden_units            = 300
hidden_func_name        = 'relu'
hidden_activation       = hidden_activation_functions(hidden_func_name)
hidden_dropout          = 0. # in this experiment, we should not have any dropouts because I cannot solve the math...
patience                = 5
output_activation       = 'sigmoid'
model_saving_name       = f'{pretrain_model_name}_{hidden_units}_{hidden_func_name}_{hidden_dropout}_{output_activation}'
testing                 = True #
# n_experiment_runs       = 1000

n_noise_levels          = 50
# n_keep_going            = 32
# start_decoding          = False
to_round                = 9

results_dir             = '../results/use_hinge_loss'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
if not os.path.exists(os.path.join(model_dir,model_saving_name)):
    os.makedirs(os.path.join(model_dir,model_saving_name))

if not os.path.exists(os.path.join(model_dir,model_saving_name)):
    os.mkdir(os.path.join(model_dir,model_saving_name))
f_name                  = os.path.join(model_dir,model_saving_name,model_saving_name+'.pth')

# if torch.cuda.is_available():torch.cuda.empty_cache();torch.cuda.manual_seed(12345);
# print(f'device:{device}')

# initialize the model and some parameters
if output_activation   == 'softmax':
    output_units        = 2
    categorical         = True
elif output_activation == 'sigmoid':
    output_units        = 1
    categorical         = False
elif output_activation == 'hinge':
    output_units        = 1
    categorical         = False
# build the model, loss fuction, and optimizer
model_to_train = build_model(
    pretrain_model_name,
    hidden_units,
    hidden_activation,
    hidden_dropout,
    output_units,
    )

augmentations = {
            'train':simple_augmentations(image_resize,noise_level = None),
            'valid':simple_augmentations(image_resize,noise_level = None),
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
from torch.autograd import Variable
from torch.optim import Adam
optimizer = Adam([params for params in model_to_train.parameters()],
                 lr = lr,
                 weight_decay = 1e-6,
                 )

best_loss = torch.tensor(float('inf'),dtype = torch.float64)
train_count = 0
for idx_epochs in range(1000):
    # train
    model_to_train.to(device).train(True)
    iterator = tqdm(enumerate(train_loader))
    train_loss = 0
    
    for ii,(features,labels) in iterator:
        labels[labels == 0] = -1
    
        # shuffle the training batch
        idx_shuffle         = np.random.choice(features.shape[0],features.shape[0],replace = False)
        features            = features[idx_shuffle]
        labels              = labels[idx_shuffle]
    
        if ii + 1 <= len(train_loader): # drop last
            # load the data to memory
            inputs      = Variable(features).to(device)
            # one of the most important step, reset the gradients
            optimizer.zero_grad()
            # compute the outputs
            outputs,_   = model_to_train(inputs)
            loss = torch.mean(torch.clamp(1 - labels * outputs.squeeze(),min = 0))
            # backprop
            loss.backward()
            # 
            optimizer.step()
            #
            train_loss += loss.data
            iterator.set_description(f'train loss = {train_loss / ii:5f}')
    train_loss = train_loss / ii
    
    # valid
    model_to_train.eval()
    with torch.no_grad():
        valid_loss = 0
        for ii,(features,labels) in tqdm(enumerate(valid_loader)):
            labels[labels == 0] = -1
            if ii + 1 <= len(valid_loader):
                inputs      = Variable(features).to(device)
                outputs,_   = model_to_train(inputs)
                loss = torch.mean(torch.clamp(1 - labels * outputs.squeeze(),min = 0))
                valid_loss += loss.data
        valid_loss = valid_loss / ii
    print(f'valid loss = {valid_loss:5f}')
    if best_loss > valid_loss.detach().cpu().type(torch.float64):
        best_loss = valid_loss.detach().cpu().clone()
        train_count = 0
    else:
        train_count += 1
        if train_count > patience:
            print('finish training')
            break

model_to_train.eval()
for params in model_to_train.parameters():
    params.requires_grad = False

def _get_decode(net,
                train_root,
                batch_size = 8,
                image_resize = 128,
                device = 'cpu',
                categorical = None,
                output_activation = None,
                loss_func = torch.nn.BCELoss(),
                noise_level = None,
                ):
    # test the model on augmented images, no noise added
    transform_steps = simple_augmentations(image_resize,noise_level = noise_level,)
    DataLoader = data_loader(
        train_root,
        augmentations   = transform_steps,
        batch_size      = batch_size,
        )
    
    # get the hidden representations
    valid_loss,y_pred,y_true,features,labels = validation_loop(
        model_to_train,
        torch.nn.BCELoss(),
        DataLoader,
        'cpu',
        categorical = categorical,
        output_activation = output_activation,)
    features = torch.cat(features).detach().cpu().numpy()
    labels = torch.cat(labels).detach().cpu().numpy()
    y_true = torch.cat(y_true).detach().cpu().numpy()
    y_pred = torch.cat(y_pred).detach().cpu().numpy()
    return y_true,y_pred,features,labels

y_true,y_pred,features,labels = _get_decode(model_to_train,
                                            train_root,
                                            batch_size = batch_size,
                                            image_resize = 128,
                                            device = 'cpu',
                                            categorical = categorical,
                                            output_activation = output_activation,
                                            noise_level = None,
                                            )


fig,axes = plt.subplots(figsize = (8,12),
                        nrows = 2,
                        sharex = False,
                        sharey = False,
                        )
fig = decode_and_visualize_hidden_representations(
    fig,
    axes,
    y_true,
    y_pred,
    features,
    hidden_units = hidden_units,
    )
fig.tight_layout()

noise_levels    = np.concatenate([[0],[item for item in np.logspace(-1,3,n_noise_levels)]])
x_map           = {round(item,9):ii for ii,item in enumerate(noise_levels)}
inverse_x_map   = {round(value,9):key for key,value in x_map.items()}

import utils_deep
results = dict(noise_level = [],
               CNN_performance = [],
               SVM_performance = [],
               CNN_pval = [],
               SVM_pval = [],
               )
n_permutations = int(1e4)
for noise_level in noise_levels:
    transform_steps = simple_augmentations(image_resize,noise_level = noise_level)
    DataLoader = data_loader(
        train_root,
        augmentations   = transform_steps,
        batch_size      = batch_size,
        )
    
    # get the hidden representations
    valid_loss,y_pred,y_true,features,labels = validation_loop(
        model_to_train,
        torch.nn.BCELoss(),
        DataLoader,
        'cpu',
        categorical = categorical,
        output_activation = output_activation,)
    features = torch.cat(features).detach().cpu().numpy()
    y_true = torch.cat(y_true).detach().cpu().numpy()
    y_pred = torch.cat(y_pred).detach().cpu().numpy()
    score = metrics.roc_auc_score(y_true,y_pred,)
    y_rand = np.random.choice(y_pred.flatten(),size = (n_permutations,y_pred.shape[0]))
    def _proc(y_pred_):
        return metrics.roc_auc_score(y_true, y_pred_)
    gc.collect()
    chance = Parallel(n_jobs = -1,verbose = 1)(delayed(_proc)(**{'y_pred_':y_pred_}) for y_pred_ in y_rand)
    chance = np.array(chance)
    gc.collect()
    cnn_pval = (np.sum(chance >= score) + 1) / (n_permutations + 1)
    
    gc.collect()
    svm = utils_deep.make_decoder('linear-SVM')
    X,y = features.copy(),y_true.copy()
    cv = utils_deep.StratifiedShuffleSplit(n_splits = 50, test_size = 0.2, random_state = 12345)
    res = utils_deep.cross_validate(svm,X,y,cv = cv,scoring = 'roc_auc',
                                                     n_jobs = -1,verbose = 1)
    gc.collect()
    y_rand = np.random.uniform(0,1,size = (n_permutations,50,y.shape[0]))
    # y_rand = np.random.beta(0.5,0.5,size = (n_permutations,50,y.shape[0]))
    def _proc(y_pred_):
        return np.mean([metrics.roc_auc_score(y, item) for item in y_pred_])
    gc.collect()
    chance = Parallel(n_jobs = -1,verbose = 1)(delayed(_proc)(**{'y_pred_':y_pred_}) for y_pred_ in y_rand)
    chance = np.array(chance)
    gc.collect()
    svm_pval = (np.sum(chance >= res['test_score'].mean()) + 1) / (n_permutations + 1)
    
    del y_rand
    del chance
    results['noise_level'].append(round(noise_level,9))
    results['CNN_performance'].append(score)
    results['CNN_pval'].append(cnn_pval)
    results['SVM_performance'].append(res['test_score'].mean())
    results['SVM_pval'].append(svm_pval)


results = pd.DataFrame(results)

a = results[['noise_level','CNN_performance','CNN_pval']]
a.columns = ['noise_level','ROC AUC','p-value']
a['type'] = 'CNN'
a['x'] = a['noise_level'].map(x_map)
b = results[['noise_level','SVM_performance','SVM_pval']]
b.columns = ['noise_level','ROC AUC','p-value']
b['type'] = 'SVM'
b['x'] = b['noise_level'].map(x_map)
b['x'] = b['x'] + 0.75
df = pd.concat([a,b])
df['better than chance'] = 0.05 > df['p-value'] .values


fig,axes = plt.subplots(figsize = (12*2,8*2),
                      nrows = 2,
                      ncols = 2,
                      sharex = False,
                      sharey = False,
                      )
ax = axes[0][0]
ax = sns.scatterplot(x = 'x',
                     y = 'ROC AUC',
                     hue = 'type',
                     style = 'better than chance',
                     style_order = [True,False],
                     data = df,
                     ax = ax,
                     )
ax.set(xticks = np.arange(0,51,10),
       xticklabels = noise_levels[::10].round(2),
       xlabel = 'Noise level',
       ylabel = 'ROC AUC',
       )
ax.axhline(0.5,linestyle = '--',color = 'black',alpha = 0.5,)
from sklearn.decomposition import PCA
if True:
    ax = axes[0][1]
    noise_level = 0.1 # minmum noise
    y_true,y_pred,features,labels = _get_decode(model_to_train,
                                                train_root,
                                                batch_size = batch_size,
                                                image_resize = image_resize,
                                                device = 'cpu',
                                                categorical = categorical,
                                                output_activation = output_activation,
                                                noise_level = noise_level,
                                                )
    gc.collect()
    svm = utils_deep.make_decoder('linear-SVM')
    X,y = features.copy(),y_true.copy()
    cv = StratifiedShuffleSplit(n_splits = 300, test_size = 0.2, random_state = 12345)
    res = cross_validate(svm,X,y, cv = cv,scoring = 'roc_auc',n_jobs = -1, verbose = 0)
    gc.collect()
    cnn_score = np.array([metrics.roc_auc_score(a,b) for a,b in zip(y_true.reshape(-1,96),
                                                                   y_pred.reshape(-1,96))])
    svm_score = res['test_score']
    if features.shape[1] > 2:
        features = PCA(n_components = 2,random_state = 12345).fit_transform(features)
    sns.scatterplot(x = features[:,0], y = features[:,1],hue = y_true,ax = ax,
                    hue_order = [0,1],)
    ax.set(xlabel = 'D 1',
           ylabel = 'D 2',
           title = f'noise = {noise_level:.2f}\nCNN = {cnn_score.mean():.2f}\nSVM = {svm_score.mean():.2f}',)
    leg = ax.get_legend()
    for t,l in zip(leg.texts,['Animate','Inanimate']):t.set_text(l)

if True:
    ax = axes[1][0]
    noise_level = 3.56 # median noise
    y_true,y_pred,features,labels = _get_decode(model_to_train,
                                                train_root,
                                                batch_size = batch_size,
                                                image_resize = image_resize,
                                                device = 'cpu',
                                                categorical = categorical,
                                                output_activation = output_activation,
                                                noise_level = noise_level,
                                                )
    gc.collect()
    svm = utils_deep.make_decoder('linear-SVM')
    X,y = features.copy(),y_true.copy()
    cv = StratifiedShuffleSplit(n_splits = 300, test_size = 0.2, random_state = 12345)
    res = cross_validate(svm,X,y, cv = cv,scoring = 'roc_auc',n_jobs = -1, verbose = 0)
    gc.collect()
    cnn_score = np.array([metrics.roc_auc_score(a,b) for a,b in zip(y_true.reshape(-1,96),
                                                                   y_pred.reshape(-1,96))])
    svm_score = res['test_score']
    if features.shape[1] > 2:
        features = PCA(n_components = 2,random_state = 12345).fit_transform(features)
    sns.scatterplot(x = features[:,0], y = features[:,1],hue = y_true,ax = ax,
                    hue_order = [0,1],)
    ax.set(xlabel = 'D 1',
           ylabel = 'D 2',
           title = f'noise = {noise_level:.2f}\nCNN = {cnn_score.mean():.2f}\nSVM = {svm_score.mean():.2f}',)
    leg = ax.get_legend()
    for t,l in zip(leg.texts,['Animate','Inanimate']):t.set_text(l)

if True:
    ax = axes[1][1]
    noise_level = 23.3 # high noise
    y_true,y_pred,features,labels = _get_decode(model_to_train,
                                                train_root,
                                                batch_size = batch_size,
                                                image_resize = image_resize,
                                                device = 'cpu',
                                                categorical = categorical,
                                                output_activation = output_activation,
                                                noise_level = noise_level,
                                                )
    gc.collect()
    svm = utils_deep.make_decoder('linear-SVM')
    X,y = features.copy(),y_true.copy()
    cv = StratifiedShuffleSplit(n_splits = 300, test_size = 0.2, random_state = 12345)
    res = cross_validate(svm,X,y, cv = cv,scoring = 'roc_auc',n_jobs = -1, verbose = 0)
    gc.collect()
    cnn_score = np.array([metrics.roc_auc_score(a,b) for a,b in zip(y_true.reshape(-1,96),
                                                                   y_pred.reshape(-1,96))])
    svm_score = res['test_score']
    if features.shape[1] > 2:
        features = PCA(n_components = 2,random_state = 12345).fit_transform(features)
    sns.scatterplot(x = features[:,0], y = features[:,1],hue = y_true,ax = ax,
                    hue_order = [0,1],)
    ax.set(xlabel = 'D 1',
           ylabel = 'D 2',
           title = f'noise = {noise_level:.2f}\nCNN = {cnn_score.mean():.2f}\nSVM = {svm_score.mean():.2f}',)
    leg = ax.get_legend()
    for t,l in zip(leg.texts,['Animate','Inanimate']):t.set_text(l)
    
fig.tight_layout()
results.to_csv(os.path.join(results_dir,f'CNN_SVM_{hidden_units}.csv'),index=False)
fig.savefig(os.path.join(results_dir,f'CNN_SVM_{hidden_units}.jpg'),
            bbox_inches = 'tight')

fig,ax = plt.subplots(figsize = (12,8),
                      )
ax = sns.scatterplot(x = 'x',
                     y = 'ROC AUC',
                     hue = 'type',
                     style = 'better than chance',
                     style_order = [True,False],
                     data = df,
                     ax = ax,
                     )
ax.set(xticks = np.arange(0,51,10),
       xticklabels = noise_levels[::10].round(2),
       xlabel = 'Noise level',
       ylabel = 'ROC AUC',
       )
ax.axhline(0.5,linestyle = '--',color = 'black',alpha = 0.5,)
fig.savefig(os.path.join(results_dir,f'CNN_SVM_score_{hidden_units}.jpg'),
            bbox_inches = 'tight')
"""
transform_steps = simple_augmentations(image_resize,noise_level = 3,)
DataLoader = data_loader(
    train_root,
    augmentations   = transform_steps,
    batch_size      = batch_size,
    )

# get the hidden representations
valid_loss,y_pred,y_true,features,labels = validation_loop(
    model_to_train,
    torch.nn.BCELoss(),
    DataLoader,
    'cpu',
    categorical = categorical,
    output_activation = output_activation,)
features = torch.cat(features).detach().cpu().numpy()
y_true = torch.cat(y_true).detach().cpu().numpy()
y_pred = torch.cat(y_pred).detach().cpu().numpy()

fig,axes = plt.subplots(figsize = (8,12),
                        nrows = 2,
                        sharex = False,
                        sharey = False,
                        )
fig = decode_and_visualize_hidden_representations(
    fig,
    axes,
    y_true,
    y_pred,
    features,
    hidden_units = hidden_units,
    )
fig.tight_layout()
"""