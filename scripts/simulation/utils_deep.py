
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 10:54:54 2020
@author: nmei
"""


import os
from glob import glob
from collections import OrderedDict
import pandas as pd
import numpy as np

import torch
from torch import nn,no_grad
from torch.utils import data
from torch.nn import functional as F
import torch.optim as optim
from torch.autograd import Variable

from torchvision.datasets import ImageFolder
from torchvision import transforms
import torchvision.models as Tmodels

from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC,SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedShuffleSplit,cross_validate,permutation_test_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle as sk_shuffle

try:
    from xgboost import XGBClassifier
except:
    pass
output_act_func_dict = {'softmax':F.softmax, # softmax dim = 1
                        'sigmoid':torch.sigmoid,}
probability_func_dict = {'softmax':F.softmax,    # softmax dim = 1
                         'sigmoid':torch.sigmoid}
softmax_dim = 1

#candidate models
def candidates(model_name):
    candidate_models    = dict(
            alexnet     = Tmodels.alexnet(pretrained=True),
            vgg19       = Tmodels.vgg19(pretrained=True),
            densenet    = Tmodels.densenet169(pretrained=True),
            inception   = Tmodels.inception_v3(pretrained=True),
            mobilenet   = Tmodels.mobilenet_v2(pretrained=True),
            resnet      = Tmodels.wide_resnet50_2(pretrained=True),
                            )
    return candidate_models[model_name]

def define_type(model_name):
    model_type          = dict(
            alexnet     = 'simple',
            vgg19       = 'simple',
            densenet    = 'simple',
            inception   = 'inception',
            mobilenet   = 'simple',
            resnet      = 'resnet',
            )
    return model_type[model_name]

def hidden_activation_functions(activation_func_name):
    funcs = dict(relu = nn.ReLU(),
                 selu = nn.SELU(),
                 elu = nn.ELU(),
                 sigmoid = nn.Sigmoid(),
                 tanh = nn.Tanh(),
                 linear = None,
                 )
    return funcs[activation_func_name]

def output_activation_functions(activation_func_name):
    funcs = dict(softmax = F.log_softmax,
                 sigmoid = F.logsigmoid,
                 )
    return funcs[activation_func_name]

class customizedDataset(ImageFolder):
    def __getitem__(self, idx):
        original_tuple  = super(customizedDataset,self).__getitem__(idx)
        path = self.imgs[idx][0]
        tuple_with_path = (original_tuple +  (path,))
        return tuple_with_path
    
def data_loader(data_root:str,
                augmentations:transforms    = None,
                batch_size:int              = 8,
                num_workers:int             = 2,
                shuffle:bool                = True,
                return_path:bool            = False,
                )->data.DataLoader:
    """
    Create a batch data loader from a given image folder.
    The folder must be organized as follows:
        main ---
             |
             -----class 1 ---
                         |
                         ----- image 1.jpeg
                         .
                         .
                         .
            |
            -----class 2 ---
                        |
                        ---- image 1.jpeg
                        .
                        .
                        .
            |
            -----class 3 ---
                        |
                        ---- image 1.jpeg
                        .
                        .
                        .
    Input
    --------------------------
    data_root: str, the main folder
    augmentations: torchvision.transformers.Compose, steps of augmentation
    batch_size: int, batch size
    num_workers: int, CPU --> GPU carrier, number of CPUs
    shuffle: Boolean, whether to shuffle the order
    return_pth: Boolean, lod the image paths

    Output
    --------------------------
    loader: DataLoader, a Pytorch dataloader object
    """
    if return_path:
        datasets = customizedDataset(
                root                        = data_root,
                transform                   = augmentations
                )
    else:
        datasets    = ImageFolder(
                root                        = data_root,
                transform                   = augmentations
                )
    loader      = data.DataLoader(
                datasets,
                batch_size                  = batch_size,
                num_workers                 = num_workers,
                shuffle                     = shuffle,
                )
    return loader

class easy_model(nn.Module):
    """
    Models are not created equally
    Some pretrained models are composed by a {feature} and a {classifier} component
    thus, they are very easy to modify and transfer learning

    Inputs
    --------------------
    pretrain_model: nn.Module, pretrained model object
    hidden_units: int, hidden layer units
    hidden_activation: nn.Module, activation layer
    hidden_dropout: float (0,1), dropout rate
    output_units: int, output layer units

    Outputs
    --------------------
    model: nn.Module, a modified model with new {classifier} component with
    {feature} frozen untrainable <-- this is done prior to feed to this function
    """
    def __init__(self,
                 pretrain_model,
                 hidden_units,
                 hidden_activation,
                 hidden_dropout,
                 output_units,
                 in_shape = (1,3,128,128),
                 ):
        super(easy_model,self).__init__()

        self.pretrain_model     = pretrain_model
        self.hidden_units       = hidden_units
        self.avgpool            = nn.AdaptiveAvgPool2d((1,1))
        self.in_shape           = in_shape
        self.in_features        = self.avgpool(self.pretrain_model.features(torch.rand(*self.in_shape))).shape[1]
        print(f'feature dim = {self.in_features}')
        self.hidden_layer       = nn.Linear(self.in_features,hidden_units)
        self.hidden_activation  = hidden_activation
        self.hidden_dropout     = hidden_dropout
        self.dropout            = nn.Dropout(p = hidden_dropout)
        self.output_units       = output_units
        self.output_layer       = nn.Linear(hidden_units,output_units)
#        self.normalize          = nn.Sigmoid()

    def forward(self,x,):
        features                = self.pretrain_model.features(x)
        pooling                 = self.avgpool(features)
        pooling                 = pooling.view(pooling.shape[0],-1)
        hidden                  = self.hidden_layer(pooling)
        if self.hidden_activation is not None:
            hidden                  = self.hidden_activation(hidden)
        if self.hidden_dropout > 0:
            hidden              = self.dropout(hidden)
        outputs                 = self.output_layer(hidden)
        return outputs,hidden

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class resnet_model(nn.Module):
    """
    Models are not created equally
    Some pretrained models are composed by a {feature} and a {fc} component
    thus, they are very easy to modify and transfer learning

    Inputs
    --------------------
    pretrain_model: nn.Module, pretrained model object
    hidden_units: int, hidden layer units
    hidden_activation: nn.Module, activation layer
    hidden_dropout: float (0,1), dropout rate
    output_units: int, output layer units

    Outputs
    --------------------
    model: nn.Module, a modified model with new {fc} component with
    {feature} frozen untrainable <-- this is done prior to feed to this function
    """

    def __init__(self,
                 pretrain_model,
                 hidden_units,
                 hidden_activation,
                 hidden_dropout,
                 output_units,
                 ):
        super(resnet_model,self).__init__()

        self.pretrain_model         = pretrain_model
        self.hidden_units           = hidden_units
        self.avgpool                = nn.AdaptiveAvgPool2d((1,1))
        in_features                 = self.pretrain_model.fc.in_features
        print(f'feature dim = {in_features}')
        self.hidden_layer           = nn.Linear(in_features,hidden_units)
        self.hidden_activation      = hidden_activation
        self.hidden_dropout         = hidden_dropout
        self.output_units           = output_units
        self.dropout                = nn.Dropout(p = hidden_dropout)
        self.output_layer           = nn.Linear(hidden_units,output_units)

    def forward(self,x):
        res_net = torch.nn.Sequential(*list(self.pretrain_model.children())[:-2])
        features = res_net(x)
        pooling                 = self.avgpool(features)
        pooling                 = pooling.view(pooling.shape[0],-1)
        hidden                  = -self.hidden_layer(pooling)# to avoid initial negatives
        if self.hidden_activation is not None:
            hidden                  = self.hidden_activation(hidden)
        if self.hidden_dropout > 0:
            hidden              = self.dropout(hidden)
        outputs                 = self.output_layer(hidden)
        return outputs,hidden

class inception_model(nn.Module):
    """
    Models are not created equally
    Some pretrained models are composed by a {feature} and a {fc} component,
    while having an {aux_logits} object

    Inputs
    --------------------
    pretrain_model: nn.Module, pretrained model object
    hidden_units: int, hidden layer units
    hidden_activation: nn.Module, activation layer
    hidden_dropout: float (0,1), dropout rate
    output_units: int, output layer units

    Outputs
    --------------------
    model: nn.Module, a modified model with new {fc} component with
    {feature} frozen untrainable <-- this is done prior to feed to this function
    """
    def __init__(self,
                 pretrain_model,
                 hidden_units,
                 hidden_activation,
                 hidden_dropout,
                 output_units,
                 ):
        super(inception_model,self).__init__()


        self.pretrain_model             = pretrain_model
        self.hidden_units               = hidden_units
        self.hidden_activation          = hidden_activation
        self.hidden_dropout             = hidden_dropout
        self.output_units               = output_units
        self.dropout                    = nn.Dropout(p = hidden_dropout)
        in_features                     = self.pretrain_model.fc.in_features
        print(f'feature dim = {in_features}')
        self.hidden_layer               = nn.Linear(in_features,self.hidden_units)
        self.output_layer               = nn.Linear(self.hidden_units,self.output_units)
        self.pretrain_model.aux_logits  = False
        self.pretrain_model.fc          = self.hidden_layer

    def forward(self,x):
        inception_net                   = self.pretrain_model
        hidden                          = inception_net(x)
        hidden                          = self.hidden_layer(hidden)
        if self.hidden_activation is not None:
            hidden                  = self.hidden_activation(hidden)
        if self.hidden_dropout > 0:
            hidden              = self.dropout(hidden)
        outputs                 = self.output_layer(hidden)

        return outputs,hidden

def build_model(pretrain_model_name,
                hidden_units,
                hidden_activation,
                hidden_dropout,
                output_units,
                ):
    pretrain_model      = candidates(pretrain_model_name)
    for params in pretrain_model.parameters():
        params.requires_grad = False
    if define_type(pretrain_model_name) == 'simple':
        model_to_train = easy_model(
                            pretrain_model      = pretrain_model,
                            hidden_units        = hidden_units,
                            hidden_activation   = hidden_activation,
                            hidden_dropout      = hidden_dropout,
                            output_units        = output_units,
                            )
    elif define_type(pretrain_model_name) == 'inception':
        model_to_train = inception_model(
                            pretrain_model      = pretrain_model,
                            hidden_units        = hidden_units,
                            hidden_activation   = hidden_activation,
                            hidden_dropout      = hidden_dropout,
                            output_units        = output_units,
                            )
    elif define_type(pretrain_model_name) == 'resnet':
        model_to_train = resnet_model(
                            pretrain_model      = pretrain_model,
                            hidden_units        = hidden_units,
                            hidden_activation   = hidden_activation,
                            hidden_dropout      = hidden_dropout,
                            output_units        = output_units,
                            )
    return model_to_train

def createLossAndOptimizer(net, learning_rate:float = 1e-4):
    """
    To create the loss function and the optimizer

    Inputs
    ----------------
    net: nn.Module, torch model class containing parameters method
    learning_rate: float, learning rate

    Outputs
    ----------------
    loss: nn.Module, loss function
    optimizer: torch.optim, optimizer
    """
    #Loss function
    loss        = nn.BCELoss()
    #Optimizer
    optimizer   = optim.Adam([params for params in net.parameters()],
                              lr = learning_rate,
                              weight_decay = 1e-6)

    return(loss, optimizer)

def train_loop(net,
               loss_func,
               optimizer,
               dataloader,
               device,
               categorical          = True,
               idx_epoch            = 1,
               print_train          = False,
               output_activation    = 'softmax',
               l2_lambda            = 0,
               l1_lambda            = 0,
               n_noise              = 4,
               ):
    """
    A for-loop of train the autoencoder for 1 epoch

    Inputs
    -----------
    net: nn.Module, torch model class containing parameters method
    loss_func: nn.Module, loss function
    optimizer: torch.optim, optimizer
    dataloader: torch.data.DataLoader
    device:str or torch.device, where the training happens
    categorical:Boolean, whether to one-hot the label according to the output activation function
    idx_epoch:int, for print
    print_train:Boolean, debug tool
    output_activation:string, activation function prior to loss function computation, calling the inner dictionary
    l2_lambda:float, L2 regularization lambda term
    l1_lambda:float, L1 regularization lambdd term
    n_noise: int, number of noise images to add to the training batch

    Outputs
    ----------------
    train_loss: torch.Float, average training loss
    net: nn.Module, the updated model

    """
    from tqdm import tqdm
    train_loss              = 0.
    output_activation_func  = output_act_func_dict[output_activation]
    # set the model to "train"
    net.to(device).train(True)
    # verbose level
    if print_train:
        iterator = enumerate(dataloader)
    else:
        iterator = tqdm(enumerate(dataloader))

    for ii,(features,labels) in iterator:
        if "Binary Cross Entropy" in loss_func.__doc__:
            labels = labels.float() # I indeed hate this

        # in order to have desired classification behavior, which is to predict
        # chance when no signal is present, we manually add some noise samples
        noise_generator     = torch.distributions.normal.Normal(0,10)
        noisy_features      = noise_generator.sample(features.shape)[:n_noise]
        noisy_labels        = torch.tensor([0.5] * labels.shape[0])[:n_noise]

        features            = torch.cat([features,noisy_features])
        labels              = torch.cat([labels,noisy_labels])

        # shuffle the training batch
        idx_shuffle         = np.random.choice(features.shape[0],features.shape[0],replace = False)
        features            = features[idx_shuffle]
        labels              = labels[idx_shuffle]

        if ii + 1 <= len(dataloader):
            # load the data to memory
            inputs      = Variable(features).to(device)
            # one of the most important step, reset the gradients
            optimizer.zero_grad()
            # compute the outputs
            outputs,_   = net(inputs)
            # compute the losses
            if categorical:
                outputs = output_activation_func(outputs.clone(),softmax_dim)
                labels  = torch.stack([labels,1- labels]).T
            else:
                outputs = output_activation_func(outputs.clone())
            loss_batch  = loss_func(outputs.to(device),labels.view(outputs.shape).to(device))
            # add L2 loss to the weights
            weight_norm = torch.norm(list(net.parameters())[-4],2)
            loss_batch  += l2_lambda * weight_norm
            # add L1 loss to the weights
            weight_norm = torch.norm(list(net.parameters())[-4],1)
            loss_batch  += l1_lambda * weight_norm
            # backpropagation
            loss_batch.backward()
            # modify the weights
            optimizer.step()
            # record the training loss of a mini-batch
            train_loss  += loss_batch.data
            if print_train:
#                score = metrics.roc_auc_score(labels.detach().cpu(),outputs.detach().cpu())
                print(f'epoch {idx_epoch+1}-{ii + 1:3.0f}/{100*(ii+1)/len(dataloader):2.3f}%,loss = {train_loss/(ii+1):.6f}')#, score = {score:.4f}')
    return train_loss/(ii+1)

def validation_loop(net,
                    loss_func,
                    dataloader,
                    device,
                    categorical = True,
                    output_activation = 'softmax',
                    ):
    """
    net:nn.Module, torch model object
    loss_func:nn.Module, loss function
    dataloader:torch.data.DataLoader
    device:str or torch.device
    categorical:Boolean, whether to one-hot labels
    output_activation:string, calling the activation function from an inner dictionary

    """
    from tqdm import tqdm
    probability_func        = probability_func_dict[output_activation]
    output_activation_func  = output_act_func_dict[output_activation]
    # specify the gradient being frozen and dropout etc layers will be turned off
    net.to(device).eval()
    with no_grad():
        valid_loss      = 0.
        y_pred          = []
        y_true          = []
        features,labels = [],[]
        for ii,(batch_features,batch_labels) in tqdm(enumerate(dataloader)):
            if "Binary Cross Entropy" in loss_func.__doc__:
                batch_labels = batch_labels.float()
            batch_labels.to(device)
            if ii + 1 <= len(dataloader):
                # load the data to memory
                inputs      = Variable(batch_features).to(device)
                # compute the outputs
                outputs,feature_   = net(inputs)
                # activation function for outputs
                if categorical:
                    y_pred.append(probability_func(outputs.clone(),softmax_dim))
                    outputs         = output_activation_func(outputs.clone(),softmax_dim)
                    batch_labels    = torch.stack([batch_labels,1- batch_labels]).T
                else:
                    y_pred.append(probability_func(outputs.clone()))
                    outputs         = output_activation_func(outputs.clone())
                # compute the losses
                loss_batch  = loss_func(outputs.to(device),batch_labels.view(outputs.shape).to(device))
                # record the validation loss of a mini-batch
                valid_loss  += loss_batch.data
                denominator = ii

                y_true.append(batch_labels)
                features.append(feature_)
                labels.append(batch_labels)
        valid_loss = valid_loss / (denominator + 1)
    return valid_loss,y_pred,y_true,features,labels

def validation_loop_with_path(net,
                              loss_func,
                              dataloader,
                              device,
                              categorical = True,
                              output_activation = 'softmax',
                              ):
    """
    net:nn.Module, torch model object
    loss_func:nn.Module, loss function
    dataloader:torch.data.DataLoader
    device:str or torch.device
    categorical:Boolean, whether to one-hot labels
    output_activation:string, calling the activation function from an inner dictionary

    """
    from tqdm import tqdm
    probability_func        = probability_func_dict[output_activation]
    output_activation_func  = output_act_func_dict[output_activation]
    # specify the gradient being frozen and dropout etc layers will be turned off
    net.to(device).eval()
    with no_grad():
        valid_loss      = 0.
        y_pred          = []
        y_true          = []
        features,labels,item = [],[],[]
        for ii,(batch_features,batch_labels,batch_path) in enumerate(dataloader):
            item.append([item.split('/')[-1].split('.')[0] for item in batch_path])
            if "Binary Cross Entropy" in loss_func.__doc__:
                batch_labels = batch_labels.float()
            batch_labels.to(device)
            if ii + 1 <= len(dataloader):
                # load the data to memory
                inputs      = Variable(batch_features).to(device)
                # compute the outputs
                outputs,feature_   = net(inputs)
                # activation function for outputs
                if categorical:
                    y_pred.append(probability_func(outputs.clone(),softmax_dim))
                    outputs         = output_activation_func(outputs.clone(),softmax_dim)
                    batch_labels    = torch.stack([batch_labels,1- batch_labels]).T
                else:
                    y_pred.append(probability_func(outputs.clone()))
                    outputs         = output_activation_func(outputs.clone())
                # compute the losses
                loss_batch  = loss_func(outputs.to(device),batch_labels.view(outputs.shape).to(device))
                # record the validation loss of a mini-batch
                valid_loss  += loss_batch.data
                denominator = ii

                y_true.append(batch_labels)
                features.append(feature_)
                labels.append(batch_labels)
        valid_loss = valid_loss / (denominator + 1)
    return valid_loss,y_pred,y_true,features,labels,item

def resample_behavioral_estimate(y_true,y_pred,n_sampling = int(1e3),shuffle = False):
    scores = np.zeros(n_sampling)
    for _idx in range(n_sampling):
        idx_picked = np.random.choice(y_true.shape[0],y_true.shape[0],replace = True)
        if shuffle:
            _y_pred = sk_shuffle(y_pred)
            scores[_idx] = metrics.roc_auc_score(y_true[idx_picked],_y_pred[idx_picked])

        else:
            scores[_idx] = metrics.roc_auc_score(y_true[idx_picked],y_pred[idx_picked])
    return scores

def behavioral_evaluate(net,
                        n_experiment_runs,
                        loss_func,
                        dataloader,
                        device,
                        categorical = True,
                        output_activation = 'softmax',
                        image_type = 'clear',
                        small_dataset = True,
                        ):
    """
    This function evaluates the trained network with given dataloader (could be noisy) for
    a few blocks (like an experiment blocks). The performance of the network is estimated
    by the average of the blocks

    Inputs
    ----------------
    net: nn.Module, the trained network
    n_experiment_runs: int, number of blocks of evaluating the network
    loss_func: torch.nn, loss function
    dataloader: torch.utils.dataset, a dataloader with agumentation procedures
    device: torch.device, where to put the network and the data
    categorical: Boolean, corresponding to the output layer and activation
    output_activation: String, the name of the output activation function, it is used to called the torch function
    image_type: for printing the information
    small_dataset: Boolean, not functional

    Outputs
    -----------------
    y_trues: list of torch.tensors
    y_preds: list of torch.tensors
    scores: list of float
    features: list of torch.autograd.Variables
    labels: list of torch.tensors
    """
    if len(dataloader) > 100: # when the validation data is large
        small_dataset   = False
    # when the validation data is small
    if small_dataset:
        y_preds,y_trues = [],[]
        features,labels = [],[]
        for n_run in range(n_experiment_runs):
            _,y_pred,y_true,_features,_labels       = validation_loop(
                                net,
                                loss_func,
                                dataloader          = dataloader,
                                device              = device,
                                categorical         = categorical,
                                output_activation   = output_activation,
                                )
            y_preds.append(torch.cat(y_pred).detach().cpu())
            y_trues.append(torch.cat(y_true).detach().cpu())
            features.append(_features)
            labels.append(_labels)
        yy_trues = torch.cat(y_trues).detach().cpu().numpy()
        yy_preds = torch.cat(y_preds).detach().cpu().numpy()

        scores = resample_behavioral_estimate(yy_trues,yy_preds)

        if categorical:
            confidence = torch.cat(y_preds).cpu().numpy().max(1)
        else:
            temp = torch.cat(y_preds).cpu().numpy()
            temp[temp < 0.5] = 1- temp[temp < 0.5]
            confidence = temp.copy()

        print(f'\nwith {image_type} images, score = {np.mean(scores):.4f}+/-{np.std(scores):.4f},confidence = {np.mean(confidence):.2f}+/-{np.std(confidence):.2f}')

        return y_trues,y_preds,scores,features,labels
    else:
        _,y_pred,y_true,features,labels = validation_loop(
                                net,
                                loss_func,
                                dataloader = dataloader,
                                device = device,
                                categorical = categorical,
                                output_activation = output_activation,
                                )
        scores = np.zeros(n_experiment_runs)
        for jj in range(n_experiment_runs):
            idx_ = np.random.choice(y_true.shape[0],size = int(y_true.shape[0]),replace = True)
            y_pred_,y_true_ = y_pred[idx_],y_true[idx_]
            score = metrics.roc_auc_score(y_true_,y_pred_)
            scores[jj] = score
        print(f'\nwith {image_type} images, score = {np.mean(scores):.4f}+/-{np.std(scores):.4f}')
        return y_trues,y_preds,scores,features,labels

def behavioral_evaluate_with_path(net,
                        n_experiment_runs,
                        loss_func,
                        dataloader,
                        device,
                        categorical = True,
                        output_activation = 'softmax',
                        image_type = 'clear',
                        small_dataset = True,
                        ):
    """
    This function evaluates the trained network with given dataloader (could be noisy) for
    a few blocks (like an experiment blocks). The performance of the network is estimated
    by the average of the blocks

    Inputs
    ----------------
    net: nn.Module, the trained network
    n_experiment_runs: int, number of blocks of evaluating the network
    loss_func: torch.nn, loss function
    dataloader: torch.utils.dataset, a dataloader with agumentation procedures
    device: torch.device, where to put the network and the data
    categorical: Boolean, corresponding to the output layer and activation
    output_activation: String, the name of the output activation function, it is used to called the torch function
    image_type: for printing the information
    small_dataset: Boolean, not functional

    Outputs
    -----------------
    y_trues: list of torch.tensors
    y_preds: list of torch.tensors
    scores: list of float
    features: list of torch.autograd.Variables
    labels: list of torch.tensors
    """
    from tqdm import tqdm
    if len(dataloader) > 100: # when the validation data is large
        small_dataset   = False
    # when the validation data is small
    if small_dataset:
        y_preds,y_trues = [],[]
        features,labels = [],[]
        items = []
        for n_run in tqdm(range(n_experiment_runs)):
            _,y_pred,y_true,_features,_labels,_items= validation_loop_with_path(
                                net,
                                loss_func,
                                dataloader          = dataloader,
                                device              = device,
                                categorical         = categorical,
                                output_activation   = output_activation,
                                )
            y_preds.append(torch.cat(y_pred).detach().cpu())
            y_trues.append(torch.cat(y_true).detach().cpu())
            features.append(_features)
            labels.append(_labels)
            items.append(_items)
        yy_trues = torch.cat(y_trues).detach().cpu().numpy()
        yy_preds = torch.cat(y_preds).detach().cpu().numpy()

        scores = resample_behavioral_estimate(yy_trues,yy_preds)

        if categorical:
            confidence = torch.cat(y_preds).cpu().numpy().max(1)
        else:
            temp = torch.cat(y_preds).cpu().numpy()
            temp[temp < 0.5] = 1- temp[temp < 0.5]
            confidence = temp.copy()

        print(f'\nwith {image_type} images, score = {np.mean(scores):.4f}+/-{np.std(scores):.4f},confidence = {np.mean(confidence):.2f}+/-{np.std(confidence):.2f}')

        return y_trues,y_preds,scores,features,labels,items
    else:
        _,y_pred,y_true,features,labels = validation_loop(
                                net,
                                loss_func,
                                dataloader = dataloader,
                                device = device,
                                categorical = categorical,
                                output_activation = output_activation,
                                )
        scores = np.zeros(n_experiment_runs)
        for jj in range(n_experiment_runs):
            idx_ = np.random.choice(y_true.shape[0],size = int(y_true.shape[0]),replace = True)
            y_pred_,y_true_ = y_pred[idx_],y_true[idx_]
            score = metrics.roc_auc_score(y_true_,y_pred_)
            scores[jj] = score
        print(f'\nwith {image_type} images, score = {np.mean(scores):.4f}+/-{np.std(scores):.4f}')
        return y_trues,y_preds,scores,features,labels
    
def noise_fuc(x,noise_level = 1):
    """
    add guassian noise to the images during agumentation procedures

    Inputs
    --------------------
    x: torch.tensor, batch_size x 3 x height x width
    noise_level: float, standard deviation of the gaussian distribution
    """
    generator = torch.distributions.normal.Normal(0,noise_level)
    return x + generator.sample(x.shape)

def make_decoder(decoder_name,n_jobs = 1,):
    """
    Make decoders for the hidden representations

    Inputs
    ---------------
    decoder_name: String, to call the dictionary
    n_jobs: int, parallel argument
    """
    np.random.seed(12345)

    # linear SVM
    lsvm = LinearSVC(penalty        = 'l2', # default
                     dual           = True, # default
                     tol            = 1e-3, # not default
                     random_state   = 12345, # not default
                     max_iter       = int(1e3), # default
                     class_weight   = 'balanced', # not default
                     )
    # to make the probabilistic predictions from the SVM
    lsvm = CalibratedClassifierCV(
                     base_estimator = lsvm,
                     method         = 'sigmoid',
                     cv             = 8,
                     )

    # RBF SVM
    svm = SVC(
                     tol            = 1e-3,
                     random_state   = 12345,
                     max_iter       = int(1e3),
                     class_weight   = 'balanced',
                     )
    # to make the probabilistic predictions from the SVM
    svm = CalibratedClassifierCV(
                     base_estimator = svm,
                     method         = 'sigmoid',
                     cv             = 8,
                     )

    # random forest implemented by XGBoost
    xgb = XGBClassifier(
                     learning_rate  = 1e-3, # not default
                     max_depth      = 10, # not default
                     n_estimators   = 100, # not default
                     objective      = 'binary:logistic', # default
                     booster        = 'gbtree', # default
                     subsample      = 0.9, # not default
                     colsample_bytree = 0.9, # not default
                     reg_alpha      = 0, # default
                     reg_lambda     = 1, # default
                     random_state   = 12345, # not default
                     importance_type= 'gain', # default
                     n_jobs         = n_jobs,# default to be 1
                     )

    # logistic regression
    logitstic = LogisticRegression(random_state = 12345)

    if decoder_name == 'linear-SVM':
        decoder = make_pipeline(StandardScaler(),
                                lsvm,
                                )
    elif decoder_name == 'RBF-SVM':
        decoder = make_pipeline(StandardScaler(),
                                svm,
                                )
    elif decoder_name == 'RF':
        decoder = make_pipeline(StandardScaler(),
                                xgb,
                                )
    elif decoder_name == 'logit':
        decoder = make_pipeline(StandardScaler(),
                                logitstic,
                                )
    return decoder

def decode_hidden_layer(decoder,
                        features,
                        labels,
                        cv                  = None,
                        groups              = None,
                        n_splits            = 50,
                        test_size           = .2,
                        categorical         = True,
                        output_activation   = 'softmax',):
    """
    Decode the hidden layer outputs from a convolutional neural network by a scikit-learn classifier

    Inputs
    -----------------------
    decoder: scikit-learn object
    features: numpy narray, n_sample x n_features
    labels: numpy narray, n_samples x 1
    cv: scikit-learn object or None, default being sklearn.model_selection.StratifiedShuffleSplit
    n_splits: int, number of cross validation
    test_size: float, between 0 and 1.
    categorical: Boolean, corresponding to the output activation function of the convolutional neural network
    output_activation: String, name of the output activation fucntion of the convolutional neural network
    """
    if cv == None:
        cv = StratifiedShuffleSplit(n_splits        = n_splits,
                                    test_size       = test_size,
                                    random_state    = 12345,
                                    )
        print(f'CV not defined, use StratifiedShuffleSplit(n_splits = {n_splits})')

    res = cross_validate(decoder,
                         features,
                         labels,
                         groups             = groups,
                         cv                 = cv,
                         scoring            = 'roc_auc',
                         n_jobs             = -1,
                         verbose            = 1,
                         return_estimator   = True,
                         )
    # plase uncomment below and test this when you have enough computational power, i.e. parallel in more than 16 CPUs
#    _,permu,pval = permutation_test_score(decoder,
#                                          features,
#                                          labels,
#                                          groups,
#                                          cv = cv,
#                                          scoring = 'roc_auc',
#                                          n_jobs = -1,
#                                          verbose = 1,
#                                          )
    return res,cv

def resample_ttest(x,
                   baseline         = 0.5,
                   n_permutation    = 10000,
                   one_tail         = False,
                   n_jobs           = 12,
                   verbose          = 0,
                   full_size        = True
                   ):
    """
    http://www.stat.ucla.edu/~rgould/110as02/bshypothesis.pdf
    https://www.tau.ac.il/~saharon/StatisticsSeminar_files/Hypothesis.pdf
    Inputs:
    ----------
    x: numpy array vector, the data that is to be compared
    baseline: the single point that we compare the data with
    n_ps: number of p values we want to estimate
    one_tail: whether to perform one-tailed comparison
    """
    import numpy as np
    import gc
    from joblib import Parallel,delayed
    # statistics with the original data distribution
    t_experiment    = np.mean(x)
    null            = x - np.mean(x) + baseline # shift the mean to the baseline but keep the distribution

    if null.shape[0] > int(1e4): # catch for big data
        full_size   = False
    if not full_size:
        size        = (int(1e3),n_permutation)
    else:
        size = (null.shape[0],n_permutation)
    
    null_dist = np.random.choice(null,size = size,replace = True)
    t_null = np.mean(null_dist,0)
    
    if one_tail:
        return ((np.sum(t_null >= t_experiment)) + 1) / (size[1] + 1)
    else:
        return ((np.sum(np.abs(t_null) >= np.abs(t_experiment))) + 1) / (size[1] + 1) /2

    
def resample_ttest_2sample(a,b,
                           n_permutation        = 10000,
                           one_tail             = False,
                           match_sample_size    = True,
                           n_jobs               = 6,
                           verbose              = 0):
    from joblib import Parallel,delayed
    import gc
    # when the samples are dependent just simply test the pairwise difference against 0
    # which is a one sample comparison problem
    if match_sample_size:
        difference  = a - b
        ps          = resample_ttest(difference,
                                     baseline       = 0,
                                     n_permutation  = n_permutation,
                                     one_tail       = one_tail,
                                     n_jobs         = n_jobs,
                                     verbose        = verbose,)
        return ps
    else: # when the samples are independent
        t_experiment        = np.mean(a) - np.mean(b)
        if not one_tail:
            t_experiment    = np.abs(t_experiment)
        def t_statistics(a,b):
            group           = np.concatenate([a,b])
            np.random.shuffle(group)
            new_a           = group[:a.shape[0]]
            new_b           = group[a.shape[0]:]
            t_null          = np.mean(new_a) - np.mean(new_b)
            if not one_tail:
                t_null      = np.abs(t_null)
            return t_null
        try:
           gc.collect()
           t_null_null = Parallel(n_jobs = n_jobs,verbose = verbose)(delayed(t_statistics)(**{
                            'a':a,
                            'b':b}) for i in range(n_permutation))
        except:
            t_null_null = np.zeros(n_permutation)
            for ii in range(n_permutation):
                t_null_null = t_statistics(a,b)
        if one_tail:
            ps = ((np.sum(t_null_null >= t_experiment)) + 1) / (n_permutation + 1)
        else:
            ps = ((np.sum(np.abs(t_null_null) >= np.abs(t_experiment))) + 1) / (n_permutation + 1) / 2
        return ps

def Find_Optimal_Cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations
    predicted : Matrix with predicted data, where rows are observations
    Returns
    -------
    list type, with optimal cutoff value

    """
    from sklearn.metrics import roc_curve
    fpr, tpr, threshold         = roc_curve(target, predicted)
    i                           = np.arange(len(tpr))
    roc                         = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t                       = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold']) 
