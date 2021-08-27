
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
from sklearn.decomposition import PCA
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
                        'sigmoid':torch.sigmoid,
                        'hinge':torch.nn.HingeEmbeddingLoss}
probability_func_dict = {'softmax':F.softmax,    # softmax dim = 1
                         'sigmoid':torch.sigmoid}
softmax_dim = 1

#candidate models
def candidates(model_name,pretrained = True,):
    picked_models = dict(
            resnet18        = Tmodels.resnet18(pretrained           = pretrained,
                                              progress              = False,),
            alexnet         = Tmodels.alexnet(pretrained            = pretrained,
                                             progress               = False,),
            # squeezenet      = Tmodels.squeezenet1_1(pretrained      = pretrained,
            #                                        progress         = False,),
            vgg19_bn        = Tmodels.vgg19_bn(pretrained           = pretrained,
                                              progress              = False,),
            densenet169     = Tmodels.densenet169(pretrained        = pretrained,
                                                 progress           = False,),
            inception       = Tmodels.inception_v3(pretrained       = pretrained,
                                                  progress          = False,),
            # googlenet       = Tmodels.googlenet(pretrained          = pretrained,
            #                                    progress             = False,),
            # shufflenet      = Tmodels.shufflenet_v2_x0_5(pretrained = pretrained,
            #                                             progress    = False,),
            mobilenet       = Tmodels.mobilenet_v2(pretrained       = pretrained,
                                                  progress          = False,),
            # resnext50_32x4d = Tmodels.resnext50_32x4d(pretrained    = pretrained,
            #                                          progress       = False,),
            resnet50        = Tmodels.resnet50(pretrained           = pretrained,
                                              progress              = False,),
            )
    return picked_models[model_name]

def define_type(model_name):
    model_type          = dict(
            alexnet     = 'simple',
            vgg19_bn    = 'simple',
            densenet169 = 'simple',
            inception   = 'inception',
            mobilenet   = 'simple',
            resnet18    = 'resnet',
            resnet50    = 'resnet',
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

def _cut_bins(x,n_noise_levels = 50):
    _,bins          = pd.cut(np.arange(n_noise_levels),2,retbins = True)
    if bins[0] <= x < bins[1]:
        return 'low'
    # elif bins[1] <= x < bins[2]:
    #     return 'medium'
    else:
        return 'high'

def output_activation_functions(activation_func_name):
    funcs = dict(softmax = F.log_softmax,
                 sigmoid = F.logsigmoid,
                 )
    return funcs[activation_func_name]

def define_augmentations(image_resize = 128,noise_level = None):
    augmentations = {
        'train':simple_augmentations(image_resize,noise_level),
        'valid':simple_augmentations(image_resize,noise_level),
    }
    return augmentations

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

def simple_augmentations(image_resize = 128,noise_level = None):
    if noise_level is not None:
        return transforms.Compose([
    transforms.Resize((image_resize,image_resize)),
    transforms.RandomHorizontalFlip(p = 0.5),
    transforms.RandomRotation(45,),
    transforms.RandomVerticalFlip(p = 0.5,),
    transforms.ToTensor(),
    transforms.Lambda(lambda x:noise_fuc(x,noise_level)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    else:
        return transforms.Compose([
    transforms.Resize((image_resize,image_resize)),
    transforms.RandomHorizontalFlip(p = 0.5),
    transforms.RandomRotation(45,),
    transforms.RandomVerticalFlip(p = 0.5,),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
class customizedDataset(ImageFolder):
    def __getitem__(self, idx):
        original_tuple  = super(customizedDataset,self).__getitem__(idx)
        path = self.imgs[idx][0]
        tuple_with_path = (original_tuple +  (path,))
        return tuple_with_path

def freeze_layer_weights(layer):
    for param in layer.parameters():
        param.requries_grad = False

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
        torch.manual_seed(12345)
        in_features             = nn.AdaptiveAvgPool2d((1,1))(pretrain_model.features(torch.rand(*in_shape))).shape[1]
        avgpool                 = nn.AdaptiveAvgPool2d((1,1))
        hidden_layer            = nn.Linear(in_features,hidden_units)
        output_layer            = nn.Linear(hidden_units,output_units)
        if hidden_dropout > 0:
            dropout             = nn.Dropout(p = hidden_dropout)
        
        print(f'feature dim = {in_features}')
        self.features           = nn.Sequential(pretrain_model.features,
                                                avgpool,)
        if (hidden_activation is not None) and (hidden_dropout > 0):
            self.hidden_layer   = nn.Sequential(hidden_layer,
                                                hidden_activation,
                                                dropout,)
        elif (hidden_activation is not None) and (hidden_dropout == 0):
            self.hidden_layer   = nn.Sequential(hidden_layer,
                                                hidden_activation,)
        elif (hidden_activation == None) and (hidden_dropout > 0):
            self.hidden_layer   = nn.Sequential(hidden_layer,
                                                dropout,)
        elif (hidden_activation == None) and (hidden_dropout == 0):
            self.hidden_layer   = hidden_layer
        
        self.output_layer       = output_layer

    def forward(self,x,):
        out     = torch.squeeze(torch.squeeze(self.features(x),3),2)
        hidden  = self.hidden_layer(out)
        outputs = self.output_layer(hidden)
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
        torch.manual_seed(12345)
        avgpool         = nn.AdaptiveAvgPool2d((1,1))
        in_features     = pretrain_model.fc.in_features
        hidden_layer    = nn.Linear(in_features,hidden_units)
        dropout         = nn.Dropout(p = hidden_dropout)
        output_layer    = nn.Linear(hidden_units,output_units)
        res_net         = torch.nn.Sequential(*list(pretrain_model.children())[:-2])
        print(f'feature dim = {in_features}')
        
        self.features           = nn.Sequential(res_net,
                                      avgpool)
        if (hidden_activation is not None) and (hidden_dropout > 0):
            self.hidden_layer   = nn.Sequential(hidden_layer,
                                                hidden_activation,
                                                dropout,)
        elif (hidden_activation is not None) and (hidden_dropout == 0):
            self.hidden_layer   = nn.Sequential(hidden_layer,
                                                hidden_activation,)
        elif (hidden_activation == None) and (hidden_dropout > 0):
            self.hidden_layer   = nn.Sequential(hidden_layer,
                                                dropout,)
        elif (hidden_activation == None) and (hidden_dropout == 0):
            self.hidden_layer   = hidden_layer
        self.output_layer       = output_layer
        
    def forward(self,x):
        out     = torch.squeeze(torch.squeeze(self.features(x),3),2)
        hidden  = self.hidden_layer(out)
        outputs = self.output_layer(hidden)
        return outputs,hidden

class _inception_model(nn.Module):
    """
    MARK: private function
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
        super(_inception_model,self).__init__()


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
        model_to_train = _inception_model(
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

def build_model_first_layer(CNN_backbone_model_name,
                            trained_model):
    for params in trained_model.parameters():
        params.requires_grad = False
    first_layer = trained_model[0]
    return first_layer

class modified_model(nn.Module):
    def __init__(self,
                 model_to_train,
                 hidden_units       = 2,
                 layer_type         = 'linear', # or RNN
                 layer_units        = 2,
                 layer_activation   = 'selu',
                 layer_dropout      = 0.
                 ):
        super(modified_model,self).__init__()
        layer_activation = hidden_activation_functions(layer_activation)
        self.model_to_train = model_to_train
        if layer_type == 'linear':
            RL_layer            = nn.Linear(hidden_units,layer_units)
            dropout             = nn.Dropout(p = layer_dropout)
            output_layer        = nn.Linear(layer_units,2)
            
            if (layer_activation is not None) and (layer_dropout > 0):
                self.RL_layer   = nn.Sequential(RL_layer,
                                                layer_activation,
                                                dropout,)
            elif (layer_activation is not None) and (layer_dropout == 0):
                self.RL_layer   = nn.Sequential(RL_layer,
                                                layer_activation,)
            elif (layer_activation == None) and (layer_dropout == 0):
                self.RL_layer   = RL_layer
            self.output_layer   = nn.Sequential(output_layer,
                                                nn.Softmax(dim = 1),
                                                )
    def forward(self,x):
        catagory,hidden_representation = self.model_to_train(x)
        RL_output = self.RL_layer(hidden_representation)
        out = self.output_layer(RL_output)
        return catagory,hidden_representation,out
    
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
               n_noise              = 0,
               use_hingeloss        = False,
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
        iterator = tqdm(enumerate(dataloader))
    else:
        iterator = enumerate(dataloader)

    for ii,(features,labels) in iterator:
        if "Binary Cross Entropy" in loss_func.__doc__:
            labels = labels.float() # I indeed hate this
        
        if n_noise > 0:
            # in order to have desired classification behavior, which is to predict
            # chance when no signal is present, we manually add some noise samples
            noise_generator = torch.distributions.normal.Normal(features.mean(),
                                                                features.std())
            noisy_features  = noise_generator.sample(features.shape)[:n_noise]
            noisy_labels    = torch.tensor([0.5] * labels.shape[0])[:n_noise]
            
            
            features        = torch.cat([features,noisy_features])
            labels          = torch.cat([labels,noisy_labels])

        # shuffle the training batch
        np.random.seed(12345)
        idx_shuffle         = np.random.choice(features.shape[0],features.shape[0],replace = False)
        features            = features[idx_shuffle]
        labels              = labels[idx_shuffle]

        if ii + 1 <= len(dataloader): # drop last
            # load the data to memory
            inputs      = Variable(features).to(device)
            # one of the most important steps, reset the gradients
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
            """
            # add L2 loss to the weights
            if l2_lambda > 0:
                weight_norm = torch.norm(list(net.parameters())[-4],2)
                loss_batch  += l2_lambda * weight_norm
            # add L1 loss to the weights
            if l1_lambda > 0:
                weight_norm = torch.norm(list(net.parameters())[-4],1)
                loss_batch  += l1_lambda * weight_norm
            """
            # backpropagation
            loss_batch.backward()
            # modify the weights
            optimizer.step()
            # record the training loss of a mini-batch
            train_loss  += loss_batch.data
            if print_train:
                iterator.set_description(f'epoch {idx_epoch+1}-{ii + 1:3.0f}/{100*(ii+1)/len(dataloader):2.3f}%,loss = {train_loss/(ii+1):.6f}')
                
    return train_loss/(ii+1)

def validation_loop(net,
                    loss_func,
                    dataloader,
                    device,
                    categorical = True,
                    output_activation = 'softmax',
                    verbose = 0,
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
        if verbose == 0:
            iterator = enumerate(dataloader)
        else:
            iterator        = tqdm(enumerate(dataloader))
        for ii,(batch_features,batch_labels) in iterator:
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
                              pretrain_model_name,
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
    # from tqdm import tqdm
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

def train_and_validation(
        model_to_train,
        f_name,
        output_activation,
        loss_func,
        optimizer,
        image_resize = 128,
        device = 'cpu',
        batch_size = 8,
        n_epochs = int(3e3),
        print_train = True,
        patience = 5,
        train_root = '',
        valid_root = '',
        n_noise = 0,
        noise_level = None,):
    """
    This function is to train a new CNN model on clear images
    
    The training and validation processes should be modified accordingly if 
    new modules (i.e., a secondary network) are added to the model
    
    Arguments
    ---------------
    model_to_train:torch.nn.Module, a nn.Module class
    f_name:string, the name of the model that is to be trained
    output_activation:torch.nn.activation, the activation function that is used
        to apply non-linearity to the output layer
    loss_func:torch.nn.modules.loss, loss function
    optimizer:torch.optim, optimizer
    image_resize:int, default = 128, the number of pixels per axis for the image
        to be resized to
    device:string or torch.device, default = "cpu", where to train model
    batch_size:int, default = 8, batch size
    n_epochs:int, default = int(3e3), the maximum number of epochs for training
    print_train:bool, default = True, whether to show verbose information
    patience:int, default = 5, the number of epochs the model is continuely trained
        when the validation loss does not change
    train_root:string, default = '', the directory of data for training
    valid_root:string, default = '', the directory of data for validation
    
    Output
    -----------------
    model_to_train:torch.nn.Module, a nn.Module class
    """
    if output_activation   == 'softmax':
        categorical         = True
    elif output_activation == 'sigmoid':
        categorical         = False
    augmentations = {
            'train':simple_augmentations(image_resize,noise_level = noise_level),
            'valid':simple_augmentations(image_resize,noise_level = noise_level),
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
    
    
    model_to_train.to(device)
    model_parameters    = filter(lambda p: p.requires_grad, model_to_train.parameters())
    if print_train:
        params          = sum([np.prod(p.size()) for p in model_parameters])
        print(f'total params: {params:d}')
    
    best_valid_loss     = torch.tensor(float('inf'),dtype = torch.float64)
    losses = []
    for idx_epoch in range(n_epochs):
        # train
        print('\ntraining ...')
        _               = train_loop(
        net                 = model_to_train,
        loss_func           = loss_func,
        optimizer           = optimizer,
        dataloader          = train_loader,
        device              = device,
        categorical         = categorical,
        idx_epoch           = idx_epoch,
        print_train         = print_train,
        output_activation   = output_activation,
        n_noise             = n_noise,
        )
        print('\nvalidating ...')
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
    return model_to_train

def resample_behavioral_estimate(y_true,y_pred,n_sampling = int(1e3),shuffle = False):
    from joblib import Parallel,delayed
    def _temp_func(idx_picked,shuffle = shuffle):
        if shuffle:
            _y_pred = sk_shuffle(y_pred)
            score = metrics.roc_auc_score(y_true[idx_picked],_y_pred[idx_picked])

        else:
            score = metrics.roc_auc_score(y_true[idx_picked],y_pred[idx_picked])
        return score
    scores = Parallel(n_jobs = -1,verbose = 0)(delayed(_temp_func)(**{
        'idx_picked':np.random.choice(y_true.shape[0],y_true.shape[0],replace = True),
        'shuffle':shuffle}) for _ in range(n_sampling))
    
    return scores

def behavioral_evaluate(net,
                        n_experiment_runs,
                        loss_func,
                        dataloader,
                        device,
                        categorical = True,
                        output_activation = 'softmax',
                        verbose = 0,
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
    

    Outputs
    -----------------
    y_trues: list of torch.tensors
    y_preds: list of torch.tensors
    scores: list of float
    features: list of torch.autograd.Variables
    labels: list of torch.tensors
    """
    
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
                            verbose             = verbose,
                            )
        y_preds.append(torch.cat(y_pred).detach().cpu())
        y_trues.append(torch.cat(y_true).detach().cpu())
        features.append(_features)
        labels.append(_labels)
    yy_trues = torch.cat(y_trues).detach().cpu().numpy()
    yy_preds = torch.cat(y_preds).detach().cpu().numpy()
    
    return yy_trues,yy_preds,features,labels

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
    


def make_decoder(decoder_name,n_jobs = 1,):
    """
    Make decoders for the hidden representations

    Inputs
    ---------------
    decoder_name: String, to call the dictionary
    n_jobs: int, parallel argument
    """
    np.random.seed(12345)
    from sklearn.decomposition import PCA
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
    elif decoder_name == 'PCA-linear-SVM':
        decoder = make_pipeline(StandardScaler(),
                                PCA(random_state = 12345,),
                                lsvm,)
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
                        test_size           = .2,):
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
    pval = resample_ttest(res['test_score'],baseline = 0.5,n_permutation = int(1e4),
                                  one_tail = True,n_jobs = -1.)
    return res,cv,pval

def resample_ttest(x,
                   baseline         = 0.5,
                   n_permutation    = 10000,
                   one_tail         = False,
                   n_jobs           = 12,
                   verbose          = 0,
                   full_size        = True,
                   metric_func      = np.mean,
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
    # import gc
    # from joblib import Parallel,delayed
    # statistics with the original data distribution
    t_experiment    = metric_func(x)
    null            = x - metric_func(x) + baseline # shift the mean to the baseline but keep the distribution

    if null.shape[0] > int(1e4): # catch for big data
        full_size   = False
    if not full_size:
        size        = (int(1e3),n_permutation)
    else:
        size = (null.shape[0],n_permutation)
    
    null_dist = np.random.choice(null,size = size,replace = True)
    t_null = metric_func(null_dist,0)
    
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

def decode_and_visualize_hidden_representations(fig,axes,
                                                y_true,y_pred,features,
                                                hidden_units = 2,
                                                return_estimator = False):
    # visualize the hidden representations
    import gc
    import seaborn as sns
    if len(y_true.shape) == 2:
        y_true = y_true[:,-1]
        y_pred = y_pred[:,-1]
    if hidden_units == 2: # when we don't need to PCA for visualization
        thr = Find_Optimal_Cutoff(y_true,y_pred)
        print(metrics.classification_report(y_true,y_pred>=thr[0]))
        ax = axes.flatten()[0]
        sns.scatterplot(x = features[:,0], y = features[:,1],hue = y_true,ax = ax,)
        ax.set(xlabel = 'feature 1',
               ylabel = 'feature 2',
               title = f'hidden representations,CNN = {metrics.roc_auc_score(y_true,y_pred):.4f}',)
        
        gc.collect()
        svm = make_decoder('linear-SVM')
        X,y = features.copy(),y_true.copy()
        cv = StratifiedShuffleSplit(n_splits = 300, test_size = 0.2, random_state = 12345)
        res = cross_validate(svm,X,y, cv = cv,scoring = 'roc_auc',n_jobs = -1, verbose = 0)
        gc.collect()
        ax = axes.flatten()[-1]
        ax.hist(res['test_score'])
        ax.set(title = 'decoding scores from decoding the hidden layer')
    else:
        features_pca = PCA(n_components = 2,random_state = 12345).fit_transform(features)
        thr = Find_Optimal_Cutoff(y_true,y_pred)
        print(metrics.classification_report(y_true,y_pred>=thr[0]))
        ax = axes.flatten()[0]
        sns.scatterplot(x = features_pca[:,0], y = features_pca[:,1],hue = y_true,ax = ax,)
        ax.set(xlabel = 'PC 1',
               ylabel = 'PC 2',
               title = f'PCA of hidden representations,CNN = {metrics.roc_auc_score(y_true,y_pred):.4f}',)
        
        gc.collect()
        svm = make_decoder('linear-SVM')
        X,y = features.copy(),y_true.copy()
        cv = StratifiedShuffleSplit(n_splits = 300, test_size = 0.2, random_state = 12345)
        res = cross_validate(svm,X,y, cv = cv,scoring = 'roc_auc',n_jobs = -1, verbose = 0)
        gc.collect()
        ax = axes.flatten()[-1]
        ax.hist(res['test_score'])
        ax.set(title = 'decoding scores from decoding the hidden layer')
    if return_estimator:
        svm.fit(X,y)
        return fig,svm
    else:
        return fig