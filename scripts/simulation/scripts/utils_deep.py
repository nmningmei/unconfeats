#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 16:02:43 2020

@author: nmei
"""

import os,gc

import numpy as np

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader,TensorDataset
from torch import nn
from torchvision import models,transforms

from PIL  import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics         import roc_auc_score
from sklearn.utils           import shuffle as sk_shuffle
from sklearn.preprocessing   import MinMaxScaler



def candidate_pretrained_CNNs(model_name = 'alexnet',pretrained = True):
    picked_models = dict(
            resnet18        = models.resnet18(pretrained            = pretrained,
                                              progress              = False,),
            alexnet         = models.alexnet(pretrained             = pretrained,
                                             progress               = False,),
            squeezenet      = models.squeezenet1_1(pretrained       = pretrained,
                                                   progress         = False,),
            vgg19_bn        = models.vgg19_bn(pretrained            = pretrained,
                                              progress              = False,),
            densenet169     = models.densenet169(pretrained         = pretrained,
                                                 progress           = False,),
            inception       = models.inception_v3(pretrained        = pretrained,
                                                  progress          = False,),
            googlenet       = models.googlenet(pretrained           = pretrained,
                                               progress             = False,),
            shufflenet      = models.shufflenet_v2_x0_5(pretrained  = pretrained,
                                                        progress    = False,),
            mobilenet       = models.mobilenet_v2(pretrained        = pretrained,
                                                  progress          = False,),
            resnext50_32x4d = models.resnext50_32x4d(pretrained     = pretrained,
                                                     progress       = False,),
            resnet50        = models.resnet50(pretrained            = pretrained,
                                              progress              = False,),
            )
    return picked_models[model_name]

def define_type(model_name):
    model_type          = dict(
            alexnet     = 'simple',
            vgg19_bn    = 'simple',
            densenet    = 'simple',
            inception   = 'inception',
            mobilenet   = 'simple',
            resnet18    = 'resnet',
            resnet50    = 'resnet',
            )
    return model_type[model_name]

class CustomImageLoader(Dataset):
    def __init__(self,
                 df_data,
                 data, # this is a tuple
                 image_folder,
                 transform = None):
        self.df_data        = df_data
        self.data1          = data[0]
        self.data2          = data[1]
        self.image_folder   = image_folder
        self.transform      = transform
    def __len__(self):
        return self.df_data.shape[0]
    def __getitem__(self,index):
        filename = os.path.join(self.image_folder,
                                self.df_data.loc[index,'targets'],
                                self.df_data.loc[index,'subcategory'],
                                self.df_data.loc[index,'paths'].split('.')[0] + '.jpg')
        image   = Image.open(filename).convert('RGB')
        BOLD1   = torch.from_numpy(self.data1[index])
        BOLD2   = torch.from_numpy(self.data2[index])
        if self.transform is None:
            self.transform = transforms.Compose([
                    transforms.Resize((128,128)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                         [0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
        image = self.transform(image)
        return image,BOLD1,BOLD2,index

def simple_image_augmentation(target_size = 128):
    transform = transforms.Compose([
# transforms.Grayscale(num_output_channels=1),
 transforms.Resize((target_size,target_size)),
 transforms.RandomRotation(45),
 transforms.RandomHorizontalFlip(),
 transforms.RandomVerticalFlip(),
 transforms.ToTensor(),
 transforms.Lambda(lambda x:x/255.),
 # transforms.Normalize(
 #                         mean=[0.485, 0.456, 0.406],
 #                         std=[0.229, 0.224, 0.225])
 ])
    return transform

def loader_wraper(_CustomLoader,
                  batch_size = 8,
                  shuffle = True,
                  num_workers = 1,
                  drop_last = True, 
                  multiprocessing_context = 'fork',):
    """just a pytorch data loader wrapper"""
    Loader = DataLoader(_CustomLoader,
                         batch_size                 = batch_size,
                         shuffle                    = shuffle,
                         num_workers                = num_workers,
                         drop_last                  = drop_last,
                         multiprocessing_context    = multiprocessing_context,
                         )
    return Loader

def linear_block(input_size,output_size,activation_func = nn.ReLU):
    return nn.Sequential(
            nn.Linear(input_size,output_size,bias = True),
            nn.BatchNorm1d(output_size),
            activation_func(),
            )

def conv2d_block(in_channels,
                 out_channels,
                 kernel_size = (3,3),
                 *args,**kwargs):
    return nn.Sequential(
            nn.Conv2d(in_channels   = in_channels,
                      out_channels  = out_channels,
                      kernel_size   = kernel_size,
                      stride        = 1,
                      padding       = 0,
                      padding_mode  = 'zeros',
                      bias          = True,
                      ),
            nn.Conv2d(in_channels   = out_channels,
                      out_channels  = out_channels,
                      kernel_size   = kernel_size,
                      stride        = 1,
                      padding       = 0,
                      padding_mode  = 'zeros',
                      bias          = True,
                      ),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(kernel_size    = 2,
                         stride         = 1,),
            nn.ReLU(),
            )

def TransConv2d_block(in_channels,
                      out_channels,
                      kernel_size = (3,3,),
                      scale_factor = 2,
                      align_corners = 'bilinear',
                      *args,**kwargs):
    return nn.Sequential(
            nn.Upsample(scale_factor = scale_factor,
                        mode = align_corners,
                        align_corners = True,),
            nn.ConvTranspose1d(in_channels = in_channels,
                               out_channels = out_channels,
                               kernel_size = kernel_size,
                               stride = 1,
                               padding = 0,
                               padding_mode = 'zeros',
                               bias = True,
                               ),
            )

def conv1d_block(in_channels,
                 out_channels,
                 kernel_size = 20,
                 *args,**kwargs):
    return nn.Sequential(
                nn.Conv1d(in_channels   = in_channels,
                          out_channels  = out_channels,
                          kernel_size   = kernel_size,
                          stride        = 1,
                          padding       = 0,
                          padding_mode  = 'zeros',
                          bias          = True,),
                nn.Conv1d(in_channels   = out_channels,
                          out_channels  = out_channels,
                          kernel_size   = kernel_size,
                          stride        = 1,
                          padding       = 0,
                          padding_mode  = 'zeros',
                          bias          = True,),
                nn.BatchNorm1d(out_channels),
                nn.MaxPool1d(kernel_size    = int(kernel_size)),
                nn.SELU(),
                )

class image_encoder(nn.Module):
    def __init__(self,
                 batch_size             = 8,
                 device                 = 'cpu',
                 model_name             = 'alexnet',
                 pretrained             = True,
                 latent_size            = 128,
                 ):
        super(image_encoder,self).__init__()
        torch.manual_seed(12345)
        self.batch_size             = batch_size
        self.device                 = device
        self.model_name             = model_name
        self.pretrained             = pretrained
        self.latent_size            = latent_size
        torch.manual_seed(12345)
        self.pretrain_model         = candidate_pretrained_CNNs(model_name = self.model_name,
                                                                pretrained = self.pretrained,)
        if define_type(self.model_name) == 'simple':
            self.pretrained_model = self.pretrain_model.features
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))
            self.in_features = 512#self.pretrain_model.classifier[0].in_features
        elif define_type(self.model_name) == 'resnet':
            self.pretrained_model = torch.nn.Sequential(*list(self.pretrain_model.children())[:-2])
            self.in_features = self.pretrain_model.fc.in_features
        self.latent_mu = linear_block(self.in_features,self.latent_size,nn.SELU)
        self.latent_sigma = linear_block(self.in_features,self.latent_size,nn.SELU)
    def forward(self,x):
        if self.pretrained:
            for params in self.pretrained_model:
                params.requires_grad = False
        features = self.pretrained_model(x)
        pooling = torch.squeeze(torch.squeeze(self.avgpool(features),dim = 3),dim = 2)
        mu = self.latent_mu(pooling)
        var = self.latent_sigma(pooling)
        return mu,var

class image_decoder(nn.Module):
    def __init__(self,
                 batch_size             = 8,
                 device                 = 'cpu',
                 image_size             = 128,
                 latent_size            = 128,
                 out_channels           = [128,64,32,16,8,4,3],
                 kernel_size            = (2,2)
                 ):
        super(image_decoder,self).__init__()
        torch.manual_seed(12345)
        self.batch_size             = batch_size
        self.device                 = device
        self.image_size             = image_size
        self.latent_size            = latent_size
        self.out_channels           = out_channels
        self.kernel_size            = kernel_size
        torch.manual_seed(12345)
        
        self.transpose_conv_layers = []
        for in_channel,out_channel in zip(self.out_channels[:-1],self.out_channels[1:]):
            self.transpose_conv_layers.append(TransConv2d_block(in_channel,out_channel,kernel_size = self.kernel_size))
        self.transpose_conv_layers = nn.Sequential(*self.transpose_conv_layers)
        self.transposeConv2D = nn.ConvTranspose2d(3, 3, self.kernel_size)
        self.output_activation = nn.Sigmoid()
    
    def forward(self,x):
        out = torch.unsqueeze(x,2)
        out = torch.unsqueeze(out,3)
        out = self.transpose_conv_layers(out)
        out = self.output_activation(self.transposeConv2D(out))
        return out

class IMAGE_VAE(nn.Module):
    def __init__(self,
                 batch_size             = 8,
                 device                 = 'cpu',
                 model_name             = 'alexnet',
                 pretrained             = True,
                 latent_size            = 128,
                 image_size             = 128,
                 out_channels           = [128,64,32,16,8,4,3],
                 kernel_size            = (2,2),
                 sampling_method        = 'distribution',
                 ):
        super(IMAGE_VAE,self).__init__()
        torch.manual_seed(12345)
        self.batch_size             = batch_size
        self.device                 = device
        self.model_name             = model_name
        self.pretrained             = pretrained
        self.latent_size            = latent_size
        self.out_channels           = out_channels
        self.kernel_size            = kernel_size
        self.sampling_method        = sampling_method
        
        self.encoder                = image_encoder(
            batch_size             = self.batch_size,
            device                 = self.device,
            model_name             = self.model_name,
            pretrained             = self.pretrained,
            latent_size            = self.latent_size,)
        self.decoder                = image_decoder(
            batch_size             = self.batch_size,
            device                 = self.device,
            out_channels           = self.out_channels,
            kernel_size            = self.kernel_size,
            latent_size            = self.latent_size,)
        self.log_scale              = nn.Parameter(torch.Tensor([0.0]))
        self.latent_activation      = nn.Tanh
    
    def reparameterize(self,):
        mu = self.mu
        log_var = self.log_var
        std = torch.exp(torch.mul(log_var,0.5))
        if self.sampling_method == 'old':
            eps = torch.rand_like(std)
            sample = mu + (eps * std)
        elif self.sampling_method == 'distribution':
            q = torch.distributions.Normal(mu,std)
            sample = q.rsample() # only rsample allows gradient track
            return sample
        elif self.sampling_method == 'sampling_for_decoder':
            q = torch.distributions.Normal(mu,std)
            sample = q.sample()
        if self.latent_activation is not None:
            sample = self.latent_activation(sample)
        return sample
    
    # strange behavior: nonzero loss of identical input and output
    def gaussian_likelihood(self,y_true,y_pred,logscale = None):
        if logscale is None:
            logscale = nn.Parameter(torch.Tensor([0.0]))
        scale = torch.exp(logscale)
        # how huch y_pred could have been distributed using the current state as center
        dist = torch.distributions.Normal(y_pred,scale)
        
        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(y_true)
        
        return log_pxz.mean(dim = (1,2,3))
    
    #https://towardsdatascience.com/variational-autoencoder-demystified-with-pytorch-implementation-3a06bee395ed
    def kl_divergence(self,z,mu,std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first 2 probabilities
        p = torch.distributions.Normal(torch.zeros_like(mu),torch.ones_like(std))
        q = torch.distributions.Normal(mu,std)
        
        # 2. get the probability from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)
        
        # kl
        kl = (log_qzx - log_pz)
        kl = kl.mean(-1)
        return kl
    
    def forward(self,data_tuple):
        inputs,outputs = data_tuple
        # encoding
        self.mu,self.log_var = self.encoder(inputs)
        latent_sample = self.reparameterize()
        # decoding
        outputs_hat = self.decoder(latent_sample)
        # reconstruction loss
        recon_loss = self.gaussian_likelihood(outputs_hat,self.log_scale,outputs) # is this always negative?
        # recon_loss = nn.MSELoss()(outputs_hat,outputs)
        kl_loss = self.kl_divergence(latent_sample,self.mu,torch.exp(self.log_var / 2))
        
        return outputs_hat,latent_sample,kl_loss - recon_loss

class BOLD_encoder(nn.Module):
    def __init__(self,
                 batch_size             = 8,
                 device                 = 'cpu',
                 input_size             = 40000,
                 layers                 = [],
                 latent_size            = 128,
                 ):
        super(BOLD_encoder,self).__init__()
        torch.manual_seed(12345)
        self.batch_size             = batch_size
        self.device                 = device
        self.input_size             = input_size
        self.layers                 = layers
        self.latent_size            = latent_size
        if len(self.layers) > 0:
            self.intermedian_layers = nn.ModuleList()
            self.intermedian_layers.append(linear_block(self.input_size,self.layers[0],nn.SELU))
            for in_fe,out_fe in zip(self.layers[:-1],self.layers[1:]):
                self.intermedian_layers.append(linear_block(in_fe,out_fe, nn.SELU))
            self.intermedian_layers = nn.Sequential(self.intermedian_layers)
            self.latent_mu = linear_block(self.layers[-1],self.latent_size,nn.SELU)
            self.latent_sigma = linear_block(self.layers[-1],self.latent_size,nn.SELU)
        else:
            self.intermedian_layers = None
            self.latent_mu = linear_block(self.input_size,self.latent_size,nn.SELU)
            self.latent_sigma = linear_block(self.input_size,self.latent_size,nn.SELU)
    def forward(self,x):
        if self.intermedian_layers is not None:
            out = self.intermedian_layers(x)
            mu = self.latent_mu(out)
            var = self.latent_sigma(out)
        else:
            mu = self.latent_mu(x)
            var = self.latent_sigma(x)
        return mu,var

class BOLD_decoder(nn.Module):
    def __init__(self,
                 batch_size             = 8,
                 device                 = 'cpu',
                 output_size            = 40000,
                 layers                 = [],
                 latent_size            = 128,
                 ):
        super(BOLD_decoder,self).__init__()
        torch.manual_seed(12345)
        self.batch_size             = batch_size
        self.device                 = device
        self.output_size            = output_size
        self.layers                 = layers
        self.latent_size            = latent_size
        if len(self.layers) > 0:
            self.intermedian_layers = nn.ModuleList()
            self.intermedian_layers.append(linear_block(self.latent_size,self.layers[0],nn.SELU))
            for in_fe,out_fe in zip(self.layers[:-1],self.layers[1:]):
                self.intermedian_layers.append(linear_block(in_fe,out_fe, nn.SELU))
            self.intermedian_layers = nn.Sequential(self.intermedian_layers)
            self.linear_out = linear_block(self.layers[-1],self.output_size,nn.Tanh)
        else:
            self.intermedian_layers = None
            self.linear_out = linear_block(self.latent_size,self.output_size,nn.Tanh)
        
    def forward(self,x):
        if self.intermedian_layers is not None:
            x = self.intermedian_layers(x)
        out = self.linear_out(x)
        
        return out

class BOLD_VAE(nn.Module):
    def __init__(self,
                 batch_size             = 8,
                 device                 = 'cpu',
                 input_size             = 40000, # whole brain
                 output_size            = 3000, # reconstruct ROI
                 encode_layers          = [],
                 decode_layers          = [],
                 latent_size            = 128,
                 sampling_method        = 'distribution',
                 ):
        super(BOLD_VAE,self).__init__()
        torch.manual_seed(12345)
        self.batch_size             = batch_size
        self.device                 = device
        self.input_size             = input_size
        self.output_size            = output_size
        self.encode_layers          = encode_layers
        self.decode_layers          = decode_layers
        self.latent_size            = latent_size
        self.sampling_method        = sampling_method
        
        self.encoder                = BOLD_encoder(
            batch_size             = self.batch_size,
            device                 = self.device,
            input_size             = self.input_size,
            layers                 = self.encode_layers,
            latent_size            = self.latent_size,)
        self.decoder                = BOLD_decoder(
            batch_size             = self.batch_size,
            device                 = self.device,
            output_size            = self.output_size,
            layers                 = self.decode_layers,
            latent_size            = self.latent_size,)
        self.log_scale              = nn.Parameter(torch.Tensor([0.0]))
        self.latent_activation      = nn.Tanh
    
    def reparameterize(self,):
        mu = self.mu
        log_var = self.log_var
        std = torch.exp(torch.mul(log_var,0.5))
        if self.sampling_method == 'old':
            eps = torch.rand_like(std)
            sample = mu + (eps * std)
        elif self.sampling_method == 'distribution':
            q = torch.distributions.Normal(mu,std)
            sample = q.rsample() # only rsample allows gradient track
            return sample
        elif self.sampling_method == 'sampling_for_decoder':
            q = torch.distributions.Normal(mu,std)
            sample = q.sample()
        if self.latent_activation is not None:
            sample = self.latent_activation(sample)
        return sample
    
    def gaussian_likelihood(self,y_true,y_pred,logscale = None):
        if logscale is None:
            logscale = nn.Parameter(torch.Tensor([0.0]))
        scale = torch.exp(logscale)
        # how huch y_pred could have been distributed using the current state as center
        dist = torch.distributions.Normal(y_pred,scale)
        
        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(y_true)
        
        return log_pxz.mean(-1)
    
    def kl_divergence(self,z,mu,std):
        # 1. define the first 2 probabilities
        p = torch.distributions.Normal(torch.zeros_like(mu),torch.ones_like(std))
        q = torch.distributions.Normal(mu,std)
        
        # 2. get the probability from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)
        
        # kl
        kl = (log_qzx - log_pz)
        kl = kl.mean(-1)
        return kl
    
    def forward(self,data_tuple):
        inputs,outputs = data_tuple
        # encoding
        self.mu,self.log_var = self.encoder(inputs)
        latent_sample = self.reparameterize()
        print(latent_sample)
        # decoding
        outputs_hat = self.decoder(latent_sample)
        
        recon_loss = self.gaussian_likelihood(outputs_hat,self.log_scale,outputs) # is this always negative?
        # recon_loss -= nn.MSELoss()(outputs_hat,outputs)
        kl_loss = self.kl_divergence(latent_sample,self.mu,torch.exp(self.log_var / 2))
        
        return outputs_hat,latent_sample,kl_loss - recon_loss

class Conv1D_model(nn.Module):
    def __init__(self,
                 batch_size     = 8,
                 device         = 'cpu',
                 out_channels   = [1,64,128,256,512],
                 kernel_size    = 5, # why bother
                 ):
        super(Conv1D_model,self).__init__()
        torch.manual_seed(12345)
        self.batch_size         = batch_size
        self.device             = device
        self.out_channels       = out_channels
        self.kernel_sizes       = [kernel_size] * len(self.out_channels)
        self.conv_blocks        = []
        for ii,in_channel in enumerate(self.out_channels[:-1]):
             self.conv_blocks.append(conv1d_block(in_channel,
                                                  out_channels[ii + 1],
                                                  kernel_size = self.kernel_sizes[ii + 1]).to(self.device))
        self.conv_blocks        = nn.ModuleList(self.conv_blocks)
        self.adaptive_pooling   = nn.AdaptiveMaxPool1d(1).to(self.device)
        self.linear             = nn.Linear(in_features = self.out_channels[-1],
                                            out_features = 2,
                                            bias = True).to(self.device)
        # self.activation         = nn.Softmax(dim = -1).to(self.device)
    def forward(self,x):
        x                       = x.view(-1,1,x.shape[-1])
        for block_func in self.conv_blocks:
            x                   = block_func(x)
        out                     = self.adaptive_pooling(x)
        out                     = out.view(-1,self.out_channels[-1])
        out                     = self.linear(out)
        # out                     = self.activation(out)
        return out

def conv1dmodel_train_loop(classifier,
                           optimizer,
                           loss_func,
                           data_train,
                           targets_train,
                           idx_train,
                           ii_epoch,
                           activation_func  = nn.Softmax(dim = -1),
                           device           = 'cpu',
                           batch_size       = 8,
                           shuffle          = True,
                           print_train      = True,
                           valid_size       = 0.1,
                           ):
    torch.manual_seed(12345)
    features,labels = data_train[idx_train],targets_train[idx_train]
    features,labels = sk_shuffle(features,labels)
    X_train,X_valid,y_train,y_valid = train_test_split(features,
                                                       labels,
                                                       test_size = valid_size,
                                                       random_state = 12345,
                                                       shuffle = True,
                                                       )
    # initialize
    train_losses = 0.
    valid_losses = 0.
    
    #https://github.com/pytorch/pytorch/issues/44687
    dataset_train = TensorDataset(torch.from_numpy(X_train),
                                  torch.from_numpy(y_train).float(),
                                  )
    dataloader_train = DataLoader(dataset_train,
                                  batch_size                = batch_size,
                                  shuffle                   = shuffle,
                                  num_workers               = 1,
                                  drop_last                 = True,
                                  multiprocessing_context   ='fork',#
                                  )
    dataset_valid = TensorDataset(torch.from_numpy(X_valid),
                                  torch.from_numpy(y_valid).float(),
                                  )
    dataloader_valid = DataLoader(dataset_valid,
                                  batch_size                = batch_size,
                                  shuffle                   = shuffle,
                                  num_workers               = 1,
                                  drop_last                 = True,
                                  multiprocessing_context   ='fork',#
                                  )
    ############################# train #############################
    classifier.train()
    torch.set_grad_enabled(True)
    if print_train:
        t = tqdm(enumerate(dataloader_train))
    else:
        t = enumerate(dataloader_train)
    for ii,(batch_features,batch_labels) in t:
        # reset the gradients in the optimizer
        optimizer.zero_grad()
        
        batch_features  = Variable(batch_features).to(device)
        batch_labels    = Variable(batch_labels).to(device)
        
        batch_predictions = classifier(batch_features)
        
        loss = loss_func(activation_func(batch_predictions),batch_labels,)
        loss.backward()
        
        optimizer.step()
        
        train_losses += loss.data
        if print_train:
            t.set_description(f'epoch {ii_epoch + 1:3d}, train loss = {train_losses/(ii+1):.6f}')
    
    ############################# validation #############################
    classifier.eval()
    with torch.no_grad():
        if print_train:
            t = tqdm(enumerate(dataloader_valid))
        else:
            t = enumerate(dataloader_valid)
        for jj,(batch_features,batch_labels) in t:
            batch_features      = Variable(batch_features).to(device)
            batch_labels        = Variable(batch_labels).to(device)
            batch_predictions   = classifier(batch_features)
            loss                = loss_func(activation_func(batch_predictions),batch_labels,)
            valid_losses        += loss.data
            if print_train:
                t.set_description(f'epoch {ii_epoch + 1:3d}, valid loss = {valid_losses/(jj+1):.6f}')
    return train_losses / (ii + 1),valid_losses / (jj + 1)

def conv1d_model_full_cycle(
                idx_train_source,
                idx_test_target,
                data_source,
                targets_source,
                data_target,
                targets_target,
                current_fold    = 0,
                batch_size      = 8,
                device          = 'cpu',
                out_channels    = [1,2,3,4,5],
                kernel_size     = 5,
                learning_rate   = 1e-4,
                momentum        = 0.,
                weight_decay    = 0.,
                max_epochs      = int(1e3),
                patience        = 5,
                tol             = 1e-4,
                model_name      = 'temp',
                print_train     = True,
                extra_data      = None,
                ):
    # initialize the classifier
    classifier          = Conv1D_model(batch_size   = batch_size,
                                       device       = device,
                                       out_channels = out_channels,
                                       kernel_size  = kernel_size,)
    # print(classifier)
    # define optimizer and loss function
    optimizer           = torch.optim.SGD(classifier.parameters(), 
                                          lr            = learning_rate,
                                          momentum      = momentum,
                                          weight_decay  = weight_decay, # add L2 regularization
                                          )
    loss_func           = torch.nn.BCELoss()
    
    best_valid_loss     = np.inf
    count               = 0
    for ii,ii_epoch in enumerate(range(max_epochs)):
        train_loss,valid_loss = conv1dmodel_train_loop(
                                        classifier,
                                        optimizer,
                                        loss_func,
                                        data_source,
                                        targets_source,
                                        idx_train   = idx_train_source,
                                        ii_epoch    = ii_epoch,
                                        device      = device,
                                        batch_size  = batch_size,
                                        shuffle     = True,
                                        print_train = print_train,
                                        )
        gc.collect()
        # detemining stopping criteria
        temp                        = valid_loss.detach().cpu().numpy()
        if (best_valid_loss > temp) and (np.abs(best_valid_loss - temp) >= tol):
            best_valid_loss         = temp
            count                   = 0
            # save the best model
            if print_train:
                print('getting better, saving the model weights')
            torch.save(classifier.state_dict(),f'{model_name}_{current_fold}.pth')
        else:
            # classifier.load_state_dict(torch.load('temp.pth'))
            count += 1
        
        if count >= patience:
            classifier.load_state_dict(torch.load(f'{model_name}_{current_fold}.pth'))
            try:
                os.remove(f'{model_name}_{current_fold}.pth')
            except:
                pass
            if print_train:
                print("can't get any better, load the best model weights")
            break
    # arvix 2006.08476: not fully understood -- so, don't use it
    if isinstance(extra_data, (list,tuple,np.ndarray)):
        if print_train:
            print('train the model with extra data')
        with torch.no_grad():
            softmax_func = torch.nn.Softmax(dim = -1).to(device)
            psudo_labels = softmax_func(classifier(torch.from_numpy(extra_data.copy()).to(device))).detach().cpu().numpy()
            psudo_labels = np.array(psudo_labels >= 0.5,dtype = int)
        
        data_to_train = np.concatenate([data_source,extra_data])
        targets_to_train = np.concatenate([targets_source,psudo_labels])
        idx_to_train = np.concatenate([idx_train_source,np.arange(extra_data.shape[0]) + data_source.shape[0]])
        
        # initialize the classifier
        # classifier          = Conv1D_model(batch_size   = batch_size,
        #                                     device       = device,
        #                                     out_channels = out_channels,
        #                                     kernel_size  = kernel_size,)
        
        best_valid_loss                 = np.inf
        count                           = 0
        for ii,ii_epoch in enumerate(range(max_epochs)):
            train_loss,valid_loss = conv1dmodel_train_loop(
                                            classifier,
                                            optimizer,
                                            loss_func,
                                            data_to_train,
                                            targets_to_train,
                                            idx_train   = idx_to_train,
                                            ii_epoch    = ii_epoch,
                                            device      = device,
                                            batch_size  = batch_size,
                                            shuffle     = True,
                                            print_train = print_train,
                                            )
            gc.collect()
            # detemining stopping criteria
            temp                        = valid_loss.detach().cpu().numpy()
            if (best_valid_loss > temp) and (np.abs(best_valid_loss - temp) >= tol):
                best_valid_loss         = temp
                count                   = 0
                # save the best model
                if print_train:
                    print('getting better, saving the model weights')
                torch.save(classifier.state_dict(),f'{model_name}_{current_fold}.pth')
            else:
                # classifier.load_state_dict(torch.load('temp.pth'))
                count += 1
            
            if count >= patience:
                classifier.load_state_dict(torch.load(f'{model_name}_{current_fold}.pth'))
                try:
                    os.remove(f'{model_name}_{current_fold}.pth')
                except:
                    pass
                if print_train:
                    print("can't get any better, load the best model weights")
                break
    
    # test
    classifier.eval()
    with torch.no_grad():
        softmax_func = torch.nn.Softmax(dim = -1).to(device)
        y_pred = softmax_func(classifier(torch.from_numpy(data_target[idx_test_target]).to(device))).detach().cpu().numpy()
    y_true = targets_target[idx_test_target]
    
    score = roc_auc_score(y_true,y_pred)
    return score

class _Conv2D_model(nn.Module):
    def __init__(self,
                 batch_size     = 8,
                 device         = 'cpu',
                 in_channel     = 88,
                 out_channels   = [100,64,128,256,512],
                 kernel_size    = (5,5),
                 ):
        super(_Conv2D_model,self).__init__()
        torch.manual_seed(12345)
        self.batch_size         = batch_size
        self.device             = device
        self.in_channel         = in_channel
        self.out_channels       = out_channels
        self.kernel_size        = kernel_size
        self.conv_blocks        = []
        in_channel              = self.in_channel
        for ii,out_channel in enumerate(self.out_channels):
             self.conv_blocks.append(conv2d_block(in_channel,
                                                  out_channel,
                                                  kernel_size = self.kernel_size).to(self.device))
             in_channel = out_channel
        self.conv_blocks        = nn.ModuleList(self.conv_blocks)
        self.adaptive_pooling   = nn.AdaptiveMaxPool2d((1,1)).to(self.device)
    def forward(self,x):
        for block_func in self.conv_blocks:
            x                   = block_func(x)
        out                     = self.adaptive_pooling(x)
        out                     = out.view(-1,self.out_channels[-1])
        return out

class Conv2D_model(nn.Module):
    def __init__(self,
                 batch_size     = 8,
                 device         = 'cpu',
                 in_channels    = (66,88,88),
                 out_channels   = [100,64,128,256,],
                 kernel_sizes   = [(8,8),(8,6),(8,6)],
                 ):
        super(Conv2D_model,self).__init__()
        torch.manual_seed(12345)
        self.batch_size         = batch_size
        self.device             = device
        self.in_channels        = in_channels
        self.out_channels       = out_channels
        self.kernel_sizes       = kernel_sizes
        self.view1              = _Conv2D_model(in_channel = self.in_channels[0],
                                                kernel_size = self.kernel_sizes[0],
                                                out_channels =  self.out_channels,
                                                device = self.device,
                                                batch_size = self.batch_size,
                                                )
        self.view2              = _Conv2D_model(in_channel = self.in_channels[1],
                                                kernel_size = self.kernel_sizes[1],
                                                out_channels =  self.out_channels,
                                                device = self.device,
                                                batch_size = self.batch_size,
                                                )
        self.view3              = _Conv2D_model(in_channel = self.in_channels[2],
                                                kernel_size = self.kernel_sizes[2],
                                                out_channels =  self.out_channels,
                                                device = self.device,
                                                batch_size = self.batch_size,
                                                )
        self.activation         = nn.Sigmoid().to(self.device)
    def forward(self,x):
        x1 = x
        x2 = x.permute(0,2,3,1)
        x3 = x.permute(0,3,2,1)
        
        out1 = self.activation(self.view1(x1))
        out2 = self.activation(self.view2(x2))
        out3 = self.activation(self.view3(x3))
        return out1,out2,out3


'''
class feature_extractor(nn.Module):
    def __init__(self,
                 batch_size = 8,
                 device = 'cpu',
                 input_channels = 88,# or 66
                 out_channels = [64,64,64,128,128,256,256,512],
                 kernel_size = [(3,3),(3,3),(3,3),(3,3),(3,3),(3,3),(3,3),(3,3)],
                 ):
        super(feature_extractor,self).__init__()
        torch.manual_seed(12345)
        self.batch_size = batch_size
        self.device = device
        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.conv_block1 = conv_block(input_channels,
                                      out_channels[0],
                                      kernel_size = self.kernel_size[0],).to(self.device)
        self.conv_blocks = []
        for ii,in_channel in enumerate(out_channels[:-1]):
             self.conv_blocks.append(conv_block(in_channel,
                                                out_channels[ii + 1],
                                                kernel_size = self.kernel_size[ii + 1]).to(self.device))
        self.conv_blocks = nn.ModuleList(self.conv_blocks)
    
    def forward(self,x):
        out = self.conv_block1(x)
        for block_func in self.conv_blocks:
            out = block_func(out)
        
        return out

class simple_classifier(nn.Module):
    def __init__(self,
                 batch_size = 8,
                 device = 'cpu',
                 input_shape = 512,):
        super(simple_classifier,self).__init__()
        torch.manual_seed(12345)
        self.batch_size = batch_size
        self.device = device
        self.linear = nn.Linear(in_features = input_shape,out_features = batch_size,).to(self.device)
        self.activation = nn.Softmax(dim = -1).to(device)
    
    def forward(self,x):
        x = self.linear(x)
        x = self.activation(x)
        return x

class CNN2D_model(nn.Module):
    def __init__(self,
                 batch_size = 2,
                 device = 'cpu',
                 kernel_sizes = [(3,3),(3,2),(3,3)],
                 input_channles = [88,88,66],
                 out_channels = [64,64,64,128,128,256,256,512],
                 ):
        super(CNN2D_model,self).__init__()
        torch.manual_seed(12345)
        self.batch_size = batch_size
        self.device = device
        self.kernel_sizes = kernel_sizes
        self.input_channels = input_channles
        self.AdaptivePool = nn.AdaptiveMaxPool2d((1,1)).to(self.device)
        self.out_channels = out_channels
        self.feature_extractors_0 = feature_extractor(batch_size = self.batch_size,
                                                      device = self.device,
                                                      input_channels = self.input_channels[0],
                                                      kernel_size = self.kernel_sizes[0],
                                                      out_channels = self.out_channels,).to(self.device)
        self.feature_extractors_1 = feature_extractor(batch_size = self.batch_size,
                                                      device = self.device,
                                                      input_channels = self.input_channels[1],
                                                      kernel_size = self.kernel_sizes[1],
                                                      out_channels = self.out_channels,).to(self.device)
        self.feature_extractors_2 = feature_extractor(batch_size = self.batch_size,
                                                      device = self.device,
                                                      input_channels = self.input_channels[2],
                                                      kernel_size = self.kernel_sizes[2],
                                                      out_channels = self.out_channels,).to(self.device)
        self.activation = nn.ReLU()
        self.classifier = simple_classifier(batch_size = self.batch_size,
                                            device = self.device,
                                            input_shape = self.out_channels[-1])
    def forward(self,x):
        # x0 is x
        # print(x.shape)
        # x1 is x.permute
        x1 = x.permute(0,2,3,1)
        # print(x1.shape)
        # x2 is x.permute
        x2 = x.permute(0,3,2,1)
        # print(x2.shape)
        
        out0 = self.feature_extractors_0(x)
        out1 = self.feature_extractors_1(x1)
        out2 = self.feature_extractors_2(x2)
#        print(out0.shape,out1.shape,out2.shape)
        
        out0 = self.activation(self.AdaptivePool(out0).view(-1,1,out0.size()[1]))
        out1 = self.activation(self.AdaptivePool(out1).view(-1,1,out1.size()[1]))
        out2 = self.activation(self.AdaptivePool(out2).view(-1,1,out2.size()[1]))
        
        outs = torch.cat([out0,out1,out2],0).view(-1,self.out_channels[-1])
        
        outs = self.classifier(outs)
        
        return outs,(out0,out1,out2)

class Encoder(nn.Module):
    def __init__(self,
                 batch_size = 8,
                 device = 'cpu',
                 num_classes = 2,
                 feature_extractor = None,
                 dim_cat = 1,
                 hidden_feature_size = 1,):
        super(Encoder,self,).__init__()
        torch.manual_seed(12345)
        self.batch_size = batch_size
        self.device = device
        self.dim_cat = dim_cat
        self.hidden_feature_size = hidden_feature_size
        if feature_extractor is not None:
            self.feature_extractor = feature_extractor
        else:
            self.feature_extractor = CNN2D_model(batch_size = self.batch_size)
        self.rnn = nn.GRU(input_size = 512,
                          hidden_size = self.hidden_feature_size,
                          num_layers = 1,
                          batch_first = True,
                          bidirectional = True,)
        self.linear = nn.Linear(10,num_classes)
        self.out_activation = nn.Softmax(dim = -1)
    def forward(self,x):
        feature_extractor = self.feature_extractor
        outs = feature_extractor(x)
        features = torch.cat(outs,dim = self.dim_cat)
        out,hidden = self.rnn(features)
        return out,hidden

class rnn_classifier(nn.Module):
    def __init__(self,
                 batch_size = 8,
                 device = 'cpu',
                 num_classes = 2,
                 feature_extractor = None):
        super(rnn_classifier,self,).__init__()
        torch.manual_seed(12345)
        self.batch_size = batch_size
        self.device = device
        if feature_extractor is not None:
            self.feature_extractor = feature_extractor
        else:
            self.feature_extractor = CNN2D_model(batch_size = self.batch_size)
        self.rnn = nn.GRU(input_size = 512,
                          hidden_size = 5,
                          num_layers = 1,
                          batch_first = True,
                          bidirectional = True,)
        self.linear = nn.Linear(10,num_classes)
        self.out_activation = nn.Softmax(dim = -1)
    def forward(self,x):
        feature_extractor = self.feature_extractor
        outs = feature_extractor(x)
        features = torch.cat(outs,dim = 1)
        out,hidden = self.rnn(features)
        out = self.linear(out)
        out = torch.mean(out,dim = 1)
        out = self.out_activation(out)
        return out,hidden



class simple_encoder(nn.Module):
    def __init__(self,
                 batch_size = 8,
                 device = 'cpu',
                 input_size = 5000,
                 latent_size = 2,
                 intermediate_size = 1280,
                 ):
        super(simple_encoder,self).__init__()
        self.batch_size = batch_size
        self.device = device
        self.input_size = input_size
        self.latent_size = latent_size
        self.intermediate_size = intermediate_size
        torch.manual_seed(12345)
        self.layer1 = linear_block(self.input_size,self.intermediate_size)
        self.latent_layer = linear_block(self.intermediate_size,self.latent_size,nn.SELU)
    def forward(self,x):
#        x = x.view(-1,self.input_size)
        out = self.layer1(x)
        out = self.latent_layer(out)
        return out

class image_vae_encoder(nn.Module):
    def __init__(self,
                 batch_size = 8,
                 device = 'cpu',
                 input_size = (1,3,128,128),
                 latent_size = 2,
                 intermediate_size = 1280,
                 model_name = 'alexnet',
                 pretrained = True,
                 ):
        super(image_vae_encoder,self).__init__()
        self.batch_size = batch_size
        self.device = device
        self.input_size = input_size
        self.latent_size = latent_size
        self.intermediate_size = intermediate_size
        self.model_name = model_name
        self.pretrained = pretrained
        torch.manual_seed(12345)
        self.pretrain_model = candidate_pretrained_CNNs(model_name = self.model_name,
                                                        pretrained = self.pretrained,)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.scaling = nn.SELU()
        if define_type(self.model_name) == 'simple':
            self.pretrained_model = self.pretrain_model.features
            self.in_features = self.avgpool(self.pretrained_model(torch.rand(*self.input_size))).shape[1]
        elif define_type(self.model_name) == 'resnet':
            self.pretrained_model = torch.nn.Sequential(*list(self.pretrain_model.children())[:-2])
            self.in_features = self.pretrain_model.fc.in_features
        self.latent_layer = linear_block(self.in_features,self.latent_size,nn.SELU)
    def forward(self,x):
        if self.pretrained:
            for params in self.pretrained_model:
                params.requires_grad = False
        features = self.pretrained_model(x)
        pooling = self.scaling(torch.squeeze(self.avgpool(features)))
        mu = self.latent_layer(pooling)
        var = self.latent_layer(pooling)
        return mu,var

class vae_encoder(nn.Module):
    def __init__(self,
                 batch_size = 8,
                 device = 'cpu',
                 input_size = 5000,
                 latent_size = 2,
                 intermediate_size = 1280,
                 ):
        super(vae_encoder,self).__init__()
        self.batch_size = batch_size
        self.device = device
        self.input_size = input_size
        self.latent_size = latent_size
        self.intermediate_size = intermediate_size
        torch.manual_seed(12345)
        self.layer1 = linear_block(self.input_size,self.intermediate_size)
        self.latent_layer = linear_block(self.intermediate_size,self.latent_size,nn.SELU)
    def forward(self,x):
#        x = x.view(-1,self.input_size)
        out = self.layer1(x)
        mu = self.latent_layer(out)
        var = self.latent_layer(out)
        return mu,var

def stack_perceptron_layers(latent_size = 2,
                            intermediate_sizes = (1280,256),
                            ):
    temp = [linear_block(latent_size,intermediate_sizes[0])]
    for ii,intermediate_size in enumerate(intermediate_sizes):
        if ii != 0:
            temp.append(linear_block(intermediate_sizes[ii - 1],
                                     intermediate_size))
    return nn.Sequential(*temp)

class simple_decoder(nn.Module):
    def __init__(self,
                 batch_size = 8,
                 device = 'cpu',
                 output_size = 5000,
                 latent_size = 2,
                 intermediate_size = 1280
                 ):
        super(simple_decoder,self).__init__()
        self.batch_size = batch_size
        self.device = device
        self.output_size = output_size
        self.latent_size = latent_size
        self.intermediate_size = intermediate_size
        torch.manual_seed(12345)
        if type(intermediate_size) == int:
            self.latent_layer = linear_block(self.latent_size,self.intermediate_size)
            self.layer_out = linear_block(self.intermediate_size,self.output_size,nn.Sigmoid)
        else:
            self.stack_layers = stack_perceptron_layers(self.latent_size,self.intermediate_size)
            self.layer_out = linear_block(self.intermediate_size[-1],self.output_size,nn.Sigmoid)
    def forward(self,x):
        if type(self.intermediate_size) == int:
            out = self.latent_layer(x)
        else:
            out = self.stack_layers(x)
        out = self.layer_out(out)
        return out

class AE(nn.Module):
    def __init__(self,
                 batch_size = 8,
                 device = 'cpu',
                 input_size = 5000,
                 output_size = 5000,
                 latent_size = 2,
                 intermediate_size = 1280,
                 ):
        super(AE,self).__init__()
        self.batch_size = batch_size
        self.device = device
        self.input_size = input_size
        self.output_size = output_size
        self.latent_size = latent_size
        self.intermediate_size = intermediate_size
        torch.manual_seed(12345)
        self.Encoder = simple_encoder(batch_size = self.batch_size,
                                      device = self.device,
                                      input_size = self.input_size,
                                      latent_size = self.latent_size,
                                      intermediate_size = self.intermediate_size,
                                      )
        self.Decoder = simple_decoder(batch_size = self.batch_size,
                                      device = self.device,
                                      output_size = self.output_size,
                                      latent_size = self.latent_size,
                                      intermediate_size = self.intermediate_size
                                      )
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))
        
    def gaussian_likelihood(self,y_hat,logscale,y):
        scale = torch.exp(logscale)
        mean = y_hat
        dist = torch.distributions.Normal(mean,scale)
        
        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(y)
        return torch.abs(log_pxz.mean(-1))
    
    def forward(self,data_tuple):
        x,y = data_tuple
        latent_space = self.Encoder(x)
        y_hat = self.Decoder(latent_space)
        recon_loss = self.gaussian_likelihood(y_hat,self.log_scale,y)
        recon_loss += nn.MSELoss()(y_hat,y)
        return y_hat,recon_loss

class LinearSVM(nn.Module):
    def __init__(self,
                 class_size = 2,
                 batch_size = 8,
                 device = 'cpu',
                 input_size = 5000,
                 ):
        super(LinearSVM,self).__init__()
        self.batch_size = batch_size
        self.device = device
        self.class_size = class_size
        self.input_size = input_size
        torch.manual_seed(12345)
        self.classify_layer = linear_block(self.input_size,self.class_size,nn.Softmax)
    def forward(self,data_tuple):
        x,y = data_tuple
        out = self.classify_layer(x)
        loss = nn.BCELoss()(out.float(),y.float())
        return out,loss


class VAE(nn.Module):
    def __init__(self,
                 batch_size = 8,
                 device = 'cpu',
                 input_size = 5000,
                 output_size = 5000,
                 latent_size = 2,
                 intermediate_size = 1280,
                 sample_method = 'old',
                 model_name = 'alexnet',
                 pretrained = True,
                 ):
        super(VAE,self).__init__()
        self.batch_size = batch_size
        self.device = device
        self.input_size = input_size
        self.output_size = output_size
        self.latent_size = latent_size
        self.sample_method = sample_method
        self.intermediate_size = intermediate_size
        self.model_name = model_name
        self.pretrained = pretrained
        torch.manual_seed(12345)
        if type(self.input_size) != int:
            self.Encoder = image_vae_encoder(batch_size = self.batch_size,
                                             device = self.device,
                                             input_size = input_size,
                                             latent_size = self.latent_size,
                                             intermediate_size = self.intermediate_size,
                                             model_name = self.model_name,
                                             pretrained = self.pretrained,
                                             )
        else:
            self.Encoder = vae_encoder(batch_size = self.batch_size,
                                          device = self.device,
                                          input_size = self.input_size,
                                          latent_size = self.latent_size,
                                          intermediate_size = self.intermediate_size,
                                          )
        self.Decoder = simple_decoder(batch_size = self.batch_size,
                                      device = self.device,
                                      output_size = self.output_size,
                                      latent_size = self.latent_size,
                                      intermediate_size = self.intermediate_size
                                      )
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))
    
    def reparameterize(self,mu,log_var,activation_func = None):
        if self.sample_method == 'old':
            std = torch.exp(0.5 * log_var)
            eps = torch.rand_like(std)
            sample = mu + (eps * std)
            return sample
        elif self.sample_method == 'distribution':
            std = torch.exp(log_var / 2)
            q = torch.distributions.Normal(mu,std)
            sample = q.rsample() # only rsample allows gradient track
            return sample
        elif self.sample_method == 'sampling_for_decoder':
            std = torch.exp(log_var / 2)
            q = torch.distributions.Normal(mu,std)
            sample = q.sample()
            if activation_func is not None:
                sample = activation_func(sample)
            return sample
    
    def gaussian_likelihood(self,y_hat,logscale,y):
        scale = torch.exp(logscale)
        mean = y_hat
        dist = torch.distributions.Normal(mean,scale)
        
        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(y)
        
#        _sum_of_errors = torch.sum(torch.pow(y - y_hat,2)).item()
#        _total_erros = torch.sum(torch.pow(y - torch.mean(y),2)).item()
#        _y_sum = torch.sum(y).item()
#        _y_sq_sum = torch.sum(torch.pow(y,2)).item()
##        r2_loss = 1 - _sum_of_errors / (_y_sq_sum - (_y_sum ** 2) / y.shape[0])
#        r2_loss = 1 - _sum_of_errors / _total_erros
#        if r2_loss >= 0 :
#            r2_loss = -(1 - r2_loss)
        
        return log_pxz.mean(-1)
    
    def kl_divergence(self,z,mu,std):
        # 1. define the first 2 probabilities
        p = torch.distributions.Normal(torch.zeros_like(mu),torch.ones_like(std))
        q = torch.distributions.Normal(mu,std)
        
        # 2. get the probability from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)
        
        # kl
        kl = (log_qzx - log_pz)
        kl = kl.mean(-1)
        return kl
    
    def forward(self,data_tuple):
        x,y = data_tuple
        # encoding
        self.mu,self.log_var = self.Encoder(x)
        sample = self.reparameterize(self.mu,self.log_var,nn.SELU)
        # decoding
        y_hat = self.Decoder(sample)
        
        recon_loss = self.gaussian_likelihood(y_hat,self.log_scale,y)
        recon_loss -= nn.MSELoss()(y_hat,y)
        kl_loss = self.kl_divergence(sample,self.mu,torch.exp(self.log_var / 2))
        
        return y_hat,kl_loss - recon_loss

class combined_model(nn.Module):
    def __init__(self,
                 batch_size = 8,
                 device = 'cpu',
                 class_size = 2,
                 input_size = 5000,
                 output_size = 5000,
                 latent_size = 2,
                 intermediate_size = 1280,
                 sample_method = 'old',
                 hidden_classification = False,
                 ):
        super(combined_model,self).__init__()
        self.batch_size = batch_size
        self.device = device
        self.input_size = input_size
        self.output_size = output_size
        self.latent_size = latent_size
        self.sample_method = sample_method
        self.intermediate_size = intermediate_size
        self.hidden_classification = hidden_classification
        self.class_size = class_size
        torch.manual_seed(12345)
        
        if self.sample_method is None:
            self.autoencoder = AE(batch_size = self.batch_size,
                                  device = self.device,
                                  input_size = self.input_size,
                                  output_size = self.output_size,
                                  latent_size = self.latent_size,
                                  intermediate_size = self.intermediate_size,
                                  )
        else:
            self.autoencoder = VAE(batch_size = self.batch_size,
                                   device = self.device,
                                   input_size = self.input_size,
                                   output_size = self.output_size,
                                   latent_size = self.latent_size,
                                   intermediate_size = self.intermediate_size,
                                   sample_method = self.sample_method,
                                   )
        if self.hidden_classification:
            self.classifier = LinearSVM(class_size = self.class_size,
                                        batch_size = self.batch_size,
                                        device = self.device,
                                        input_size = self.latent_size,
                                        )
        else:
            self.classifier = LinearSVM(class_size = self.class_size,
                                        batch_size = self.batch_size,
                                        device = self.device,
                                        input_size = self.input_size,
                                        )
    def forward(self,data_tuple):
        x,y,label = data_tuple
        if self.sample_method is not None: # variational
            # encoding
            self.mu,self.log_var = self.autoencoder.Encoder(x)
            sample = self.autoencoder.reparameterize(self.mu,self.log_var)
            # decoding
            y_hat = self.autoencoder.Decoder(sample)
            
            recon_loss = self.autoencoder.gaussian_likelihood(y_hat,self.log_scale,y)
            kl_loss = self.autoencoder.kl_divergence(sample,self.mu,torch.exp(self.log_var / 2))
            autoencoder_loss = kl_loss - recon_loss
        else: # simple autoencoder
            sample = self.autoencoder.Encoder(x)
            # decoding
            y_hat = self.autoencoder.Decoder(sample)
#            recon_loss = self.gaussian_likelihood(y_hat,self.log_scale,y)
            recon_loss = nn.MSELoss()(y_hat,y)
            autoencoder_loss = recon_loss
        
        if self.hidden_classification:
            preds,classifier_loss = self.classifier((sample,label))
        else:
            preds,classifier_loss = self.classifier((y_hat,label))
        
        return y_hat,autoencoder_loss + classifier_loss

def train_loop(model,dataloader,optimizer,classification = False,device = 'cpu',ii_epoch = None,print_train = False):
    model.train()
    torch.set_grad_enabled(True)
    train_loss = 0.
    for ii,data_tuple in enumerate(dataloader):
        # reset the gradients in the optimizer
        optimizer.zero_grad()
        
        if classification:
            inputs,outputs,labels = data_tuple
            labels = Variable(labels).to(device)
        else:
            inputs,outputs = data_tuple
#        inputs = inputs.view(-1,inputs.shape[-1])
        inputs = Variable(inputs).to(device)
        outputs = Variable(outputs).to(device)
        
        if classification:
            out_hat,loss = model((inputs,outputs,labels))
        else:
            out_hat,loss = model((inputs,outputs))
        loss = loss.mean()
        loss.backward()
        
        optimizer.step()
        train_loss  += loss.data
        if print_train:
            print(f'{ii + 1:3.0f}/{100*(ii+1)/ len(dataloader):2.3f}%,loss = {train_loss/(ii+1):.6f}')
    else:
        print(f'epoch {ii_epoch + 1}, train loss = {train_loss/(ii+1):.6f}')
    return train_loss / (ii + 1)

def validation_loop(model,dataloader,classification = False,device = 'cpu',ii_epoch = None):
    model.eval()
    with torch.no_grad():
        validation_loss = 0.
        for ii,data_tuple in enumerate(dataloader):
            
            if classification:
                inputs,outputs,labels = data_tuple
                labels = Variable(labels).to(device)
            else:
                inputs,outputs = data_tuple
#            inputs = inputs.view(-1,inputs.shape[-1])
            inputs = Variable(inputs).to(device)
            outputs = Variable(outputs).to(device)
            if classification:
                out_hat,loss = model((inputs,outputs,labels))
            else:
                out_hat,loss = model((inputs,outputs))
            loss = loss.mean()
            validation_loss  += loss.data
        print(f'epoch {ii_epoch + 1}, valid loss = {validation_loss / (ii + 1):.6f}')
    return validation_loss / (ii + 1)

def train_validation_block(max_epochs,
                           model,
                           dataloader,
                           optimizer,
                           device = 'cpu',
                           classification = True,
                           model_saving_name = 'best.path',
                           ):
    # data placeholder for the train-validation losses
    res                             = torch.zeros((2,max_epochs))
    best_valid_loss                 = np.inf
    count                           = 0
    for epoch in range(max_epochs):
        # train cycle
        train_loss                  =       train_loop(model,
                                                       dataloader,
                                                       classification   = classification,
                                                       optimizer        = optimizer,
                                                       device           = device,
                                                       ii_epoch         = epoch)
        # validation cycle
        valid_loss                  =       validation_loop(model,
                                                            dataloader,
                                                            classification  = classification,
                                                            device          = device,
                                                            ii_epoch        = epoch)
        res[0,epoch]                = train_loss
        res[1,epoch]                = valid_loss
        print()
        # detemining stopping criteria
        temp                        = valid_loss.detach().cpu().numpy()
        if best_valid_loss > temp:
            best_valid_loss         = temp
            count                   = 0
            # save the best model
            torch.save(model.state_dict(),model_saving_name)
        else:
            count += 1
        
        if count >= 50:
            model.load_state_dict(torch.load(model_saving_name))
            break
'''
