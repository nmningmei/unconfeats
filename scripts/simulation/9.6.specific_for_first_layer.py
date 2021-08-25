#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 11:10:17 2021

@author: nmei
"""

import os,gc,torch

import numpy as np
import pandas as pd

from utils_deep import (data_loader,
                        candidates,
                        define_augmentations,
                        createLossAndOptimizer,
                        train_and_validation,
                        resample_ttest_2sample,
                        noise_fuc,
                        make_decoder,
                        decode_hidden_layer,
                        resample_ttest,
                        resample_behavioral_estimate,
                        simple_augmentations,
                        behavioral_evaluate
                        )

from joblib import Parallel,delayed
from sklearn.decomposition import PCA
for _model_name in ['vgg19_bn','resnet50','alexnet','densenet169','mobilenet']:
# experiment control
    model_dir               = '../models'
    results_dir             = '../results/first_layer_only'
    train_folder            = 'greyscaled'
    valid_folder            = 'experiment_images_greyscaled'
    train_root              = f'../data/{train_folder}/'
    valid_root              = f'../data/{valid_folder}'
    image_resize            = 128
    batch_size              = 8
    lr                      = 1e-4
    n_epochs                = int(1e3)
    device                  = 'cpu'
    pretrain_model_name     = _model_name
    output_activation       = 'softmax'
    output_units            = 2
    categorical             = True
    n_experiment_runs       = 20
    n_noise_levels          = 50
    n_permutations          = int(1e4)
    noise_levels            = np.concatenate([[0],[item for item in np.logspace(-1,3,n_noise_levels)]])
    
    if not os.path.exists(results_dir,):
        os.mkdir(results_dir)
    
    class FCNN_model(torch.nn.Module):
        def __init__(self,
                     model_name,
                     in_shape = (1,3,128,128),
                     ):
            super(FCNN_model,self).__init__()
            torch.manual_seed(12345)
            self.pretrained_model = candidates(model_name,pretrained=True)
            self.in_shape = in_shape
            if model_name == 'resnet50':
                self.first_layer_func = self.pretrained_model.conv1
            else:
                self.first_layer_func = self.pretrained_model.features[0]
            
            # freeze the weights
            for params in self.pretrained_model.parameters():
                params.requires_grad = False
            
            # add a last layer
            self.linear_layer = torch.nn.Linear(1000,2,)
        def forward(self,x):
            out = self.pretrained_model(x)
            pred = self.linear_layer(out)
            features = self.first_layer_func(x).view(x.shape[0],-1)
            return pred,features
    
    print('set up random seeds')
    torch.manual_seed(12345)
    if torch.cuda.is_available():torch.cuda.empty_cache();torch.cuda.manual_seed(12345);
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device:{device}')
    
    model_to_train = FCNN_model(pretrain_model_name,)
    model_to_train.to(device)
    model_parameters                            = filter(lambda p: p.requires_grad, model_to_train.parameters())
    params                                      = sum([np.prod(p.size()) for p in model_parameters])
    print(pretrain_model_name,
    #      model_to_train(next(iter(train_loader))[0]),
          f'total params = {params}')
    
    f_name = os.path.join(model_dir,f'{pretrain_model_name}_first_layer.pth')
    
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
            print_train     = True,
            patience        = 5,
            n_noise         = 1,
            train_root      = train_root,
            valid_root      = valid_root,)
    
    del model_to_train
    model_to_train = torch.load(f_name)
    model_to_train.to('cpu')
    
    np.random.seed(12345)
    torch.manual_seed(12345)
    to_round = 9
    
    csv_saving_name     = os.path.join(results_dir,pretrain_model_name,'performance_results.csv')
    if not os.path.exists(os.path.join(results_dir,pretrain_model_name)):
        os.mkdir(os.path.join(results_dir,pretrain_model_name))
    
    results         = dict(model_name           = [],
                           noise_level          = [],
                           cnn_score            = [],
                           cnn_pval             = [],
                           first_score_mean     = [],
                           first_score_std      = [],
                           svm_first_pval       = [],
                           )
    
    model_to_train.eval()
    for param in model_to_train.parameters():
        param.requires_grad = False
    loss_func,optimizer = createLossAndOptimizer(model_to_train,learning_rate = lr)
    
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
                            model_to_train,
                            n_experiment_runs,
                            loss_func,
                            valid_loader,
                            device,
                            categorical = categorical,
                            output_activation = output_activation,
                            )
        behavioral_scores = resample_behavioral_estimate(y_trues,y_preds,int(1e3),shuffle = False)
        # print(var,np.mean(behavioral_scores))
        # del features,labels
        gc.collect()
        chance_level = Parallel(n_jobs = -1,verbose = 1)(delayed(resample_behavioral_estimate)(**{
            'y_true':y_trues,
            'y_pred':y_preds,
            'n_sampling':int(1e1),
            'shuffle':True,}) for _ in range(n_permutations))
        gc.collect()
        cnn_pval = (np.sum(np.array(chance_level).mean(1) >= np.mean(behavioral_scores)) + 1) / (n_permutations + 1)
        
        decoder = make_decoder('linear-SVM',n_jobs = 1)
        decode_features = torch.cat([torch.cat(item).detach().cpu() for item in features]).numpy()
        decode_labels   = torch.cat([torch.cat(item).detach().cpu() for item in labels  ]).numpy()
        del features,labels
        if len(decode_labels.shape) > 1:
            decode_labels = decode_labels[:,-1]
        print('fitting PCA...')
        pca_features = PCA(n_components = .9,random_state = 12345).fit_transform(decode_features)
        gc.collect()
        res,_,svm_first_pval = decode_hidden_layer(decoder,pca_features,decode_labels,
                                  n_splits = 50,
                                  test_size = 0.2,)
        gc.collect()
        svm_first_scores = res['test_score']
        print(var,np.mean(behavioral_scores),np.mean(svm_first_scores))
        results['model_name'].append(pretrain_model_name)
        results['noise_level'].append(var)
        results['cnn_score'].append(np.mean(behavioral_scores))
        results['cnn_pval'].append(cnn_pval)
        results['first_score_mean'].append(np.mean(svm_first_scores))
        results['first_score_std'].append(np.std(svm_first_scores))
        results['svm_first_pval'].append(svm_first_pval)
        gc.collect()
        results_to_save = pd.DataFrame(results)
        results_to_save.to_csv(os.path.join(results_dir,pretrain_model_name,'decodings.csv'),index = False)








