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
                        resample_ttest,
                        resample_behavioral_estimate
                        )
from matplotlib import pyplot as plt
#plt.switch_backend('agg')

print('set up random seeds')
torch.manual_seed(12345)


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
hidden_units            = 100
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
start_decoding          = False
to_round                = 9

results_dir             = '../results/'
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


csv_saving_name     = os.path.join(results_dir,model_saving_name,'performance_results.csv')

if True:#not os.path.exists(csv_saving_name):
    results         = dict(model_name           = [],
                           hidden_units         = [],
                           hidden_activation    = [],
                           output_activation    = [],
                           noise_level          = [],
                           score_mean           = [],
                           score_std            = [],
                           chance_mean          = [],
                           chance_std           = [],
                           model                = [],
                           pval                 = [],
                           confidence_mean      = [],
                           confidence_std       = [],
                           dropout              = [],
                           )
else:
    df_temp         = pd.read_csv(csv_saving_name)
    results         = {col_name:list(df_temp[col_name]) for col_name in df_temp.columns}

res_temp            = []

print(noise_levels)

for var in noise_levels:
    var = round(var,to_round)
    if True:#var not in np.array(results['noise_level']).round(to_round):
        print(f'\nworking on {var:1.1e}')
        noise_folder  = os.path.join(results_dir,model_saving_name,f'{var:1.1e}')
        if not os.path.exists(noise_folder):
            os.mkdir(noise_folder)

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
                # here I turn on the shuffle like it is in a real experiment
                )
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
        yy_trues        = torch.cat(y_trues).detach().cpu().numpy()
        yy_preds        = torch.cat(y_preds).detach().cpu().numpy()
        chance_scores   = resample_behavioral_estimate(yy_trues,yy_preds,shuffle = True)

        pval            = resample_ttest_2sample(scores,chance_scores,
                                                 match_sample_size = False,
                                                 one_tail = False,
                                                 n_ps = 1,
                                                 n_permutation = int(1e5),
                                                 n_jobs = -1,
                                                 verbose = 1,
                                                 )

        # save the features and labels from the hidden layer
        decode_features = torch.cat([torch.cat(run) for run in features])
        decode_labels   = torch.cat([torch.cat(run) for run in labels])

        decode_features = decode_features.detach().cpu().numpy()
        decode_labels   = decode_labels.detach().cpu().numpy()

        if categorical:
            decode_labels = decode_labels[:,-1]
        np.save(os.path.join(noise_folder,'features.npy'),decode_features)
        np.save(os.path.join(noise_folder,'labels.npy'  ),decode_labels)
        gc.collect()
        results['model_name'        ].append(pretrain_model_name)
        results['hidden_units'      ].append(hidden_units)
        results['hidden_activation' ].append(hidden_func_name)
        results['output_activation' ].append(output_activation)
        results['noise_level'       ].append(round(var,to_round))
        results['score_mean'        ].append(np.mean(scores))
        results['score_std'         ].append(np.std(scores))
        results['chance_mean'       ].append(np.mean(chance_scores))
        results['chance_std'        ].append(np.std(chance_scores))
        results['model'             ].append('CNN')
        results['pval'              ].append(np.mean(pval))
        results['dropout'           ].append(hidden_dropout)
        y_preds = torch.cat(y_preds).detach().cpu().numpy()
        if len(y_preds.shape) > 1:
            confidence = y_preds.max(1)
        else:
            confidence = y_preds.copy()
        results['confidence_mean'   ].append(np.mean(confidence))
        results['confidence_std'    ].append(np.std(confidence))

        # save example noisy images
        print('plotting example images')

        batches,batch_labels    = next(iter(visualize_loader))

        PIL_transformer         = transforms.ToPILImage()
        plt.close('all')
        fig,axes                = plt.subplots(figsize = (16,16),nrows = 4,ncols = 4)
        for ax,batch_,batch_label in zip(axes.flatten(),batches,batch_labels):
            batch_              = np.array(PIL_transformer(batch_))
            ax.imshow(batch_,)
            ax.axis('off')
            ax.set(title        = {0:'living',
                                   1:'nonliving'}[int(batch_label.numpy())])
        fig.suptitle(f'noise level = {var:1.1e}, performance = {np.mean(scores):.3f} +/- {np.std(scores):.3f}\nconfidence = {np.mean(confidence):.3f} +/- {np.std(confidence):.3f}')
        fig.savefig(os.path.join(noise_folder,'examples.jpeg'),bbox_inches = 'tight')
        plt.close('all')
        gc.collect()

        if np.mean(scores) < 0.55:
            start_decoding  = True
        if start_decoding:
            decode_scores   = []
            for decoder_name in ['linear-SVM','RBF-SVM','RF']:#,'logit']:
                decoder     = make_decoder(decoder_name,n_jobs = 1)
                res,cv      = decode_hidden_layer(decoder,
                                                  decode_features,
                                                  decode_labels,
                                                  n_splits          = 100,
                                                  test_size         = 0.2,
                                                  categorical       = categorical,
                                                  output_activation = output_activation,)
                decode_scores.append(res['test_score'].mean())
                y_preds = []
                for (_,idx_test),est in zip(cv.split(decode_features,decode_labels),res['estimator']):
                    y_pred_ = est.predict_proba(decode_features[idx_test])
                    y_preds.append(y_pred_)
                y_preds = np.concatenate(y_preds)
                if len(y_preds.shape) > 1:
                    confidence = y_preds.max(1)
                else:
                    confidence = y_preds.copy()

                pval = resample_ttest(res['test_score'],
                                      0.5,
                                      one_tail      = True,
                                      n_jobs        = -1,
                                      n_ps          = 50,
                                      n_permutation = int(1e5),
                                      verbose       = 1,
                                      )
                gc.collect()
                results['model_name'        ].append(pretrain_model_name)
                results['hidden_units'      ].append(hidden_units)
                results['hidden_activation' ].append(hidden_func_name)
                results['output_activation' ].append(output_activation)
                results['noise_level'       ].append(round(var,to_round))
                results['score_mean'        ].append(np.mean(res['test_score']))
                results['score_std'         ].append(np.std(res['test_score']))
                results['chance_mean'       ].append(.5)
                results['chance_std'        ].append(0.)
                results['model'             ].append(decoder_name)
                results['pval'              ].append(np.mean(pval))
                results['confidence_mean'   ].append(np.mean(confidence))
                results['confidence_std'    ].append(np.std(confidence))
                results['dropout'           ].append(hidden_dropout)
                print(f"\nwith {var:1.1e} noise images, {decoder_name} = {np.mean(res['test_score']):.4f}+/-{np.std(res['test_score']):.4f},pval = {np.mean(pval):1.1e}")

        results_to_save = pd.DataFrame(results)
        results_to_save.to_csv(csv_saving_name,index = False)

        res_temp.append(np.mean(scores))

        # CNN being too good above 0.5 for only a little bit
        if len(res_temp) > n_keep_going:
            criterion1  = (np.abs(np.median(res_temp[-n_keep_going:]) - .5) < 1e-3)
        else:
            criterion1  = False
        # CNN is below 0.5 for a long time
        criterion2      = (np.median(res_temp[-n_keep_going:]) < 0.5)
        # decoders can no longer decoder
        try:
            criterion3  = np.logical_and(len(decode_scores) > 1,all(np.array(decode_scores) < 0.5))
        except:
            criterion3  = False

        if criterion1 and criterion2 and criterion3:
            break
    else:
        idx_ ,          = np.where(np.array(results['noise_level']).round(5) == var)
        score_mean      = results['score_mean'][idx_[0]]
        score_std       = results['score_std'][idx_[0]]
        confidence_mean = results['confidence_mean'][idx_[0]]
        confidence_std  = results['confidence_std'][idx_[0]]
        print(f'\nwith {var:1.1e} noise images, score = {score_mean:.4f}+/-{score_std:.4f},confidence = {confidence_mean:.2f}+/-{confidence_std:.2f}')
        res_temp.append(score_mean)
        if ((np.abs(np.mean(res_temp[-n_keep_going:]) - .5) < 1e-3) and (len(res_temp) > n_keep_going)) or (np.mean(res_temp[-n_keep_going:]) < 0.5):
            break
print('done')
