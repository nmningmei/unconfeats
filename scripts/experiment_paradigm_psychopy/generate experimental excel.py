    #!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 12:15:37 2019

@author: nmei
"""

import os
from glob import glob
import pandas as pd
import numpy as np

figure_dir = '../experiment_stimuli/bw_gau_bl' #cor_bc_bl #bw_bc_bl #bw_gau_bl
scramble_dir = 'experiment_stimuli/experiment_images_tilted_scramble_greyscale_2' #experiment_images_tilted_scramble_colored #experiment_images_tilted_scramble_greyscale
csv_saving_dir = 'csvs'
if not os.path.exists(csv_saving_dir):
    os.mkdir(csv_saving_dir)
categories = {'Living_Things':1,'Nonliving_Things':2}
n_repeats = 50
n_repeats += 1 # the last one is for calibration
subtitle = 'EEG_gau'#EEG_bw # fMRI #EEG_cl




results = dict(
#                premask_path = [],
                probe_path = [],
#                postmask_path = [],
#                fixation_duration = [],
#                premask_onset = [],
#                premask_frames = [],
#                probe_onset = [],
#                probe_frames = [],
#                postmask_onset = [],
#                postmask_frames = [],
                category = [],
                subcategory = [],
                label = [],
#                respond_onset = [],
#                respond_frames = [],
#                visibility_onset = [],
#                visibility_frames = [],
#                correctAns = [],
                )
frameRate = 100

n_masks = 5
mask_duration = 100.
mask_frames = int(mask_duration/1000 * frameRate)
probe_frames = 2
respond_frames = 120

if subtitle != 'fMRI':
    results['fixation_duration'] = []

if subtitle == 'fMRI':
    sample_bound = 7.,9.
else:
    sample_bound = 0.5,1.5

#for n in range(n_masks):
#    results['premask_path_{}'.format(n + 1)] = []
#    results['premask_duration_{}'.format(n + 1)] = []
#    results['postmask_path_{}'.format(n + 1)] = []
#    results['postmask_duration_{}'.format(n + 1)] = []


for category,corrAn in categories.items():
    for subcategory in glob(os.path.join(figure_dir,category,'*')):
        images = glob(subcategory+'/*')
        np.random.seed(12345)
        
        chunks = np.unique([image.split('/')[-1].split('.')[0].split('_')[0] for image in images])
        chunks = [[image for image in images if (image.split('/')[-1].split('.')[0].split('_')[0] == subsub)] for subsub in chunks]
        sampled = []
        for chunk in chunks:
            if "Thumbs" not in chunk[0]:
                ii = np.random.choice(chunk,size=n_repeats,replace=False)
                [sampled.append(image) for image in ii]
        for image in sampled:
            temp_dir = image.split('/')
            temp_dir[2] = scramble_dir.split('/')[-1]
            
            mask = os.path.join(*temp_dir)
            masks = [mask[:-4] + '_{}'.format(n + 1) + '.jpg' for n in range(n_masks)]
            
            if subtitle != 'fMRI':
                fixation = int(np.random.uniform(sample_bound[0],sample_bound[1],size=1)[0] / (1./frameRate))
                results['fixation_duration'].append(fixation)
            
#            [results['premask_path_{}'.format(n + 1)].append(mask_) for n,mask_ in enumerate(masks)]
#            results['premask_onset'].append(fixation)
#            [results['premask_duration_{}'.format(n + 1)].append(mask_frames) for n in range(n_masks)]
            
            results['probe_path'].append(image)
#            results['probe_onset'].append(fixation + mask_frames*n_masks)
#            results['probe_frames'].append(probe_frames)
            
#            [results['postmask_path_{}'.format(n + 1)].append(mask_) for n,mask_ in enumerate(masks)]
#            results['postmask_onset'].append(fixation + mask_frames + probe_frames)
#            [results['postmask_duration_{}'.format(n + 1)].append(mask_frames) for n in range(n_masks)]
            
            results['category'].append(category)
            results['subcategory'].append(subcategory.split('/')[-1])
            results['label'].append(image.split('/')[-1].split('.')[0].split('_')[0])
            
#            results['respond_onset'].append(fixation + mask_frames*n_masks + probe_frames + mask_frames*n_masks)
#            results['respond_frames'].append(respond_frames)
            
#            results['visibility_onset'].append(fixation + mask_frames*n_masks + probe_frames + mask_frames*n_masks + respond_frames)
#            results['visibility_frames'].append(respond_frames)
            
#            results['correctAns'].append(corrAn)

results = pd.DataFrame(results)
for _ in range(100):
    results = results.sample(frac=1)
#writer = pd.ExcelWriter('experiment ({} sessions).xlsx'.format(n_repeats))
#results.to_excel(writer,'sheet1')
#writer.save()
results.to_csv(os.path.join(csv_saving_dir,'experiment ({} sessions, {}).csv'.format(n_repeats,subtitle)),index=False)

experiment_table = {}
for n in range(n_repeats):
    experiment_table[n] = {name:[] for name in results.columns}

for kk,(label,df_label) in enumerate(results.groupby(['label'])):
    for n in range(n_repeats):
        [experiment_table[n][name].append(df_label[name].values[n]) for name in df_label.columns]


if subtitle == 'fMRI':
    
    for n in range(n_repeats):
        experiment_table[n] = pd.DataFrame(experiment_table[n])
        temp = experiment_table[n].sort_values(['category'])
        temp_living,temp_nonliving = temp[temp['category'] == 'Living_Things'], temp[temp['category'] == 'Nonliving_Things']
        temp_living,temp_nonliving = temp_living.reset_index(),temp_nonliving.reset_index()
        ii = 1
        ind_groups = {'group1':[],'group2':[],'group3':[]}
        for (idx,living),(_,nonliving) in zip(temp_living.iterrows(),temp_nonliving.iterrows()):
            
            ind_groups[f'group{ii}'].append(temp_living.iloc[idx])
            ind_groups[f'group{ii}'].append(temp_nonliving.iloc[idx])
            if len(ind_groups[f'group{ii}']) >= 32:
                ii += 1
        for ii,rows in enumerate(ind_groups.values()):
            temp_to_save = pd.DataFrame(rows)
            for _ in range(100):
                temp_to_save = temp_to_save.sample(frac=1)
            temp_to_save.to_csv(os.path.join(csv_saving_dir,
                        'experiment (sessions {},block {} {}).csv'.format(n+1,ii + 1,subtitle)),index=False)
        if n == n_repeats -1:
            name_temp = [os.path.join(csv_saving_dir,'experiment (sessions {},block {} {}).csv'.format(n,ii + 1,subtitle)) for ii in range(3)]
            df_temp = pd.concat([pd.read_csv(f) for f in name_temp])
            df_temp.to_csv(os.path.join(csv_saving_dir,'calibration fMRI.csv'),index=False)
else:
    for n in range(n_repeats):
        experiment_table[n] = pd.DataFrame(experiment_table[n])
    #    writer = pd.ExcelWriter('experiment (sessions {}).xlsx'.format(n+1))
    #    experiment_table[n].to_excel(writer,'sheet1')
    #    writer.save()
        experiment_table[n].to_csv(os.path.join(csv_saving_dir,
                        'experiment (sessions {}, {}).csv'.format(n+1,subtitle)),index=False)
        if n == n_repeats -1:
            experiment_table[n].to_csv(os.path.join(csv_saving_dir,'calibration.csv'),index=False)



image_path_all = glob('../experiment_images_tilted/*/*/*.jpg')
image_path_used = results['probe_path'].tolist()
image_path_left = [item for item in image_path_all if (item not in image_path_used)]
image_path_left = pd.DataFrame({'image_left':image_path_left})
image_path_left.to_csv('image_left {}.csv'.format(subtitle),index=False)







