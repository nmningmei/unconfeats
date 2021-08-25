#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 18:17:50 2019

@author: nmei

Create a csv and a tsv file that corresponding to each session each run.

The csv/tsv file contains the information that corresponding to each volume (508 in total)

"""
import numpy  as np
import pandas as pd

from shutil  import copyfile
from glob    import glob
from nibabel import load as load_mri

copyfile('../../utils.py','utils.py')
from utils import (read_behavorial_file,
                   extract)
import os
import re
import csv

sub             = 'sub-07'
folder_name     = 'onset_more_volumes'
main_dir        = '/bcbl/home/home_n-z/nmei/MRI/uncon_feat/MRI/'
MRI_dir         = '{}{}/func/'.format(main_dir,sub)
MRI_data        = np.sort(glob(os.path.join(MRI_dir,
                                '*',# session
                                '*',# run
                                'outputs',
                                'func',
                                'ICAed_filtered',
                                '*.nii.gz')))

sub_behavorial  = sub
behavorial_dir  = '../../../data/behavioral/{}'.format(sub_behavorial)
behavorial_data = glob(os.path.join(behavorial_dir,'*','*.csv'))
behavorial_data = np.sort(behavorial_data)


visible_map = {1:'unconscious',
               2:'glimpse',
               3:'conscious',
               99:'missing data'}

"""
get the 'day' and 'run' for each MRI data
'day' means the day on which the scanning
'run' means one of the nine runs in that day
because the behavioral pychophy file contain 'session' and 'block', we need to 
create these attributes
'session' means the order of a 96-trial experiment
'block' means one of the three blocks that divides the 96-trial experiment
'session' and 'block' are pre-difined so they are trackable.
"""
df_temp = []
for temp in MRI_data:
    n_sub,n_day,_,n_run = re.findall(r'\d+',temp)
    df_temp.append([n_day,n_run,temp])
df_temp             = pd.DataFrame(df_temp,
                                   columns = ['day',
                                              'run',
                                              'MRI_file_name']).sort_values([
                                              'day',
                                              'run']).reset_index(drop=True)


for ii,row in df_temp.iterrows():
    MRI_file                = row['MRI_file_name']
    n_day                   = row['day']
    n_run                   = row['run']
    # pick the one psychopy file that contains the same "session" and "block" as the MRI file
    behavorial_file_picked = [item for item in behavorial_data if (f'session-{n_day}' in item) and (f'run-{n_run}' in item)]
    
    print()
    print(MRI_file,behavorial_file_picked[0],f'day {n_day}, run {n_run}')
    # load the fMRI data with nibibal
    BOLD                = load_mri(MRI_file)
    # read the psychopy file
    df                  = read_behavorial_file(behavorial_file_picked[0])
    # psychopy did something strange and made the column contain double quotes
    numerical_columns   = ['probe_Frames_raw',
                           'response.keys_raw',
                           'visible.keys_raw',]
    for col_name in numerical_columns:
        try:
            df[col_name]    = df[col_name].apply(extract)
        except:
            df[col_name]    = df['probeFrames_raw'].apply(extract)
    
    # sort the row order
    df                  = df.sort_values(['order'])
    # pick the greenville time coordinate
    col_of_interest     = ['image_onset_time_raw',]
    for col in col_of_interest:
        df[col] = df[col] - 0.85 * 10 # take out 10 volumes during preprocessing
    # 0.4: the premasking
    # 0.5: fixation
    # 0.5: blank after the fixation
    # so the start is the onset of the beginning of a trial
    df['start'] = df['image_onset_time_raw'] - 0.4 - 0.5 - 0.5
    # we are interested in the volumes that fall between 4 to 7 seconds after the onset
    # of the probe image
    # or 0 to 3 seconds from the onset
    # or 2 to 5 seconds after the visibility response
    df['t1']    = df['image_onset_time_raw'] + 4
    df['t2']    = df['image_onset_time_raw'] + 7
    
    # after preparing the time segments, we can start inserting the 
    # so-called "time coordinate"
    total_volumes   = BOLD.shape[-1]
    time_coor       = np.arange(0,total_volumes * 0.85,0.85)
    # initialize the placeholders for numerical data
    trials          = np.zeros(time_coor.shape)
    visibility      = trials.copy()
    correctAns      = trials.copy()
    response        = trials.copy()
    correct         = trials.copy()
    RT_response     = trials.copy()
    RT_visibility   = trials.copy()
    probe_frame     = trials.copy()
    # initialize the placeholders for string data
    targets         = np.array(['AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'] * time_coor.shape[0])
    subcategory     = targets.copy()
    options         = targets.copy()
    labels          = targets.copy()
    paths           = targets.copy()
    # filling in information
    for ii,row in df.iterrows():
        # pick all the rest of rows where the time coordinates are greater
        # than the begining of the given trial
        # say for trial 1, all the rows will be selected
        # but it is fine, in trial 2, those are marked as trial 1 
        # will be leave alone because their time coordinates are less than
        # the beginning of trial 2
        # and so on
        idx                 = np.where(time_coor >= row['start'])
        trials[idx]         = row['order'] + 1
        targets[idx]        = row['category']
        subcategory[idx]    = row['subcategory']
        labels[idx]         = row['label']
        visibility[idx]     = row['visible.keys_raw']
        correctAns[idx]     = row['correctAns_raw']
        correct[idx]        = row['response.corr_raw']
        response[idx]       = row['response.keys_raw']
        options[idx]        = row['response_window_raw']
        RT_response[idx]    = row['response.rt_raw']
        RT_visibility[idx]  = row['visible.rt_raw']
        probe_frame[idx]    = row['probe_Frames_raw']
        paths[idx]          = row['probe_path'].split('/')[-1]
    to_tsv = pd.DataFrame(dict(
            time_coor       = time_coor,
            trials          = trials,
            targets         = targets,
            subcategory     = subcategory,
            labels          = labels,
            visibility      = visibility,
            correctAns      = correctAns,
            correct         = correct,
            response        = response,
            options         = options,
            RT_response     = RT_response,
            RT_visibility   = RT_visibility,
            probe_frame     = probe_frame,
            paths           = paths,)
        )
    
    # pick those volumes that we are interested in: those between 4 to 7 seconds
    temp        = []
    for ii,row in to_tsv.iterrows():
        time    = row['time_coor']
        # df[['t1','t2']].values is a 508 X 2 matrix that contains all the possible
        # time intervals
        # if any of the time in the time coordinate falls between any of the possible
        # time intervals, those are what we are interested in
        # mark them "1" else "0"
        if any([np.logical_and(interval[0] < time,
                               time < interval[1]) for interval in df[['t1','t2']].values]):
            temp.append(1)
        else:
            temp.append(0)
    to_tsv['volume_interest']   = temp
    to_tsv['visibility']        = to_tsv['visibility'].map(visible_map)
    
    # define output directory, in which the raw-raw fMRI data lives in
    output_dir = os.path.join('/'.join(MRI_file.split('/')[:-3]),f'{folder_name}')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    print(os.path.join(output_dir,
    '{}_session_{}_run_{}.csv'.format(sub,n_day,n_run)))
    # save csv
    to_tsv.to_csv(os.path.join(output_dir,
                               '{}_session_{}_run_{}.csv'.format(sub,n_day,n_run)),
        index = False)
    # prepare for converting csv to tsv
    f_csv = os.path.join(output_dir,
                         '{}_session_{}_run_{}.csv'.format(sub,n_day,n_run))
    f_tsv = os.path.join(output_dir,
                         '{}_session_{}_run_{}.tsv'.format(sub,n_day,n_run))
    
    with open(f_csv,'r') as csvin,open(f_tsv,'w') as tsvout:
        csvin   = csv.reader(csvin)
        tsvout  = csv.writer(tsvout,delimiter='\t')
        
        for row in csvin:
            tsvout.writerow(row)


















