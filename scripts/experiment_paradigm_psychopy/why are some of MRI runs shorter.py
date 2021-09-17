#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 16:52:11 2019

@author: nmei
"""

import pandas as pd
from glob import glob

working_data = glob('*trials.csv')
working_log = glob('*.log')
mapping = {"'1.0'":1,
           "'2.0'":2,
           "'3.0'":3,
           "'4.0'":4,
           "'5.0'":5,
           "'6.0'":6,
           "'7.0'":7,
           "'8.0'":8,
           "'9.0'":9,
           '1':1,
           '2':2,
           '3':3,
           '4':4,
           '5':5,
           '6':6,
           '7':7,
           '8':8,
           '9':9}

for log,f in zip(working_log,working_data):
    temp = pd.read_csv(f)
    temp = temp.sort_values(['order'])
    session = int(temp['index'].iloc[39])
    block = int(temp['index'].iloc[42])
    temp = temp.dropna()
    last_time_stamp = temp['visibil_resptime_raw'].iloc[-1]
    if last_time_stamp < 420:
        star = "*"
    else:
        star = " "
    with open(log,'r') as handle:
        for line in handle:
            if "probe: autoDraw = True" in line:
                text = 'first probe'
                key = line.split('\t')[0]
                break
        handle.close()
    try:
        trial_length = 0.2 + 0.1 * temp['probeFrames_raw'].astype(int) + 0.2 + temp['jitter1_raw'] + 1.5 + 1.5 + temp['jitter2_raw']
        print('session {} block {}, average trial length = {:.1f}+/-{:.1f}{}, {} at {}'.format(session,block,
              trial_length.mean(),
              trial_length.std(),
              star,
              text,
              key,))
    except:
        temp['probeFrames_raw'] = temp['probeFrames_raw'].map(mapping)
        trial_length = 0.2 + 0.1 * temp['probeFrames_raw'].astype(int) + 0.2 + temp['jitter1_raw'] + 1.5 + 1.5 + temp['jitter2_raw']
        print('session {} block {}, average trial length = {:.1f}+/-{:.1f}{}, {} at {}'.format(session,block,
              trial_length.mean(),
              trial_length.std(),
              star,
              text,
              key))




for log,f in zip(working_log,working_data):
    temp = pd.read_csv(f)
    temp = temp.sort_values(['order'])
    session = int(temp['index'].iloc[39])
    block = int(temp['index'].iloc[42])
    temp = temp.dropna()
    last_time_stamp = temp['visibil_resptime_raw'].iloc[-1]
    















