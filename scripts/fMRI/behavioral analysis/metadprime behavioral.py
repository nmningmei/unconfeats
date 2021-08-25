#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 10:42:38 2021

@author: ning
"""

import numpy as np
import pandas as pd
from metadPy import metad

df = pd.read_csv('~/Downloads/behavioral stacked dataframe.csv')
df['Accuracy'] = np.array(df['correctAns_raw'].values == df['response.keys_raw'].values,
                          dtype = int)
df['Confidence'] = df['visible.keys_raw'].values.astype(int)
df['Stimuli'] = df['correctAns_raw'].values.astype(int) - 1
df['Response'] = df['response.keys_raw'].values.astype(int) - 1

res = metad(data = df[['Stimuli','Accuracy','Confidence','sub']],
            nRatings = 3,
            stimuli = 'Stimuli',
            accuracy = 'Accuracy',
            confidence = 'Confidence',
            subject = 'sub',
            verbose = 0,
            )
