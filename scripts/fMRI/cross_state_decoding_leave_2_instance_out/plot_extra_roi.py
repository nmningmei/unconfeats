#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 12:34:40 2021

@author: nmei
"""
import os

from glob import glob

import numpy as np
import pandas as pd
import seaborn as sns

for folder_name in ['decoding_LOO_extra','decoding_LOOC_extra']:
    working_dir = '../../../../results/MRI/nilearn/{}/*'.format(folder_name)
    working_data = glob(os.path.join(working_dir,'*csv'))
    df = pd.concat([pd.read_csv(f) for f in working_data])
    df_score = df[df['model'] == 'None + Linear-SVM']
    df_chance = df[df['model'] != 'None + Linear-SVM']
    g = sns.catplot(x = 'condition_target',
                    y ='roc_auc',
                    hue = 'sub',
                    row = 'condition_source',
                    data = df_score,
                    kind = 'bar',
                    aspect = 2,
                )
    (g.set_titles('Source condition: {row_name}')
      .set_axis_labels('Target condition','ROC AUC'))
    [ax.axhline(0.5,linestyle = '--',color = 'black',) for ax in g.axes.flatten()]