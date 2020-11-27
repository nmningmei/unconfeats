#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 14:02:41 2019

@author: nmei
"""

import os
import re
from glob import glob
from tqdm import tqdm
from shutil import copyfile

sub_matching_dict = {
        'sub-01':'NING',
        'sub-02':'PATXI',
        'sub-03':'PEDRO',
        'sub-04':'USMAN',
        'sub-05':'BOJANA',
        }

# sub 05
sub_name = 'sub-05'

working_dir = f'/export/home/nmei/nmei/MRI/converted/{sub_name}/'
fmri_files = [item for item in glob(os.path.join(working_dir,'*','*','*','*.nii.gz')) if ('MGH' not in item)]
json_files = [item for item in glob(os.path.join(working_dir,'*','*','*','*.json')) if ('MGH' not in item)]
behavioral_dir = glob(f'../../../data/behavioral/{sub_matching_dict[sub_name].lower()}*fMRI')[0]
behavioral_files = [item for item in glob(os.path.join(behavioral_dir,'*trials.csv')) if ('calibration' not in item)]

behaviroal_corrected_dir = f'../../../data/behavioral/{sub_name}'
if not os.path.exists(behaviroal_corrected_dir):
    os.makedirs(behaviroal_corrected_dir)

fmri_corrected_dir = f'../../../data/MRI/{sub_name}/func'
if not os.path.exists(fmri_corrected_dir):
    os.makedirs(fmri_corrected_dir)

sub05 = {
        'Jul_13-1154':'BOJANA_FOREST_1-run1-session1',
        'Jul_13-1202':'BOJANA_FOREST_1-run2-session1',
        'Jul_13-1212':'BOJANA_FOREST_1-run3-session1',
        'Jul_13-1226':'BOJANA_FOREST_1-run4-session1',
        'Jul_13-1236':'BOJANA_FOREST_1-run5-session1',
        'Jul_13-1246':'BOJANA_FOREST_1-run6-session1',
        'Jul_13-1256':'BOJANA_FOREST_1-run7-session1',
        'Jul_13-1308':'BOJANA_FOREST_1-run8-session1',
        'Jul_13-1317':'BOJANA_FOREST_1-run9-session1',
        'Jul_24-1719':'BOJANA_FOREST_2-run1-session2',
        'Jul_24-1727':'BOJANA_FOREST_2-run2-session2',
        'Jul_24-1736':'BOJANA_FOREST_2-run3-session2',
        'Jul_24-1745':'BOJANA_FOREST_2-run4-session2',
        'Jul_24-1753':'BOJANA_FOREST_2-run5-session2',
        'Jul_24-1802':'BOJANA_FOREST_2-run6-session2',
        'Jul_24-1811':'BOJANA_FOREST_2-run7-session2',
        'Jul_24-1819':'BOJANA_FOREST_2-run8-session2',
        'Jul_24-1827':'BOJANA_FOREST_2-run9-session2',
        'Jul_11-1944':'BOJANA_FOREST_3-run1-session3',
        'Jul_11-1952':'BOJANA_FOREST_3-run2-session3',
        'Jul_11-2001':'BOJANA_FOREST_3-run3-session3',
        'Jul_11-2010':'BOJANA_FOREST_3-run4-session3',
        'Jul_11-2019':'BOJANA_FOREST_3-run5-session3',
        'Jul_11-2029':'BOJANA_FOREST_3-run6-session3',
        'Jul_11-2038':'BOJANA_FOREST_3-run7-session3',
        'Jul_11-2047':'BOJANA_FOREST_3-run8-session3',
        'Jul_11-2056':'BOJANA_FOREST_3-run9-session3',
        'Jul_12-1824':'BOJANA_FOREST_4-run1-session4',
        'Jul_12-1833':'BOJANA_FOREST_4-run2-session4',
        'Jul_12-1842':'BOJANA_FOREST_4-run3-session4',
        'Jul_12-1850':'BOJANA_FOREST_4-run4-session4',
        'Jul_12-1858':'BOJANA_FOREST_4-run5-session4',
        'Jul_12-1908':'BOJANA_FOREST_4-run6-session4',
        'Jul_12-1917':'BOJANA_FOREST_4-run7-session4',
        'Jul_12-1925':'BOJANA_FOREST_4-run8-session4',
        'Jul_12-1935':'BOJANA_FOREST_4-run9-session4',
        'Jul_15-1715':'BOJANA_FOREST_5-run1-session5',
        'Jul_15-1723':'BOJANA_FOREST_5-run2-session5',
        'Jul_15-1743':'BOJANA_FOREST_5-run3-session5',
        'Jul_15-1751':'BOJANA_FOREST_5-run4-session5',
        'Jul_15-1759':'BOJANA_FOREST_5-run5-session5',
        'Jul_15-1808':'BOJANA_FOREST_5-run6-session5',
        'Jul_15-1816':'BOJANA_FOREST_5-run7-session5',
        'Jul_15-1826':'BOJANA_FOREST_5-run8-session5',
        'Jul_15-1835':'BOJANA_FOREST_5-run9-session5',
        'Jul_16-1715':'BOJANA_FOREST_6_1-run1-session6',
        'Jul_16-1723':'BOJANA_FOREST_6_1-run2-session6',
        'Jul_16-1737':'BOJANA_FOREST_6_2-run3-session6',
        'Jul_16-1748':'BOJANA_FOREST_6_2-run4-session6',
        'Jul_16-1756':'BOJANA_FOREST_6_2-run5-session6',
        'Jul_16-1805':'BOJANA_FOREST_6_2-run6-session6',
        'Jul_16-1813':'BOJANA_FOREST_6_2-run7-session6',
        'Jul_16-1821':'BOJANA_FOREST_6_2-run8-session6',
        'Jul_16-1830':'BOJANA_FOREST_6_2-run9-session6',
        }
for beha_key,fmri_value in sub05.items():
    beha_key,fmri_value
    beha_picked = [item for item in behavioral_files if \
                         (beha_key.split('-')[0] in item) and \
                         (beha_key.split('-')[1] in item)][0]
    fmri_picked = [item for item in fmri_files if \
                         (fmri_value.split('-')[0] in item) and \
                         (fmri_value.split('-')[1] in item)][0]
    json_picked = [item for item in json_files if \
                         (fmri_value.split('-')[0] in item) and \
                         (fmri_value.split('-')[1] in item)][0]
#    print(beha_picked,fmri_picked,json_picked,'\n')
    n_session = int(fmri_value.split('-')[-1][-1])
    session = f'session-0{n_session}'
    n_run = int(re.findall(r'\d+',fmri_value.split('-')[1])[0])
    run = f'run-{str(n_run).zfill(2)}'
    print(session,run,fmri_value.split('-')[1])
    if not os.path.exists(os.path.join(fmri_corrected_dir,
                        session,
                        f'{sub_name}_unfeat_{run}',)):
        os.makedirs(os.path.join(fmri_corrected_dir,
                                 session,
                                 f'{sub_name}_unfeat_{run}',))
    if not os.path.exists(os.path.join(behaviroal_corrected_dir,
                                       session,)):
        os.makedirs(os.path.join(behaviroal_corrected_dir,
                                 session,))
    fmri_destination = os.path.join(fmri_corrected_dir,
                                    session,
                                    f'{sub_name}_unfeat_{run}',
                                    f'{sub_name}_unfeat_{run}_bold.nii.gz')
    json_destination = os.path.join(fmri_corrected_dir,
                                    session,
                                    f'{sub_name}_unfeat_{run}',
                                    f'{sub_name}_unfeat_{run}_bold.json')
    beha_destination = os.path.join(behaviroal_corrected_dir,
                                    session,
                                    f'{sub_name}_unfeat_{run}.csv')
    
    copyfile(fmri_picked,fmri_destination)
    copyfile(json_picked,json_destination)
    copyfile(beha_picked,beha_destination)












