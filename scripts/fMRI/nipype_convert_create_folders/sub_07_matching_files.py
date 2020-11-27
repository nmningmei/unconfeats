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
        'sub-07':'BORJA',
        }

# sub 07
sub_name = 'sub-07'

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

sub07 = {
        'Jun_24-1913':'BORJA_FOREST_1-run1-session1',
        'Jun_24-1922':'BORJA_FOREST_1-run2-session1',
        'Jun_24-1931':'BORJA_FOREST_1-run3-session1',
        'Jun_24-1939':'BORJA_FOREST_1-run4-session1',
        'Jun_24-1948':'BORJA_FOREST_1-run5-session1',
        'Jun_24-1957':'BORJA_FOREST_1-run6-session1',
        'Jun_24-2005':'BORJA_FOREST_1-run7-session1',
        'Jun_24-2014':'BORJA_FOREST_1-run8-session1',
        'Jun_24-2023':'BORJA_FOREST_1-run9-session1',
        'Jul_09-1939':'BORJA_FOREST_2-run1-session2',
        'Jul_09-1947':'BORJA_FOREST_2-run2-session2',
        'Jul_09-1955':'BORJA_FOREST_2-run3-session2',
        'Jul_09-2004':'BORJA_FOREST_2-run4-session2',
        'Jul_09-2013':'BORJA_FOREST_2-run5-session2',
        'Jul_09-2022':'BORJA_FOREST_2-run6-session2',
        'Jul_09-2030':'BORJA_FOREST_2-run7-session2',
        'Jul_09-2038':'BORJA_FOREST_2-run8-session2',
        'Jul_09-2046':'BORJA_FOREST_2-run9-session2',
        'Jul_15-1909':'BORJA_FOREST_3-run1-session3',
        'Jul_15-1917':'BORJA_FOREST_3-run2-session3',
        'Jul_15-1926':'BORJA_FOREST_3-run3-session3',
        'Jul_15-1934':'BORJA_FOREST_3-run4-session3',
        'Jul_15-1942':'BORJA_FOREST_3-run5-session3',
        'Jul_15-1950':'BORJA_FOREST_3-run6-session3',
        'Jul_15-1959':'BORJA_FOREST_3-run7-session3',
        'Jul_15-2007':'BORJA_FOREST_3-run8-session3',
        'Jul_15-2016':'BORJA_FOREST_3-run9-session3',
        'Jul_16-1901':'BORJA_FOREST_4-run1-session4',
        'Jul_16-1910':'BORJA_FOREST_4-run2-session4',
        'Jul_16-1918':'BORJA_FOREST_4-run3-session4',
        'Jul_16-1926':'BORJA_FOREST_4-run4-session4',
        'Jul_16-1934':'BORJA_FOREST_4-run5-session4',
        'Jul_16-1943':'BORJA_FOREST_4-run6-session4',
        'Jul_16-1952':'BORJA_FOREST_4-run7-session4',
        'Jul_16-2000':'BORJA_FOREST_4-run8-session4',
        'Jul_16-2009':'BORJA_FOREST_4-run9-session4',
        'Jul_17-1855':'BORJA_FOREST_5-run1-session5',
        'Jul_17-1904':'BORJA_FOREST_5-run2-session5',
        'Jul_17-1912':'BORJA_FOREST_5-run3-session5',
        'Jul_17-1921':'BORJA_FOREST_5-run4-session5',
        'Jul_17-1929':'BORJA_FOREST_5-run5-session5',
        'Jul_17-1939':'BORJA_FOREST_5-run6-session5',
        'Jul_17-1947':'BORJA_FOREST_5-run7-session5',
        'Jul_17-1955':'BORJA_FOREST_5-run8-session5',
        'Jul_17-2003':'BORJA_FOREST_5-run9-session5',
        'Jul_18-1858':'BORJA_FOREST_6-run1-session6',
        'Jul_18-1910':'BORJA_FOREST_6-run2-session6',
        'Jul_18-1918':'BORJA_FOREST_6-run3-session6',
        'Jul_18-1926':'BORJA_FOREST_6-run4-session6',
        'Jul_18-1935':'BORJA_FOREST_6-run5-session6',
        'Jul_18-1944':'BORJA_FOREST_6-run6-session6',
        'Jul_18-1952':'BORJA_FOREST_6-run7-session6',
        'Jul_18-2002':'BORJA_FOREST_6-run8-session6',
        'Jul_18-2010':'BORJA_FOREST_6-run9-session6'
        }
for beha_key,fmri_value in sub07.items():
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












