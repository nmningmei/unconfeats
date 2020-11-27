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
        'sub-04':'USMAN'
        }

# sub 03
sub_name = 'sub-03'

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

sub03 = {
        'Apr_29-1540':'pedro_forest_1-run1-session1',
        'Apr_29-1550':'pedro_forest_1-run2-session1',
        'Apr_29-1559':'pedro_forest_1-run3-session1',
        'Apr_29-1614':'pedro_forest_1-run4-session1',
        'Apr_29-1622':'pedro_forest_1-run5-session1',
        'Apr_29-1630':'pedro_forest_1-run6-session1',
        'Apr_29-1640':'pedro_forest_1-run7-session1',
        'Apr_29-1650':'pedro_forest_1-run8-session1',
        'Apr_29-1659':'pedro_forest_1-run9-session1',
        'May_15-1759':'pedro_forest_2-run1-session2',
        'May_15-1809':'pedro_forest_2-run2-session2',
        'May_15-1817':'pedro_forest_2-run3-session2',
        'May_15-1826':'pedro_forest_2-run4-session2',
        'May_15-1834':'pedro_forest_2-run5-session2',
        'May_15-1842':'pedro_forest_2-run6-session2',
        'May_15-1851':'pedro_forest_2-run7-session2',
        'May_15-1907':'pedro_forest_2_2-run7-session2',
        'May_15-1917':'pedro_forest_2_2-run8-session2',
        'May_15-1925':'pedro_forest_2_2-run9-session2',
        'May_22-0952':'PEDRO_FOREST_3-run1-session3',
        'May_22-1002':'PEDRO_FOREST_3-run2-session3',
        'May_22-1013':'PEDRO_FOREST_3-run3-session3',
        'May_22-1022':'PEDRO_FOREST_3-run4-session3',
        'May_22-1030':'PEDRO_FOREST_3-run5-session3',
        'May_22-1039':'PEDRO_FOREST_3-run6-session3',
        'May_22-1049':'PEDRO_FOREST_3-run7-session3',
        'May_22-1057':'PEDRO_FOREST_3-run8-session3',
        'May_22-1107':'PEDRO_FOREST_3-run9-session3',
        'May_27-1803':'Pedro_forrest_4-run1-session4',
        'May_27-1813':'Pedro_forrest_4-run2-session4',
        'May_27-1821':'Pedro_forrest_4-run3-session4',
        'May_27-1829':'Pedro_forrest_4-run4-session4',
        'May_27-1837':'Pedro_forrest_4-run5-session4',
        'May_27-1846':'Pedro_forrest_4-run6-session4',
        'May_27-1856':'Pedro_forrest_4-run7-session4',
        'May_27-1905':'Pedro_forrest_4-run8-session4',
        'May_27-1913':'Pedro_forrest_4-run9-session4',
        'May_28-1305':'PEDRO_FOREST_5-run1-session5',
        'May_28-1314':'PEDRO_FOREST_5-run2-session5',
        'May_28-1323':'PEDRO_FOREST_5-run3-session5',
        'May_28-1332':'PEDRO_FOREST_5-run4-session5',
        'May_28-1343':'PEDRO_FOREST_5-run5-session5',
        'May_28-1354':'PEDRO_FOREST_5-run6-session5',
        'May_28-1403':'PEDRO_FOREST_5-run7-session5',
        'May_28-1412':'PEDRO_FOREST_5-run8-session5',
        'May_28-1422':'PEDRO_FOREST_5-run9-session5',
        'Jun_26-1516':'PEDRO_FOREST_6-run1-session6',
        'Jun_26-1527':'PEDRO_FOREST_6-run2-session6',
        'Jun_26-1535':'PEDRO_FOREST_6-run3-session6',
        'Jun_26-1543':'PEDRO_FOREST_6-run4-session6',
        'Jun_26-1552':'PEDRO_FOREST_6-run5-session6',
        'Jun_26-1600':'PEDRO_FOREST_6-run6-session6',
        'Jun_26-1610':'PEDRO_FOREST_6-run7-session6',
        'Jun_26-1618':'PEDRO_FOREST_6-run8-session6',
        'Jun_26-1627':'PEDRO_FOREST_6-run9-session6',
        }
for beha_key,fmri_value in sub03.items():
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












