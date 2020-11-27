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

# sub 04
sub_name = 'sub-04'

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

sub04 = {
        'May_23-1833':'Usman_Forest_1-run1-session1',
        'May_23-1842':'Usman_Forest_1-run2-session1',
        'May_23-1852':'Usman_Forest_1-run3-session1',
        'May_23-1902':'Usman_Forest_1-run4-session1',
        'May_23-1910':'Usman_Forest_1-run5-session1',
        'May_23-1918':'Usman_Forest_1-run6-session1',
        'May_23-1927':'Usman_Forest_1-run7-session1',
        'May_23-1935':'Usman_Forest_1-run8-session1',
        'May_23-1943':'Usman_Forest_1-run9-session1',
        'Jun_06-1841':'USMAN_FOREST_2-run1-session2',
        'Jun_06-1849':'USMAN_FOREST_2-run2-session2',
        'Jun_06-1858':'USMAN_FOREST_2-run3-session2',
        'Jun_06-1906':'USMAN_FOREST_2-run4-session2',
        'Jun_06-1915':'USMAN_FOREST_2-run5-session2',
        'Jun_06-1924':'USMAN_FOREST_2-run6-session2',
        'Jun_06-1933':'USMAN_FOREST_2-run7-session2',
        'Jun_06-1948':'USMAN_FOREST_2-run8-session2',
        'Jun_06-1957':'USMAN_FOREST_2-run9-session2',
        'Jun_24-1455':'USMAN_FOREST_3-run1-session3',
        'Jun_24-1504':'USMAN_FOREST_3-run2-session3',
        'Jun_24-1512':'USMAN_FOREST_3-run3-session3',
        'Jun_24-1520':'USMAN_FOREST_3-run4-session3',
        'Jun_24-1529':'USMAN_FOREST_3-run5-session3',
        'Jun_24-1537':'USMAN_FOREST_3-run6-session3',
        'Jun_24-1547':'USMAN_FOREST_3-run7-session3',
        'Jun_24-1555':'USMAN_FOREST_3-run8-session3',
        'Jun_24-1603':'USMAN_FOREST_3-run9-session3',
        'Jun_25-1515':'USMAN_FOREST_4-run1-session4',
        'Jun_25-1523':'USMAN_FOREST_4-run2-session4',
        'Jun_25-1532':'USMAN_FOREST_4-run3-session4',
        'Jun_25-1540':'USMAN_FOREST_4-run4-session4',
        'Jun_25-1548':'USMAN_FOREST_4-run5-session4',
        'Jun_25-1557':'USMAN_FOREST_4-run6-session4',
        'Jun_25-1606':'USMAN_FOREST_4-run7-session4',
        'Jun_25-1616':'USMAN_FOREST_4-run8-session4',
        'Jun_25-1624':'USMAN_FOREST_4-run9-session4',
        'Jun_27-1545':'USMAN_FOREST_5-run1-session5',
        'Jun_27-1553':'USMAN_FOREST_5-run2-session5',
        'Jun_27-1601':'USMAN_FOREST_5-run3-session5',
        'Jun_27-1611':'USMAN_FOREST_5-run4-session5',
        'Jun_27-1620':'USMAN_FOREST_5-run5-session5',
        'Jun_27-1628':'USMAN_FOREST_5-run6-session5',
        'Jun_27-1637':'USMAN_FOREST_5-run7-session5',
        'Jun_27-1645':'USMAN_FOREST_5-run8-session5',
        'Jun_27-1654':'USMAN_FOREST_5-run9-session5',
        'Jun_28-1309':'USMAN_FOREST_6-run1-session6',
        'Jun_28-1317':'USMAN_FOREST_6-run2-session6',
        'Jun_28-1325':'USMAN_FOREST_6-run3-session6',
        'Jun_28-1333':'USMAN_FOREST_6-run4-session6',
        'Jun_28-1342':'USMAN_FOREST_6-run5-session6',
        'Jun_28-1350':'USMAN_FOREST_6-run6-session6',
        'Jun_28-1358':'USMAN_FOREST_6-run7-session6',
        'Jun_28-1407':'USMAN_FOREST_6-run8-session6',
        'Jun_28-1415':'USMAN_FOREST_6-run9-session6',
        }
for beha_key,fmri_value in sub04.items():
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












