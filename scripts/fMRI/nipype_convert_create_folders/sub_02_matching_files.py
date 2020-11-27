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
        'sub-02':'PATXI',
        'sub-03':'PEDRO',
        'sub-04':'USMAN'
        }

# sub 02
sub_name = 'sub-02'

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

sub02 = {
        'Mar_22-1939':'PATXI_1_FORREST-run1-session1',
        'Mar_22-1950':'PATXI_1_FORREST-run2-session1',
        'Mar_22-1959':'PATXI_1_FORREST-run3-session1',
        'Mar_22-2007':'PATXI_1_FORREST-run4-session1',
        'Mar_22-2016':'PATXI_1_FORREST-run5-session1',
        'Mar_22-2025':'PATXI_1_FORREST-run6-session1',
        'Mar_22-2038':'PATXI_1_FORREST-run7-session1',
        'Mar_22-2047':'PATXI_1_FORREST-run8-session1',
        'Mar_22-2056':'PATXI_1_FORREST-run9-session1',
        'Mar_25-1704':'PATXI_2_FORREST-run1-session2',
        'Mar_25-1714':'PATXI_2_FORREST-run2-session2',
        'Mar_25-1725':'PATXI_2_FORREST-run3-session2',
        'Mar_25-1733':'PATXI_2_FORREST-run4-session2',
        'Mar_25-1745':'PATXI_2_FORREST-run5-session2',
        'Mar_25-1754':'PATXI_2_FORREST-run6-session2',
        'Mar_25-1802':'PATXI_2_FORREST-run7-session2',
        'Mar_25-1811':'PATXI_2_FORREST-run8-session2',
        'Mar_25-1820':'PATXI_2_FORREST-run9-session2',
        'Apr_30-1035':'PAXTI_FOREST-run1-session3',
        'Apr_30-1044':'PAXTI_FOREST-run2-session3',
        'Apr_30-1053':'PAXTI_FOREST-run3-session3',
        'Apr_30-1103':'PAXTI_FOREST-run4-session3',
        'Apr_30-1112':'PAXTI_FOREST-run5-session3',
        'Apr_30-1124':'PAXTI_FOREST-run6-session3',
        'Apr_30-1132':'PAXTI_FOREST-run7-session3',
        'Apr_30-1144':'PAXTI_FOREST-run8-session3',
        'Apr_30-1153':'PAXTI_FOREST-run9-session3',
        'Apr_30-1203':'PAXTI_FOREST-run10-session3',
        'Apr_30-1213':'PAXTI_FOREST-run11-session3',
        'May_07-1922':'PATXI_FOREST_4-run1-session4',
        'May_07-1932':'PATXI_FOREST_4-run2-session4',
        'May_07-1947':'PATXI_FOREST_4-run3-session4',
        'May_07-1956':'PATXI_FOREST_4-run4-session4',
        'May_07-2010':'PATXI_FOREST_4-run5-session4',
        'May_15-1414':'PATXI_FOREST_5-run1-session5',
        'May_15-1424':'PATXI_FOREST_5-run2-session5',
        'May_15-1433':'PATXI_FOREST_5-run3-session5',
        'May_15-1442':'PATXI_FOREST_5-run4-session5',
        'May_15-1451':'PATXI_FOREST_5-run5-session5',
        'May_15-1503':'PATXI_FOREST_5-run6-session5',
        'May_15-1512':'PATXI_FOREST_5-run7-session5',
        'May_15-1523':'PATXI_FOREST_5-run8-session5',
        'May_15-1531':'PATXI_FOREST_5-run9-session5',
        'May_16-1157':'PATXI_FOREST6-run6-session6',
        'May_16-1207':'PATXI_FOREST6-run7-session6',
        'May_16-1216':'PATXI_FOREST6-run8-session6',
        'May_16-1224':'PATXI_FOREST6-run9-session6',
        'May_16-1234':'PATXI_FOREST6-run10-session6',
        'May_24-1648':'FOREST_PATXI_7-run1-session7',
        'May_24-1658':'FOREST_PATXI_7-run2-session7',
        'May_24-1706':'FOREST_PATXI_7-run3-session7',
        'May_24-1714':'FOREST_PATXI_7-run4-session7',
        'May_24-1727':'FOREST_PATXI_7-run5-session7',
        'May_24-1735':'FOREST_PATXI_7-run6-session7',
        'May_24-1743':'FOREST_PATXI_7-run7-session7',
        'May_24-1753':'FOREST_PATXI_7-run8-session7',
        }
for beha_key,fmri_value in sub02.items():
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












