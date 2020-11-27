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

# sub 01
sub_name = 'sub-01'

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

sub01 = {
        'Apr_03-1839':'ning_forest_2-run1-session2',
        'Apr_03-1850':'ning_forest_2-run2-session2',
        'Apr_03-1858':'ning_forest_2-run3-session2',
        'Apr_03-1907':'ning_forest_2-run4-session2',
        'Apr_03-1916':'ning_forest_2-run5-session2',
        'Apr_03-1927':'ning_forest_2-run6-session2',
        'Apr_03-1936':'ning_forest_2-run7-session2',
        'Apr_03-1947':'ning_forest_2-run8-session2',
        'Apr_03-1956':'ning_forest_2-run9-session2',
        'Apr_04-1835':'NING_FORREST_3-run1-session3',
        'Apr_04-1846':'NING_FORREST_3-run2-session3',
        'Apr_04-1855':'NING_FORREST_3-run3-session3',
        'Apr_04-1903':'NING_FORREST_3-run4-session3',
        'Apr_04-1912':'NING_FORREST_3-run5-session3',
        'Apr_04-1921':'NING_FORREST_3-run6-session3',
        'Apr_04-1930':'NING_FORREST_3-run7-session3',
        'Apr_04-1939':'NING_FORREST_3-run8-session3',
        'Apr_04-1947':'NING_FORREST_3-run9-session3',
        'Apr_05-1919':'NING_FORREST_4-run1-session4',
        'Apr_05-1929':'NING_FORREST_4-run2-session4',
        'Apr_05-1938':'NING_FORREST_4-run3-session4',
        'Apr_05-1947':'NING_FORREST_4-run4-session4',
        'Apr_05-1955':'NING_FORREST_4-run5-session4',
        'Apr_05-2004':'NING_FORREST_4-run61-session4',
        'Apr_05-2015':'NING_FORREST_4-run62-session4',
        'Apr_05-2026':'NING_FORREST_4-run7-session4',
        'Apr_05-2034':'NING_FORREST_4-run8-session4',
        'Apr_05-2045':'NING_FORREST_4-run9-session4',
        'Apr_09-1446':'NING_FOREST_5-run1-session5',
        'Apr_09-1457':'NING_FOREST_5-run2-session5',
        'Apr_09-1506':'NING_FOREST_5-run3-session5',
        'Apr_09-1515':'NING_FOREST_5-run4-session5',
        'Apr_09-1523':'NING_FOREST_5-run5-session5',
        'Apr_09-1531':'NING_FOREST_5-run6-session5',
        'Apr_09-1540':'NING_FOREST_5-run7-session5',
        'Apr_09-1550':'NING_FOREST_5-run8-session5',
        'Apr_09-1559':'NING_FOREST_5-run9-session5',
        'Apr_09-1822':'NING_FORREST_6-run1-session6',
        'Apr_09-1835':'NING_FORREST_6-run2-session6',
        'Apr_09-1843':'NING_FORREST_6-run3-session6',
        'Apr_09-1853':'NING_FORREST_6-run4-session6',
        'Apr_09-1902':'NING_FORREST_6-run5-session6',
        'Apr_09-1917':'NING_FORREST_6-run6-session6',
        'Apr_09-1926':'NING_FORREST_6-run7-session6',
        'Apr_09-1936':'NING_FORREST_6-run8-session6',
        'Apr_09-1946':'NING_FORREST_6-run9-session6',
        'May_07-1541':'NING_FOREST_1-run1-session7',
        'May_07-1552':'NING_FOREST_1-run2-session7',
        'May_07-1601':'NING_FOREST_1-run3-session7',
        'May_07-1609':'NING_FOREST_1-run4-session7',
        'May_07-1622':'NING_FOREST_1-run5-session7',
        'May_07-1631':'NING_FOREST_1-run6-session7',
        'May_07-1639':'NING_FOREST_1-run7-session7',
        'May_07-1649':'NING_FOREST_1-run8-session7',
        'May_07-1658':'NING_FOREST_1-run9-session7',
        }
for beha_key,fmri_value in sub01.items():
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












