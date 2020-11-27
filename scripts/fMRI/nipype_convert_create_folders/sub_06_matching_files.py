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
        'sub-06':'STEFANO',
        }

# sub 06
sub_name = 'sub-06'

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

sub06 = {
        'Jul_08-1821':'STEFANO_FOREST_1-run1-session1',
        'Jul_08-1830':'STEFANO_FOREST_1-run2-session1',
        'Jul_08-1839':'STEFANO_FOREST_1-run3-session1',
        'Jul_08-1848':'STEFANO_FOREST_1-run4-session1',
        'Jul_08-1857':'STEFANO_FOREST_1-run5-session1',
        'Jul_08-1906':'STEFANO_FOREST_1-run6-session1',
        'Jul_08-1916':'STEFANO_FOREST_1-run7-session1',
        'Jul_08-1925':'STEFANO_FOREST_1-run8-session1',
        'Jul_08-1934':'STEFANO_FOREST_1-run9-session1',
        'Jul_09-1806':'STEFANO_FOREST_2-run1-session2',
        'Jul_09-1815':'STEFANO_FOREST_2-run2-session2',
        'Jul_09-1823':'STEFANO_FOREST_2-run3-session2',
        'Jul_09-1832':'STEFANO_FOREST_2-run4-session2',
        'Jul_09-1840':'STEFANO_FOREST_2-run5-session2',
        'Jul_09-1849':'STEFANO_FOREST_2-run6-session2',
        'Jul_09-1858':'STEFANO_FOREST_2-run7-session2',
        'Jul_09-1906':'STEFANO_FOREST_2-run8-session2',
        'Jul_09-1915':'STEFANO_FOREST_2-run9-session2',
        'Jul_10-1907':'STEFANO_FOREST_3-run1-session3',
        'Jul_10-1915':'STEFANO_FOREST_3-run2-session3',
        'Jul_10-1923':'STEFANO_FOREST_3-run3-session3',
        'Jul_10-1932':'STEFANO_FOREST_3-run4-session3',
        'Jul_10-1940':'STEFANO_FOREST_3-run5-session3',
        'Jul_10-1949':'STEFANO_FOREST_3-run6-session3',
        'Jul_10-1958':'STEFANO_FOREST_3-run7-session3',
        'Jul_10-2007':'STEFANO_FOREST_3-run8-session3',
        'Jul_10-2015':'STEFANO_FOREST_3-run9-session3',
        'Jul_18-1531':'STEFANO_FOREST_5-run1-session5',
        'Jul_18-1539':'STEFANO_FOREST_5-run2-session5',
        'Jul_18-1547':'STEFANO_FOREST_5-run3-session5',
        'Jul_18-1556':'STEFANO_FOREST_5-run4-session5',
        'Jul_18-1605':'STEFANO_FOREST_5-run5-session5',
        'Jul_18-1613':'STEFANO_FOREST_5-run6-session5',
        'Jul_18-1621':'STEFANO_FOREST_5-run7-session5',
        'Jul_18-1630':'STEFANO_FOREST_5-run8-session5',
        'Jul_18-1638':'STEFANO_FOREST_5-run9-session5',
        'Jul_19-1449':'STEFANO_FOREST_6-run1-session6',
        'Jul_19-1457':'STEFANO_FOREST_6-run2-session6',
        'Jul_19-1505':'STEFANO_FOREST_6-run3-session6',
        'Jul_19-1514':'STEFANO_FOREST_6-run4-session6',
        'Jul_19-1523':'STEFANO_FOREST_6-run5-session6',
        'Jul_19-1531':'STEFANO_FOREST_6-run6-session6',
        'Jul_19-1539':'STEFANO_FOREST_6-run7-session6',
        'Jul_19-1548':'STEFANO_FOREST_6-run8-session6',
        'Jul_19-1556':'STEFANO_FOREST_6-run9-session6',
        'Jul_24-1546':'STEFANO_FOREST_4_2-run1-session4',
        'Jul_24-1554':'STEFANO_FOREST_4_2-run2-session4',
        'Jul_24-1604':'STEFANO_FOREST_4_2-run3-session4',
        'Jul_24-1613':'STEFANO_FOREST_4_2-run4-session4',
        'Jul_24-1621':'STEFANO_FOREST_4_2-run5-session4',
        'Jul_24-1630':'STEFANO_FOREST_4_2-run6-session4',
        'Jul_24-1639':'STEFANO_FOREST_4_2-run7-session4',
        'Jul_24-1648':'STEFANO_FOREST_4_2-run8-session4',
        'Jul_24-1657':'STEFANO_FOREST_4_2-run9-session4',
        }
for beha_key,fmri_value in sub06.items():
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












