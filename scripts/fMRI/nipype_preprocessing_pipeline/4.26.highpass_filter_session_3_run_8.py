#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 12:35:01 2019

@author: nmei
"""

import os
import re
from glob import glob
from shutil import copyfile
copyfile('../../utils.py','utils.py')
from utils import create_highpass_filter_workflow
sub = 'sub-07'
data_dir = '../../../data/MRI/{}/func/*/*/outputs/func/ICA_AROMA'.format(sub)
ICAed_data = glob(os.path.join(
        data_dir,
        'denoised_func_data_nonaggr.nii.gz'))
HP_freq = 60
TR = 0.85
# pick one of the ICAed data:
file_name = '../../../data/MRI/sub-07/func/session-03/sub-07_unfeat_run-08/outputs/func/ICA_AROMA/denoised_func_data_nonaggr.nii.gz'

file_name
temp = re.findall('\d+',file_name)
n_session = temp[1]
n_run = temp[-1]
output_dir = os.path.join('/'.join(file_name.split('/')[:-2]),'ICAed_filtered')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
highpass_workflow = create_highpass_filter_workflow(HP_freq = HP_freq, 
                                                    TR = TR,
                            workflow_name = f"highpassfiler_{n_session}_{n_run}")
highpass_workflow.base_dir = 'hpf'
highpass_workflow.write_graph(dotfilename='session {} run {}.dot'.format(n_session,n_run))

highpass_workflow.inputs.inputspec.ICAed_file = os.path.abspath(file_name)
highpass_workflow.inputs.addmean.out_file = os.path.abspath(os.path.join(output_dir,
                                                                                 'filtered.nii.gz'))

highpass_workflow.run()
for log_file in glob(os.path.join(highpass_workflow.base_dir,"*","*","*","*","*",'report.rst')):
    log_name = log_file.split('/')[-5]
    copyfile(log_file,os.path.join(output_dir,'log_{}.rst'.format(log_name)))

