#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 12:21:33 2019

@author: nmei
"""

import os
from glob import glob
import re

sub = 'sub-07'
data_dir = '../../../data/MRI/{}/func/*/*/outputs/func/ICA_AROMA'.format(sub)
ICAed_data = glob(os.path.join(
        data_dir,
        'denoised_func_data_nonaggr.nii.gz'))

# make python script 0.preprocessing fmri.py for each run
[os.remove(item) for item in glob('4.*.*.py') if (item != "4.highpass filter.py")]
[os.remove(item) for item in glob('highpass_filter_*')]
for ii, func_data_file in enumerate(ICAed_data):
    temp = re.findall('\d+',func_data_file)
    n_session = int(temp[1])
    n_run = int(temp[-1])
    print(n_session,n_run)
    with open(f'4.{ii+1}.highpass_filter_session_{n_session}_run_{n_run}.py','w') as new_file:
        with open('4.highpass filter.py','r') as old_file:
            for line in old_file:
                new_file.write(line.replace("file_name = ''",f"file_name = '{func_data_file}'"))
        old_file.close()
    new_file.close()

if not os.path.exists('hpf'):
    os.mkdir('hpf')
else:
    [os.remove('hpf/'+f) for f in os.listdir('hpf') if ('.txt' in f)]

for ii, func_data_file in enumerate(ICAed_data):
    temp = re.findall('\d+',func_data_file)
    n_session = int(temp[1])
    n_run = int(temp[-1])
    print(ii,n_session,n_run)
    content = f"""
#!/bin/bash

# This is a script to send "4.{ii+1}.highpass_filter_session_{n_session}_run_{n_run}.py" as a batch job.
# it works on dataset {ii + 1}

#$ -cwd
#$ -o hpf/out_{n_session}{n_run}.txt
#$ -e hpf/err_{n_session}{n_run}.txt
#$ -m be
#$ -q fsl.q
#$ -M nmei@bcbl.eu
#$ -N "hpf{n_session}{n_run}"
#$ -S /bin/bash

module load rocks-python-3.6 rocks-fsl-5.0.10 rocks-freesurfer-6.0.0
python "4.{ii+1}.highpass_filter_session_{n_session}_run_{n_run}.py"
"""
    print(content)
    with open('highpass_filter_{}'.format(ii + 1),'w') as f:
        f.write(content)


content = '''
import os
import time
'''
with open('qsub_jobs_highpass_filter.py','w') as f:
    f.write(content)
    f.close()

with open('qsub_jobs_highpass_filter.py','a') as f:
    for ii, func_data_file in enumerate(ICAed_data):
        if ii == 0:
            f.write('\nos.system("qsub highpass_filter_{}")\n'.format(ii+1))
        else:
            f.write('time.sleep(30)\nos.system("qsub highpass_filter_{}")\n'.format(ii+1))
    f.close()
content = '''
#!/bin/bash

# This is a script to send qsub_jobs_highpass_filter.py as a batch job.

#$ -cwd
#$ -o hpf/out_q.txt
#$ -e hpf/err_q.txt
#$ -m be
#$ -M nmei@bcbl.eu
#$ -N "qsubjobs"
#$ -S /bin/bash

module load rocks-python-3.6
python "qsub_jobs_highpass_filter.py"
'''
with open('qsub_jobs_highpass_filter','w') as f:
    f.write(content)
    f.close()





















