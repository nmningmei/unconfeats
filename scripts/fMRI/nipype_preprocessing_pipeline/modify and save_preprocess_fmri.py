#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 10:48:54 2019

@author: nmei
"""

import os
import re
from glob import glob
import numpy as np

sub             = 'sub-07' #specify subject name/code
first_session   = '01' # which session is the very first functional session, to which we align the rest
func_dir        = '../../../data/MRI/{}/func/'.format(sub) # define the parent directory of the functional data

# because for one subject we remove the first session, thus, there is "wrong" in the name of that session
# which is excluded
func_data       = [item for item in glob(os.path.join(func_dir,
                                                      '*',
                                                      '*',
                                                      '*.nii*')) if ('wrong' not in item)]
func_data       = np.sort(func_data)

# make python script 0.preprocessing fmri.py for each run
[os.remove(item) for item in glob('0.1.*.py')]
[os.remove(item) for item in glob('fmri_prep_*')]

for ii, func_data_file in enumerate(func_data):
    temp = re.findall('\d+',func_data_file)
    n_session = int(temp[1])
    n_run = int(temp[-1])
    print(n_session,n_run)
    with open(f'0.1.preprocess fmri_session_{n_session}_run_{n_run}.py','w') as new_file:
        with open('0.preprocess fmri.py','r') as old_file:
            for line in old_file:
                new_file.write(line.replace("func_data_file = '../../../data/MRI/sub-01/func/session-02/sub-01_unfeat_run-01/sub-01_unfeat_run-01_bold.nii'",f"func_data_file = '{func_data_file}'"))
        old_file.close()
    new_file.close()

if not os.path.exists('fmri_prep'):
    os.mkdir('fmri_prep')
else:
    [os.remove('fmri_prep/'+f) for f in os.listdir('fmri_prep')]

for ii, func_data_file in enumerate(func_data):
    temp = re.findall('\d+',func_data_file)
    n_session = int(temp[1])
    n_run = int(temp[-1])
#    print(ii,n_session,n_run)
    content = f"""
#!/bin/bash

# This is a script to send "0.1.preprocess fmri_session_{n_session}_run_{n_run}.py" as a batch job.
# it works on dataset {ii + 1}

#$ -cwd
#$ -o fmri_prep/out_{n_session}{n_run}.txt
#$ -e fmri_prep/err_{n_session}{n_run}.txt
#$ -m be
#$ -q fsl.q
#$ -M nmei@bcbl.eu
#$ -N "fprep{n_session}{n_run}"
#$ -S /bin/bash

module load rocks-python-3.6 rocks-fsl-5.0.10 rocks-freesurfer-6.0.0
python "0.1.preprocess fmri_session_{n_session}_run_{n_run}.py"
"""
#    print(content)
    with open('fmri_prep_{}'.format(ii + 1),'w') as f:
        f.write(content)

content = '''
import os
import time
'''
with open('qsub_jobs_fmri_prep.py','w') as f:
    f.write(content)
    f.close()

with open('qsub_jobs_fmri_prep.py','a') as f:
    for ii, func_data_file in enumerate(func_data):
        if ii == 0:
            f.write('\nos.system("qsub fmri_prep_{}")\ntime.sleep(10 * 60)\n'.format(ii+1))
        else:
            f.write('time.sleep(30)\nos.system("qsub fmri_prep_{}")\n'.format(ii+1))
    f.close()
content = '''
#!/bin/bash

# This is a script to send qsub_jobs_fmri_prep.py as a batch job.

#$ -cwd
#$ -o fmri_prep/out_q.txt
#$ -e fmri_prep/err_q.txt
#$ -m be
#$ -M nmei@bcbl.eu
#$ -N "qsubjobs"
#$ -S /bin/bash

module load rocks-python-3.6
python "qsub_jobs_fmri_prep.py"
'''
with open('qsub_jobs_fmri_prep','w') as f:
    f.write(content)
    f.close()

