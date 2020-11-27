#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 18:10:27 2019

@author: nmei

create jobs for single functional data, doing ICA AROMA

This script will also submit the parallel jobs in the end

too bad that ICA AROMA can only compile with python2.7

"""

import os
from glob import glob
from nipype.interfaces.fsl import ICA_AROMA
import re
from shutil import rmtree

sub                     = 'sub-07'
MRI_dir                 = '../../../data/MRI/{}/func'.format(sub)
parent_dir              = os.path.join(MRI_dir,'session-*','*','outputs',)
preprocessed_BOLD_files = glob(os.path.join(parent_dir,
                                            'func',
                                            'prefiltered_func.nii.gz'))
first_session           = 1
first_run_dir           = os.path.join(MRI_dir,
                                       'session-0{}'.format(first_session),
                                       '{}_unfeat_run-01'.format(sub),)
first_run_dir           = os.path.abspath(first_run_dir)
to_work_dir             = 'ICA_AROMA_parallel'

if not os.path.exists(to_work_dir):
    os.mkdir(to_work_dir)
else:
    rmtree(to_work_dir)
    os.mkdir(to_work_dir)

#[os.remove(f) for f in os.listdir(to_work_dir)]
template = """
#!/bin/bash
#$ -cwd
#$ -o out_{}.txt
#$ -e err_{}.txt
#$ -m be
#$ -q fsl.q
#$ -M nmei@bcbl.eu
#$ -N "s{}r{}"
#$ -S /bin/bash

module load rocks-fsl-5.0.10
module load rocks-python-2.7
{}
"""
for sample in preprocessed_BOLD_files:
    AROMA_obj           = ICA_AROMA()
    parent_dir          = '/'.join(sample.split('/')[:-2])
    temp                = re.findall('\d+',sample)
    n_session           = int(temp[1])
    n_run               = int(temp[-1])
    print(n_session,n_run)
    func_to_struct      = os.path.join(first_run_dir,
                                       'outputs',
                                       'reg',
                                       'example_func2highres.mat')
    warpfield           = os.path.join(first_run_dir,
                                       'outputs',
                                       'reg',
                                       'highres2standard_warp.nii.gz')
    fsl_mcflirt_movpar  = os.path.join(parent_dir,
                                       'func',
                                       'MC',
                                       'MCflirt.par')
    mask                = os.path.join(parent_dir,'func','mask.nii.gz')
    output_dir          = os.path.join(parent_dir,'func','ICA_AROMA')
    AROMA_obj.inputs.in_file            = os.path.abspath(sample)
    AROMA_obj.inputs.mat_file           = os.path.abspath(func_to_struct)
    AROMA_obj.inputs.fnirt_warp_file    = os.path.abspath(warpfield)
    AROMA_obj.inputs.motion_parameters  = os.path.abspath(fsl_mcflirt_movpar)
    AROMA_obj.inputs.mask               = os.path.abspath(mask)
    AROMA_obj.inputs.denoise_type       = 'nonaggr'
    AROMA_obj.inputs.out_dir            = os.path.abspath(output_dir)
    cmdline             = 'python ../../' + AROMA_obj.cmdline + ' -ow'
    qsub                = template.format(10*n_session+n_run,
                                          10*n_session+n_run,
                                          n_session,
                                          n_run,
                                          cmdline)
    with open(os.path.join(to_work_dir,'session{}.run{}_qs'.format(n_session,n_run)),'w') as f:
        f.write(qsub)

to_qsub = """
import os
from time import sleep
"""
with open('{}/ICA_qsub_jobs.py'.format(to_work_dir),'w') as f:
    f.write(to_qsub)
    f.close()
for sample in preprocessed_BOLD_files:
    temp                = re.findall('\d+',sample)
    n_session           = int(temp[1])
    n_run               = int(temp[-1])
    with open('{}/ICA_qsub_jobs.py'.format(to_work_dir),'a') as f:
        f.write('\nos.system("qsub session{}.run{}_qs")\nsleep(60)\n'.format(n_session,n_run))
        f.close()

qsub_qsub_jobs = """
#!/bin/bash
#$ -cwd
#$ -o out_q.txt
#$ -e err_q.txt
#$ -m be
#$ -q fsl.q
#$ -M nmei@bcbl.eu
#$ -N "qjobs"
#$ -S /bin/bash

module load rocks-python-2.7
python ICA_qsub_jobs.py
"""
with open('{}/qsub_ICA_qsub_jobs'.format(to_work_dir), 'w') as f:
    f.write(qsub_qsub_jobs)
    f.close()

for f in os.listdir(to_work_dir):
    if 'txt' in f:
        os.remove(os.path.join(to_work_dir,f))

os.system("cd {}; qsub qsub_ICA_qsub_jobs".format(to_work_dir))
