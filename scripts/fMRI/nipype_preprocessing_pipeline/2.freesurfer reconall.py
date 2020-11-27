#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 23:21:50 2019

@author: nmei
"""

from nipype.interfaces.freesurfer import ReconAll
import os
from glob import glob

sub = 'sub-07'
working_dir = '../../../data/MRI/{}/anat'.format(sub)
reconall = ReconAll()
reconall.inputs.subject_id = sub
reconall.inputs.directive = 'all'
reconall.inputs.subjects_dir = os.path.abspath(working_dir)
reconall.inputs.T1_files = os.path.abspath([item for item in glob(os.path.join(working_dir,
                   '*t1*.nii*')) if ('brain' not in item)][0])
reconall.cmdline
reconall.run()