#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 23:23:52 2019

@author: nmei
"""

import os
import pandas as pd
from glob import glob
from shutil import copyfile
copyfile('../../utils.py','utils.py')
from utils import create_simple_struc2BOLD
from nipype.interfaces import freesurfer,fsl
fsl.FSLCommand.set_default_output_type('NIFTI_GZ')

freesurfer_list = pd.read_csv('../../../FreesurferLTU.csv')

sub = 'sub-07'
first_session = 1
os.environ['SUBJECTS_DIR'] = os.path.abspath('../../../data/MRI/{}/'.format(sub))
in_file = os.path.abspath('../../../data/MRI/{}/anat/{}/mri/aparc+aseg.mgz'.format(sub,sub))
original = os.path.abspath('../../../data/MRI/{}/anat/{}/mri/orig/001.mgz'.format(sub,sub))
ROI_anat_dir = '../../../data/MRI/{}/anat/ROIs_anat'.format(sub)
if not os.path.exists(ROI_anat_dir):
    os.mkdir(ROI_anat_dir)

# superiorparietal pericalcarine rostralmiddlefrontal
roi_names = """fusiform
inferiorparietal
inferiortemporal
lateraloccipital
lingual
middlefrontal
parahippocampal
pericalcarine
precuneus
superiorfrontal
superiorparietal
parsorbitalis
parstriangularis
parsopercularis"""
idx_label,label_names = [],[]
for name in roi_names.split('\n'):
    for label_name in freesurfer_list['Label Name']:
        if (name in label_name) and ('ctx' in label_name):
            idx = freesurfer_list[freesurfer_list['Label Name'] == label_name]['#No.'].values[0]
            if (str(idx)[1] == '0') and (idx < 3000) and (name in label_name) and ('caudal' not in label_name):
                idx_label.append(idx)
                label_names.append(label_name)

for idx,label_name in zip(idx_label,label_names):
    binary_file = os.path.abspath(os.path.join(ROI_anat_dir,'{}.nii.gz'.format(label_name)))
    binarizer = freesurfer.Binarize(in_file = in_file,
                                    match = [idx],
                                    binary_file = binary_file)
#    print(binarizer.cmdline)
    binarizer.run()
    fsl_swapdim = fsl.SwapDimensions(new_dims = ('x', 'z', '-y'),)
    fsl_swapdim.inputs.in_file = binarizer.inputs.binary_file
    fsl_swapdim.inputs.out_file = binarizer.inputs.binary_file.replace('.nii.gz','_fsl.nii.gz')
#    print(fsl_swapdim.cmdline)
    fsl_swapdim.run()
    mc = freesurfer.MRIConvert()
    mc.inputs.in_file = fsl_swapdim.inputs.out_file
    mc.inputs.reslice_like = original
    mc.inputs.out_file = fsl_swapdim.inputs.out_file
#    print(mc.cmdline)
    mc.run()


anat_dir = '../../../data/MRI/{}/anat'.format(sub)
ROI_in_structural = glob(os.path.join(anat_dir,'ROIs_anat','*fsl.nii.gz'))


preprocessed_functional_dir = '../../../data/MRI/{}/func/session-0{}/{}_unfeat_run-01/outputs'.format(
        sub,first_session,sub)

output_dir = os.path.join(anat_dir,'ROI_BOLD')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

for roi in ROI_in_structural:
    roi = os.path.abspath(roi)
    roi_name = roi.split('/')[-1]
    simple_workflow = create_simple_struc2BOLD(roi = roi,
                                               roi_name = roi_name,
                                               preprocessed_functional_dir = preprocessed_functional_dir,
                                               output_dir = output_dir)
    simple_workflow.base_dir = os.path.abspath(output_dir)
    simple_workflow.write_graph(dotfilename='{}.dot'.format(roi_name.split('.')[0]))
    simple_workflow.run()


# combine rois: parsorbitalis parstriangularis parsopercularis
masks_BOLD_to_combine = glob(os.path.join(output_dir,'*pars*.nii.gz'))
for direction in ['lh','rh']:
    parts = [os.path.abspath(item) for item in masks_BOLD_to_combine if (direction in item)]
    merger = fsl.ImageMaths(in_file = parts[0],
                            in_file2 = parts[1],
                            op_string = '-add')
    merger.inputs.out_file = os.path.abspath(
                            os.path.join(output_dir,
                            'ctx-{}-{}_BOLD.nii.gz'.format(direction,
                                 'ventrolateralPFC')))
    merger.run()
    
    merger2 = fsl.ImageMaths(in_file = merger.inputs.out_file,
                            in_file2 = parts[2],
                            op_string = '-add')
    merger2.inputs.out_file = os.path.abspath(
                            os.path.join(output_dir,
                            'ctx-{}-{}_BOLD.nii.gz'.format(direction,
                                 'ventrolateralPFC')))
    merger2.run()
    binarize = fsl.ImageMaths(op_string = '-bin')
    binarize.inputs.in_file = merger2.inputs.out_file
    binarize.inputs.out_file = os.path.abspath(
                            os.path.join(output_dir,
                            'ctx-{}-{}_BOLD.nii.gz'.format(direction,
                                 'ventrolateralPFC')))
    binarize.run()
achived = os.path.join(output_dir,'achieved')
if not os.path.exists(achived):
    os.mkdir(achived)
for part in masks_BOLD_to_combine:
    copyfile(part,os.path.join(achived,part.split('/')[-1]))
    os.remove(part)











