#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 07:01:54 2021

@author: nmei
"""

import os

import numpy as np
import seaborn as sns

from glob import glob
from tqdm import tqdm

from shutil import rmtree
from nipype.interfaces import fsl
from nilearn import image,plotting,input_data,datasets,surface
from matplotlib import pyplot as plt

sns.set_context('paper')

radius = 9
conscious_state = 'unconscious'
condict= dict(conscious=15,
              unconscious=14,)
average = True # true:= single map of average BOLD signal, false:= bootstrap mean
model_names = ['AlexNet', 'VGG19','MobileNetV2','ResNet50', 'DenseNet169',  ]
func_brain = '../../../../data/MRI/{sub}/func/example_func.nii.gz' # use .format to add subject info
standard_brain = '../../../../data/standard_brain/MNI152_T1_2mm_brain.nii.gz'
transformation_dir_single = '../../../../data/MRI/{sub}/reg/example_func2standard.mat'# use .format to add subject info
if average:
    working_dir = f'../../../../results/MRI/nilearn/RSA_searchlight_{radius}mm_average'
else:
    working_dir = f'/export/home/nmei/nmei/MRI/RSA_searchlight_{radius}mm_old/'
working_data = glob(os.path.join(working_dir,'*','*',f'{conscious_state}.nii.gz'))

standard_dir = f'../../../../results/MRI/nilearn/RSA_standard_{radius}mm'
#if os.path.exists(standard_dir):
#    rmtree(standard_dir)
figure_dir = f'../../../../figures/MRI/nilearn/RSA_searchlight_{radius}mm'
for f in [standard_dir,figure_dir]:
    if not os.path.exists(f):
        os.mkdir(f)
collect_dir = '/export/home/nmei/nmei/properties_of_unconscious_processing/all_figures'
# putting the ready-for-plot to a dictionary
res_for_plot = {f'sub-0{ii+1}':{model_name:None for model_name in model_names} for ii in range(7)}
iterator = tqdm(working_data)
for filename in iterator:
    temp = filename.split('/')
    subject = temp[-2]
    model_name = temp[-3]
    
    if average:
        to_filename = f'{conscious_state}_{subject}_{model_name}_average_individual_space.nii.gz'
        if not os.path.exists(os.path.join(standard_dir,to_filename)):
            brain_map = image.load_img(filename)
        
    else:
        to_filename = f'{conscious_state}_{subject}_{model_name}_bootstrap_individual_space.nii.gz'
        if not os.path.exists(os.path.join(standard_dir,to_filename)):
            masker = input_data.NiftiMasker().fit(func_brain.format(sub = subject))
            brain_map = masker.transform(filename)
            
            brain_map = brain_map.mean(0)
            brain_map = masker.inverse_transform(brain_map)
    
    if not os.path.exists(os.path.join(standard_dir,to_filename)):
        brain_map.to_filename(os.path.join(standard_dir,to_filename))
    
    iterator.set_description(f'reading individual space data, average = {average}')
    res_for_plot[subject][model_name] = os.path.join(standard_dir,to_filename)

vmax = 0.1
data = np.random.uniform(0,vmax,size = (30,30))
im = plt.imshow(data,cmap = plt.cm.Reds,vmin = 0,vmax = vmax)
subj_map = {'sub-01':'sub-01',
            'sub-03':'sub-02',
            'sub-04':'sub-03',
            'sub-05':'sub-04',
            'sub-02':'sub-05',
            'sub-06':'sub-06',
            'sub-07':'sub-07',}
bottom,top = 0.1,0.9
left,right = 0.1,0.8

# standardize
res_for_plot = {f'sub-0{ii+1}':{model_name:None for model_name in model_names} for ii in range(7)}
iterator = tqdm(working_data)
for filename in iterator:
    temp = filename.split('/')
    subject = temp[-2]
    model_name = temp[-3]
    
    if average:
        to_filename = f'{conscious_state}_{subject}_{model_name}_average_individual_space.nii.gz'
    else:
        to_filename = f'{conscious_state}_{subject}_{model_name}_bootstrap_individual_space.nii.gz'
    if not os.path.exists(os.path.join(standard_dir,to_filename.replace('individual','standard'))):
        flt = fsl.FLIRT()
        flt.inputs.in_file = os.path.abspath(os.path.join(standard_dir,to_filename))
        flt.inputs.reference = os.path.abspath(standard_brain)
        flt.inputs.output_type = 'NIFTI_GZ'
        flt.inputs.in_matrix_file = os.path.abspath(transformation_dir_single.format(sub = subject))
        flt.inputs.out_matrix_file = os.path.abspath(
                    os.path.join(standard_dir,to_filename.replace('_space.nii.gz','_flirt.mat')))
        flt.inputs.out_file = os.path.abspath(
                    os.path.join(standard_dir,to_filename.replace('individual','standard')))
        flt.inputs.apply_xfm = True
        res = flt.run()
        iterator.set_description(f'standardizing..., average = {average}')
    else:
        iterator.set_description(f'collecting..., average = {average}')
    res_for_plot[subject][model_name] = os.path.join(
            standard_dir,
            to_filename.replace('individual','standard'))

plt.close('all')
print('plotting in standard space')
fig,axes = plt.subplots(figsize = (len(model_names)*3,2*3),
                        nrows = 7,
                        ncols = len(model_names) * 2,
                        subplot_kw = {'projection':'3d'},
                        )
res_remap = {f'sub-0{ii+1}':{model_name:None for model_name in model_names} for ii in range(7)}
for original_sub,remap_sub in subj_map.items():
    for model_name in model_names:
        res_remap[remap_sub][model_name] = res_for_plot[original_sub][model_name]
fsaverage = datasets.fetch_surf_fsaverage()
for ii,row_axes in enumerate(axes):
    for jj,column_ax in enumerate(row_axes.reshape(-1,2)):
        brain_map = res_remap[f'sub-0{ii+1}'][model_names[jj]]
        masker = input_data.NiftiMasker().fit(standard_brain)
        brain_map_for_plot = masker.transform(brain_map)
        brain_map_for_plot[0 >= brain_map_for_plot] = 0
        brain_map_for_plot = masker.inverse_transform(brain_map_for_plot)
        
        ax_left = column_ax[0]#.inset_axes([0,0,0.5,0.5],subplot_kw = {'projection':'3d'},)
        ax_right = column_ax[1]#.inset_axes([0.5,0,0.5,0.5],subplot_kw = {'projection':'3d'},)
        for ax,surf_mesh,stat_mesh,bg_map,hemi,title in zip(
                                                  [ax_left,ax_right],
                                                  [fsaverage.infl_left,fsaverage.infl_right],
                                                  [fsaverage.pial_left,fsaverage.pial_right],
                                                  [fsaverage.sulc_left,fsaverage.sulc_right],
                                                  ['left','right'],
                                                  [f'sub-0{ii+1} | {model_names[jj]}',None],
                                                  ):
            brain_map_in_surf = surface.vol_to_surf(brain_map_for_plot,stat_mesh,radius = radius,)
            plotting.plot_surf_stat_map(surf_mesh,
                                        brain_map_in_surf,
                                        bg_map = bg_map,
                                        threshold = 1e-2,
                                        hemi = hemi,
                                        axes = ax,
                                        figure = fig,
                                        title = title,
                                        cmap = plt.cm.seismic,
                                        colorbar = False,
                                        vmax = vmax,
                                        symmetric_cbar = False,)
cbar_ax = fig.add_axes([0.92,bottom,0.01,top - bottom])
cbar = fig.colorbar(im,cax = cbar_ax)
cbar.set_ticks(np.array([0,vmax]))
cbar.set_ticklabels(np.array([0,vmax],dtype = str))

if average:
    fig.savefig(os.path.join(figure_dir,f'{conscious_state}_average_standard_space.jpg'),
                 dpi = 300,
                 bbox_inches = 'tight')
    fig.savefig(os.path.join(figure_dir,f'{conscious_state}_average_standard_space(light).jpg'),
                 dpi = 300,
                 bbox_inches = 'tight')
else:
    fig.savefig(os.path.join(figure_dir,f'{conscious_state}_bootstrap_standard_space.jpg'),
                dpi = 300,
                bbox_inches = 'tight')
    fig.savefig(os.path.join(figure_dir,f'{conscious_state}_bootstrap_standard_space(light).jpg'),
                dpi = 300,
                 bbox_inches = 'tight')
#fig.savefig(os.path.join(collect_dir,f'supfigure{condict[conscious_state]}.eps'),
#            dpi = 300,
#            bbox_inches = 'tight')
fig.savefig(os.path.join(collect_dir,f'supfigure{condict[conscious_state]}.png'),
            bbox_inches = 'tight')
plt.close('all')

"""
plt.close('all')
print('plotting individual space brain maps')
fig,axes = plt.subplots(figsize = (7*7,5*7),
                        nrows = 7,
                        ncols = 5,
                        )
res_remap = {f'sub-0{ii+1}':{model_name:None for model_name in model_names} for ii in range(7)}
for original_sub,remap_sub in subj_map.items():
    for model_name in model_names:
        res_remap[remap_sub][model_name] = res_for_plot[original_sub][model_name]
for ii,row_axes in enumerate(axes):
    for jj,column_ax in enumerate(row_axes):
        sub = res_remap[f'sub-0{ii+1}'][model_names[jj]].split('/')[-1].split('_')[1]
        plotting.plot_stat_map(res_remap[f'sub-0{ii+1}'][model_names[jj]],
                               func_brain.format(sub = sub),
                               cut_coords = (0,0,0),
                               draw_cross = False,
                               threshold = 1e-3,
                               axes = column_ax,
                               figure = fig,
                               title = f'sub-0{ii+1},{model_names[jj]}',
                               cmap = plt.cm.coolwarm,
                               colorbar = False,
                               vmax = vmax,
                               )
cbar_ax = fig.add_axes([0.92,bottom,0.01,top - bottom])
cbar = fig.colorbar(im,cax = cbar_ax)
cbar.set_ticks(np.array([-vmax,0,vmax]))
cbar.set_ticklabels(np.array([-vmax,0,vmax],dtype = str))

if average:
    fig.savefig(os.path.join(figure_dir,f'{conscious_state}_average_individual_space.jpg'),
                dpi = 300,
                bbox_inches = 'tight')
    fig.savefig(os.path.join(figure_dir,f'{conscious_state}_average_individual_space(light).jpg'),
#                dpi = 300,
                bbox_inches = 'tight')
else:
    fig.savefig(os.path.join(figure_dir,f'{conscious_state}_bootstrap_individual_space.jpg'),
                dpi = 300,
                bbox_inches = 'tight')
    fig.savefig(os.path.join(figure_dir,f'{conscious_state}_bootstrap_individual_space(light).jpg'),
#                dpi = 300,
                bbox_inches = 'tight')
"""

















