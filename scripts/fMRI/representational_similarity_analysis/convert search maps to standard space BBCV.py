#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 17:34:59 2022

@author: nmei
"""

import os,gc
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

from nipype.interfaces import fsl
from nilearn import image,plotting,datasets,surface
from joblib import Parallel,delayed

from matplotlib import pyplot as plt

def convert(working_data,idx,stats = 'RSA'):
    filename            = working_data[idx]
    temp                = filename.split('/')[-1]
    temp                = temp.split('_')
    (sub,
     conscious_source,
     conscious_target,
     model_name,
     radius,
     )                 = temp[:5]
    whole_brain_mask    = mask_dir.format(sub)
    example_func        = functional_brain.format(sub)
    input_file          = os.path.abspath(filename)
    mask_file           = os.path.abspath(whole_brain_mask)
    standard_brain = '../../../../data/standard_brain/MNI152_T1_2mm_brain.nii.gz'
    transformation_dir_single = os.path.abspath(
                    f'../../../../data/MRI/{sub}/reg/example_func2standard.mat')
    base_name           = os.path.abspath(filename.split('.nii')[0])
    
    flt = fsl.FLIRT()
    flt.inputs.in_file = os.path.abspath(filename)
    flt.inputs.reference = os.path.abspath(standard_brain)
    flt.inputs.output_type = 'NIFTI_GZ'
    flt.inputs.in_matrix_file = transformation_dir_single
    flt.inputs.out_matrix_file = os.path.abspath(
        os.path.join(standarded_dir,f'{sub}_{conscious_source}_{conscious_target}_{model_name}_flirt.mat'))
    flt.inputs.out_file = os.path.abspath(
        os.path.join(standarded_dir,base_name.split('/')[-1] + '.nii.gz')
        )
    flt.inputs.apply_xfm = True
    res = flt.run()
    
    return res

def plotting_img(fig,row,radius,stats = 'RSA'):
    if stats == 'RSA':
        brain_in_mesh = surface.vol_to_surf(image.mean_img(row['brain_in_standard']),
                                            row['stat_mesh'],
                                            radius = radius,)
    else:
        brain_in_mesh = surface.vol_to_surf(row['brain_in_standard'],
                                            row['stat_mesh'],
                                            radius = radius,)
    plotting.plot_surf_stat_map(row['surf_mesh'],
                                brain_in_mesh,
                                bg_map = row['bg_map'],
                                threshold = 1e-3,
                                hemi = row['hemi'],
                                title = row['title'],
                                cmap = plt.cm.seismic,
                                colorbar = False,
                                vmax = .15,
                                symmetric_cbar = False,
                                axes = row['ax'],
                                figure = fig,
                                )
if __name__ == "__main__":
    [os.remove(f) for f in glob('core*')]
    radius              = 6
    stats               = 'RSA'
    # stats               = 'tfce_corrp_tstat1'
    vmax                = 0.1 if stats == 'RSA' else None
    data                = np.random.uniform(-vmax,vmax,size = (30,30)) if stats == 'RSA' else np.random.rand(30,30)
    im                  = plt.imshow(data,cmap = plt.cm.seismic,vmin = -vmax,vmax = vmax) if stats == 'RSA' else plt.imshow(data,cmap = plt.cm.Reds)
    subj_map            = {'sub-01':'sub-01',
                           'sub-03':'sub-02',
                           'sub-04':'sub-03',
                           'sub-05':'sub-04',
                           'sub-02':'sub-05',
                           'sub-06':'sub-06',
                           'sub-07':'sub-07',
                           }
    bottom,top          = 0.1,0.9
    left,right          = 0.1,0.8
    #'AlexNet', 'VGG19','MobileNetV2','ResNet50', 'DenseNet169'
    model_name          = 'DenseNet169'
    figname_map         = {'RSA':'RSA maps',
                           'tfce_corrp_tstat1':'p value maps',
                           }
    threshold           = 5e-3 if stats == 'RSA' else -np.log(0.05)
    cmap                = plt.cm.seismic if stats == 'RSA' else plt.cm.Reds
    folder_name         = f'RSA_searchlight_long_process_{radius}mm'
    working_dir         = '../../../../results/MRI/nilearn'
    working_data        = np.sort(glob(os.path.join(working_dir,
                                                    folder_name,
                                                    "*",
                                                    "*",
                                                    f"*{model_name}*{stats}.nii.gz")))
    mask_dir            = '../../../../data/MRI/{}/anat/ROI_BOLD/combine_BOLD.nii.gz'
    functional_brain    = '../../../../data/MRI/{}/func/example_func.nii.gz'
    standarded_dir      = '../../../../results/MRI/nilearn/RSA_standard'
    if not os.path.exists(standarded_dir):
        os.mkdir(standarded_dir)
    
#    standarded = glob(os.path.join(standarded_dir,f'*{model_name}*6mm.nii.gz')) if stats == 'RSA' else glob(os.path.join(standarded_dir,f'*{model_name}*tfce*.nii.gz'))
#    [os.remove(f) for f in standarded]
    
    gc.collect()
    temp                = Parallel(n_jobs = -1,verbose = 1)(delayed(convert)(**{
                                        'working_data':working_data,
                                        'idx':idx,
                                        'stats':stats,}) for idx,f in enumerate(working_data))
    gc.collect()
    # more mat files
    [os.remove(f) for f in glob(os.path.join(standarded_dir,'*.mat'))]
    
    # plot
    figure_dir          = '../../../../figures/MRI/nilearn/encoding_based_RSA_{}mm'.format(radius)
    if not os.path.exists(figure_dir):
        os.mkdir(figure_dir)
    fsaverage           = datasets.fetch_surf_fsaverage()
    
    
    standarded = glob(os.path.join(standarded_dir,f'*{model_name}*6mm_RSA.nii.gz')) if stats == 'RSA' else glob(os.path.join(standarded_dir,f'*{model_name}*tfce*.nii.gz'))
    brain_in_standard = np.sort([f for f in standarded]).reshape(-1,3)
    print(brain_in_standard.shape)
    brain_in_standard = brain_in_standard[[0,2,3,4,1,5,6]]
    
    fig,axes = plt.subplots(figsize = (4*3,3*3),
                        nrows = 7,
                        ncols = 3 * 2,
                        subplot_kw = {'projection':'3d'},
                        )
    
    df_plot = {'ax':[],
               'surf_mesh':[],
               'stat_mesh':[],
               'bg_map':[],
               'hemi':[],
               'title':[],
               'brain_in_standard':[],
               }
    for ii,(brain_maps,ax) in enumerate(zip(brain_in_standard,axes)):
        subj = f'sub-0{ii+1}'
        
        ## unconscious -> unconscious
        df_plot['ax'].append(ax[0])
        df_plot['surf_mesh'].append(fsaverage.infl_left)
        df_plot['stat_mesh'].append(fsaverage.pial_left)
        df_plot['bg_map'].append(fsaverage.sulc_left)
        df_plot['hemi'].append('left')
        df_plot['title'].append(f'{subj} | unconscious -> unconscious')
        df_plot['brain_in_standard'].append(brain_maps[2])
        
        df_plot['ax'].append(ax[1])
        df_plot['surf_mesh'].append(fsaverage.infl_right)
        df_plot['stat_mesh'].append(fsaverage.pial_right)
        df_plot['bg_map'].append(fsaverage.sulc_right)
        df_plot['hemi'].append('right')
        df_plot['title'].append(None)
        df_plot['brain_in_standard'].append(brain_maps[2])
        
        ## conscious -> conscious
        df_plot['ax'].append(ax[2])
        df_plot['surf_mesh'].append(fsaverage.infl_left)
        df_plot['stat_mesh'].append(fsaverage.pial_left)
        df_plot['bg_map'].append(fsaverage.sulc_left)
        df_plot['hemi'].append('left')
        df_plot['title'].append('conscious -> conscious')
        df_plot['brain_in_standard'].append(brain_maps[0])
        
        df_plot['ax'].append(ax[3])
        df_plot['surf_mesh'].append(fsaverage.infl_right)
        df_plot['stat_mesh'].append(fsaverage.pial_right)
        df_plot['bg_map'].append(fsaverage.sulc_right)
        df_plot['hemi'].append('right')
        df_plot['title'].append(None)
        df_plot['brain_in_standard'].append(brain_maps[0])
        
        ## conscious -> unconscious
        df_plot['ax'].append(ax[4])
        df_plot['surf_mesh'].append(fsaverage.infl_left)
        df_plot['stat_mesh'].append(fsaverage.pial_left)
        df_plot['bg_map'].append(fsaverage.sulc_left)
        df_plot['hemi'].append('left')
        df_plot['title'].append('conscious -> unconscious')
        df_plot['brain_in_standard'].append(brain_maps[1])
        
        df_plot['ax'].append(ax[5])
        df_plot['surf_mesh'].append(fsaverage.infl_right)
        df_plot['stat_mesh'].append(fsaverage.pial_right)
        df_plot['bg_map'].append(fsaverage.sulc_right)
        df_plot['hemi'].append('right')
        df_plot['title'].append(None)
        df_plot['brain_in_standard'].append(brain_maps[1])
    df_plot = pd.DataFrame(df_plot)
    df_plot['sub'] = df_plot['brain_in_standard'].apply(lambda x:x.split('/')[-1].split('_')[0]).map(subj_map)
    df_plot['conscious_source'] = df_plot['brain_in_standard'].apply(lambda x:x.split('/')[-1].split('_')[1])
    df_plot['conscious_target'] = df_plot['brain_in_standard'].apply(lambda x:x.split('/')[-1].split('_')[2])
#    df_plot = df_plot.sort_values(['sub','conscious_source','conscious_target']).reset_index()
#    
#    df_u_u = df_plot[np.logical_and(df_plot['conscious_source'] == 'unconscious',
#                                    df_plot['conscious_target'] == 'unconscious')]
#    df_c_c = df_plot[np.logical_and(df_plot['conscious_source'] == 'conscious',
#                                    df_plot['conscious_target'] == 'conscious')]
#    df_c_u = df_plot[np.logical_and(df_plot['conscious_source'] == 'conscious',
#                                    df_plot['conscious_target'] == 'unconscious')]
#    df_plot = pd.concat([df_u_u,df_c_c,df_c_u])
#    df_plot = df_plot.sort_values(['sub'])
    
    for ii,row in tqdm(df_plot.iterrows()):
        if stats == 'RSA':
            brain_in_mesh = surface.vol_to_surf(image.mean_img(row['brain_in_standard']),
                                                row['stat_mesh'],
                                                radius = radius,)
        else:
            temp = row['brain_in_standard']
            pval = image.math_img('-np.log(1 - img + 1e-32)',img = temp)
            brain_in_mesh = surface.vol_to_surf(pval,
                                                row['stat_mesh'],
                                                radius = radius,
                                                )
            # calculate threshold in mesh space
            
        plotting.plot_surf_stat_map(row['surf_mesh'],
                                    brain_in_mesh,
                                    bg_map = row['bg_map'],
                                    threshold = threshold,
                                    hemi = row['hemi'],
                                    title = row['title'],
                                    cmap = cmap,
                                    colorbar = False,
                                    vmax = vmax,
                                    symmetric_cbar = False,
                                    axes = row['ax'],
                                    figure = fig,
                                    )
    if stats == 'RSA':
        cbar_ax = fig.add_axes([0.92,bottom,0.01,top - bottom])
        cbar = fig.colorbar(im,cax = cbar_ax)
        cbar.set_ticks(np.array([-vmax,0,vmax]))
        cbar.set_ticklabels(np.array([-vmax,0,vmax],dtype = str))
    fig.savefig(os.path.join(figure_dir,f'{model_name} {figname_map[stats]}.jpg'),
                dpi = 300,
                bbox_inches = 'tight')
    plt.close('all')