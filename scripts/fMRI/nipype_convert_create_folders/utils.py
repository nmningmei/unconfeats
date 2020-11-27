#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 10:09:21 2019

@author: nmei
"""

from autoreject import (AutoReject,get_rejection_threshold)
import mne
from glob import glob
import re
import os

import numpy as np
import pandas as pd
import pickle

from sklearn.metrics                               import roc_auc_score,roc_curve
from sklearn.metrics                               import (
                                                           classification_report,
                                                           matthews_corrcoef,
                                                           confusion_matrix,
                                                           f1_score,
                                                           log_loss,
                                                           r2_score
                                                           )

from sklearn.preprocessing                         import (MinMaxScaler,
                                                           OneHotEncoder,
                                                           FunctionTransformer,
                                                           StandardScaler)

from sklearn.pipeline                              import make_pipeline
from sklearn.ensemble.forest                       import _generate_unsampled_indices
from sklearn.utils                                 import shuffle
from sklearn.svm                                   import SVC,LinearSVC
from sklearn.calibration                           import CalibratedClassifierCV
from sklearn.decomposition                         import PCA
from sklearn.dummy                                 import DummyClassifier
from sklearn.feature_selection                     import (SelectFromModel,
                                                           SelectPercentile,
                                                           VarianceThreshold,
                                                           mutual_info_classif,
                                                           f_classif,
                                                           chi2,
                                                           f_regression,
                                                           GenericUnivariateSelect)
from sklearn.model_selection                       import (StratifiedShuffleSplit,
                                                           cross_val_score)
from sklearn.ensemble                              import RandomForestClassifier,BaggingClassifier,VotingClassifier
from sklearn.neural_network                        import MLPClassifier
from xgboost                                       import XGBClassifier
from itertools                                     import product,combinations
from sklearn.base                                  import clone
from sklearn.neighbors                             import KNeighborsClassifier
from sklearn.tree                                  import DecisionTreeClassifier
from collections                                   import OrderedDict

from scipy                                         import stats
from collections                                   import Counter

import matplotlib.pyplot  as plt
import matplotlib.patches as patches

try:
    #from mvpa2.datasets.base                           import Dataset
    from mvpa2.mappers.fx                              import mean_group_sample
    #from mvpa2.measures                                import rsa
    #from mvpa2.measures.searchlight                    import sphere_searchlight
    #from mvpa2.base.learner                            import ChainLearner
    #from mvpa2.mappers.shape                           import TransposeMapper
    #from mvpa2.generators.partition                    import NFoldPartitioner
except:
    pass#print('pymvpa is not installed')
try:
#    from tqdm import tqdm_notebook as tqdm
    from tqdm.auto import tqdm
except:
    print('why is tqdm not installed?')

def preprocessing_conscious(
                  raw,events,session,
                  n_interpolates = np.arange(1,32,4),
                  consensus_pers = np.linspace(0,1.0,11),
                  event_id = {'living':1,'nonliving':2},
                  tmin = -0.15,
                  tmax = 0.15 * 6,
                  high_pass = 0.001,
                  low_pass = 30,
                  notch_filter = 50,
                  fix = False,
                  ICA = False,
                  logging = None):
    """
    Preprocessing pipeline for conscious trials
    
    Inputs
    -------------------
    raw: MNE Raw object, contineous EEG raw data
    events: Numpy array with 3 columns, where the first column indicates time and the last column indicates event code
    n_interpolates: list of values 1 <= N <= max number of channels
    consensus_pers: ?? autoreject hyperparameter search grid
    event_id: MNE argument, to control for epochs
    tmin: first time stamp of the epoch
    tmax: last time stamp of the epoch
    high_pass: low cutoff of the bandpass filter
    low_pass: high cutoff of the bandpass filter
    notch_filter: frequency of the notch filter, 60 in US and 50 in Europe
    fix : when "True", apply autoReject algorithm to remove artifacts that was not identifed in the ICA procedure
    Output
    ICA : when "True", apply ICA artifact correction in ICA space
    logging: when not "None", output some log files for us to track the process
    -------------------
    Epochs: MNE Epochs object, segmented and cleaned EEG data (n_trials x n_channels x n_times)
    """
    
    """
    0. re-reference - explicitly
    """
    raw_ref ,_  = mne.set_eeg_reference(raw,
                                       ref_channels     = 'average',
                                       projection       = True,)
    raw_ref.apply_proj() # it might tell you it already has been re-referenced, but do it anyway
    """
    1. highpass filter 
        by a 4th order zero-phase Butterworth filter
    """
    # everytime before filtering, explicitly pick the type of channels you want
    # to perform the filters
    picks = mne.pick_types(raw_ref.info,
                           meg = False, # No MEG
                           eeg = True,  # YES EEG
                           eog = True,  # YES EOG
                           )
    # regardless the bandpass filtering later, we should always filter
    # for wire artifacts and their oscillations
    raw_ref.notch_filter(np.arange(notch_filter,241,notch_filter),
                         picks = picks)
    # high pass filtering
    picks = mne.pick_types(raw_ref.info,
                           meg = False, # No MEG
                           eeg = True,  # YES EEG
                           eog = False, # No EOG
                           )
    raw_ref.filter(high_pass,
                   None,
                   picks            = picks,
                   filter_length    = 'auto',    # the filter length is chosen based on the size of the transition regions (6.6 times the reciprocal of the shortest transition band for fir_window=’hamming’ and fir_design=”firwin2”, and half that for “firwin”)
                   l_trans_bandwidth= high_pass,
                   method           = 'fir',     # overlap-add FIR filtering
                   phase            = 'zero',    # the delay of this filter is compensated for
                   fir_window       = 'hamming', # The window to use in FIR design
                   fir_design       = 'firwin2',  # a time-domain design technique that generally gives improved attenuation using fewer samples than “firwin2”
                   )
    
    """
    2. epoch the data
    """
    picks       = mne.pick_types(raw_ref.info,
                           eeg      = True, # YES EEG
                           eog      = True, # YES EOG
                           )
    epochs      = mne.Epochs(raw_ref,
                             events,    # numpy array
                             event_id,  # dictionary
                        tmin        = tmin,
                        tmax        = tmax,
                        baseline    = (tmin,-0.2), # range of time for computing the mean references for each channel and subtract these values from all the time points per channel
                        picks       = picks,
                        detrend     = 1, # linear detrend
                        preload     = True # must be true if we want to do further processing
                        )
    
    """
    4. ica on epoch data
    """
    if ICA:
        """
        3. apply autoreject
        """
        picks       = mne.pick_types(epochs.info,
                               eeg          = True, # YES EEG
                               eog          = False # NO EOG
                               )
        ar          = AutoReject(
    #                    n_interpolate       = n_interpolates,
    #                    consensus           = consensus_pers,
    #                    thresh_method       = 'bayesian_optimization',
                        picks               = picks,
                        random_state        = 12345,
    #                    n_jobs              = 1,
    #                    verbose             = 'progressbar',
                        )
        ar.fit(epochs)
        _,reject_log = ar.transform(epochs,return_log=True)
        
        if logging is not None:
            fig = plot_EEG_autoreject_log(ar)
            fig.savefig(logging,bbox_inches = 'tight')
            for key in epochs.event_id.keys():
                evoked = epochs[key].average()
                fig_ = evoked.plot_joint(title = key) 
                fig_.savefig(logging.replace('.png',f'_{key}_pre.png'),
                             bbox_inches = 'tight')
                plt.close('all')
        # calculate the noise covariance of the epochs
        noise_cov   = mne.compute_covariance(epochs[~reject_log.bad_epochs],
                                             tmin                   = tmin,
                                             tmax                   = tmax,
                                             method                 = 'empirical',
                                             rank                   = None,)
        # define an ica function
        ica         = mne.preprocessing.ICA(n_components            = .99,
                                            n_pca_components        = .99,
                                            max_pca_components      = None,
                                            method                  = 'extended-infomax',
                                            max_iter                = int(3e3),
                                            noise_cov               = noise_cov,
                                            random_state            = 12345,)
    #    # search for a global rejection threshold globally
    #    reject      = get_rejection_threshold(epochs[~reject_log.bad_epochs],
    #                                          decim = 1,
    #                                          random_state = 12345)
        picks       = mne.pick_types(epochs.info,
                                     eeg = True, # YES EEG
                                     eog = False # NO EOG
                                     ) 
        ica.fit(epochs[~reject_log.bad_epochs],
                picks   = picks,
                start   = tmin,
                stop    = tmax,
    #            reject  = reject, # if some data in a window has values that exceed the rejection threshold, this window will be ignored when computing the ICA
                decim   = 3,
                tstep   = 1. # Length of data chunks for artifact rejection in seconds. It only applies if inst is of type Raw.
                )
        # search for artificial ICAs automatically
        # most of these hyperparameters were used in a unrelated published study
        ica.detect_artifacts(epochs[~reject_log.bad_epochs],
                             eog_ch         = ['FT9','FT10','TP9','TP10'],
                             eog_criterion  = 0.4, # arbitary choice
                             skew_criterion = 2,   # arbitary choice
                             kurt_criterion = 2,   # arbitary choice
                             var_criterion  = 2,   # arbitary choice
                             )
    #    # explicitly search for eog ICAs 
    #    eog_idx,scores = ica.find_bads_eog(raw_ref,
    #                            start       = tmin,
    #                            stop        = tmax,
    #                            l_freq      = 2,
    #                            h_freq      = 10,
    #                            )
    #    ica.exclude += eog_idx
        picks       = mne.pick_types(epochs.info,
                                     eeg = True, # YES EEG
                                     eog = False # NO EOG
                                     ) 
        epochs_ica  = ica.apply(epochs,#,[~reject_log.bad_epochs],
                                exclude    = ica.exclude,
                                )
    else:
        picks = mne.pick_types(epochs.info,
                               eeg = True,
                               eog = False,)
        epochs.filter(None,
                   low_pass,
                   picks            = picks,
                   filter_length    = 'auto',    # the filter length is chosen based on the size of the transition regions (6.6 times the reciprocal of the shortest transition band for fir_window=’hamming’ and fir_design=”firwin2”, and half that for “firwin”)
                   method           = 'fir',     # overlap-add FIR filtering
                   phase            = 'zero',    # the delay of this filter is compensated for
                   fir_window       = 'hamming', # The window to use in FIR design
                   fir_design       = 'firwin2',  # a time-domain design technique that generally gives improved attenuation using fewer samples than “firwin2”
                   )
        if logging is not None:
            for key in epochs.event_id.keys():
                evoked = epochs[key].average()
                fig_ = evoked.plot_joint(title = key) 
                fig_.savefig(logging.replace('.png',f'_{key}_post.png'),
                             bbox_inches = 'tight')
                plt.close('all')
        return epochs
    if fix:
        """
        
        """
        ar          = AutoReject(
#                        n_interpolate       = n_interpolates,
#                        consensus           = consensus_pers,
#                        thresh_method       = 'bayesian_optimization',
                        picks               = picks,
                        random_state        = 12345,
#                        n_jobs              = 1,
#                        verbose             = 'progressbar',
                        )
        epochs_clean = ar.fit_transform(epochs_ica,
                                        )
        
        return epochs_clean.pick_types(eeg=True,eog=False)
    else:
        clean_epochs = epochs_ica.pick_types(eeg = True,
                                             eog = False)
        picks = mne.pick_types(clean_epochs.info,
                               eeg = True,
                               eog = False,)
        clean_epochs.filter(None,
                   low_pass,
                   picks            = picks,
                   filter_length    = 'auto',    # the filter length is chosen based on the size of the transition regions (6.6 times the reciprocal of the shortest transition band for fir_window=’hamming’ and fir_design=”firwin2”, and half that for “firwin”)
                   method           = 'fir',     # overlap-add FIR filtering
                   phase            = 'zero',    # the delay of this filter is compensated for
                   fir_window       = 'hamming', # The window to use in FIR design
                   fir_design       = 'firwin2',  # a time-domain design technique that generally gives improved attenuation using fewer samples than “firwin2”
                   )
        if logging is not None:
            for key in clean_epochs.event_id.keys():
                evoked = epochs[key].average()
                fig_ = evoked.plot_joint(title = key) 
                fig_.savefig(logging.replace('.png',f'_{key}_post.png'),
                             bbox_inches = 'tight')
                plt.close('all')
        return clean_epochs
def plot_EEG_autoreject_log(autoreject_object,):
    ar = autoreject_object
    loss = ar.loss_['eeg'].mean(axis=-1)  # losses are stored by channel type.
    fig,ax = plt.subplots(figsize=(10,6))
    im = ax.matshow(loss.T * 1e6, cmap=plt.get_cmap('viridis'))
    ax.set(xticks = range(len(ar.consensus)), 
           xticklabels = ar.consensus.round(2),
           yticks = range(len(ar.n_interpolate)), 
           yticklabels = ar.n_interpolate)
    
    # Draw rectangle at location of best parameters
    idx, jdx = np.unravel_index(loss.argmin(), loss.shape)
    rect = patches.Rectangle((idx - 0.5, jdx - 0.5), 1, 1, linewidth=2,
                             edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    ax.xaxis.set_ticks_position('bottom')
    ax.set(xlabel = r'Consensus percentage $\kappa$',
           ylabel = r'Max sensors interpolated $\rho$',
           title = 'Mean cross validation error (x 1e6)')
    plt.colorbar(im)
    return fig
def str2int(x):
    return float(re.findall(r'\d+',x)[0])
def simple_load(f,idx):
    df = pd.read_csv(f)
    df['run'] = idx
    return df
def get_frames(directory,new = True,):
    files = glob(os.path.join(directory,'*trials.csv'))
    empty_temp = ''
    for ii,f in enumerate(files):
        df = pd.read_csv(f).dropna()
        for vis,df_sub in df.groupby(['visible.keys_raw']):
            try:
                print(f'session {ii+1}, vis = {vis}, n_trials = {df_sub.shape[0]}')
                empty_temp += f'session {ii+1}, vis = {vis}, n_trials = {df_sub.shape[0]}'
                empty_temp += '\n'
            except:
                print('session {}, vis = {}, n_trials = {}'.format(ii+1,
                      vis,df_sub.shape[0]))
                
    df = pd.concat([simple_load(f,ii).dropna() for ii,f in enumerate(files)])
    for col in ['probeFrames_raw',
                'response.keys_raw',
                'visible.keys_raw']:
        df[col] = df[col].apply(str2int)
    
    df = df.sort_values(['run','order'])
    
    for vis,df_sub in df.groupby(['visible.keys_raw']):
        df_press1 = df_sub[df_sub['response.keys_raw'] == 1]
        df_press2 = df_sub[df_sub['response.keys_raw'] == 2]
        prob1 = df_press1.shape[0] / df_sub.shape[0]
        prob2 = df_press2.shape[0] / df_sub.shape[0]
        try:
            print(f"\nvis = {vis},mean frames = {np.median(df_sub['probeFrames_raw']):.5f}")
            print(f"vis = {vis},prob(press 1) = {prob1:.4f}, p(press 2) = {prob2:.4f}")
            empty_temp += f"\nvis = {vis},mean frames = {np.median(df_sub['probeFrames_raw']):.5f}\n"
            empty_temp += f"vis = {vis},prob(press 1) = {prob1:.4f}, p(press 2) = {prob2:.4f}\n"
        except:
            print("\nvis = {},mean frames = {:.5f}".format(
                    vis,np.median(df_sub['probeFrames_raw'])))
            print(f"vis = {vis},prob(press 1) = {prob1:.4f}, p(press 2) = {prob2:.4f}")
    if new:
        df = []
        for f in files:
            temp = pd.read_csv(f).dropna()
            temp[['probeFrames_raw','visible.keys_raw']]
            probeFrame = []
            for ii,row in temp.iterrows():
                if int(re.findall(r'\d',row['visible.keys_raw'])[0]) == 1:
                    probeFrame.append(row['probeFrames_raw'])
                elif int(re.findall(r'\d',row['visible.keys_raw'])[0]) == 2:
                    probeFrame.append(row['probeFrames_raw'])
                elif int(re.findall(r'\d',row['visible.keys_raw'])[0]) == 3:
                    probeFrame.append(row['probeFrames_raw'])
                elif int(re.findall(r'\d',row['visible.keys_raw'])[0]) == 4:
                    probeFrame.append(row['probeFrames_raw'])
            temp['probeFrames'] = probeFrame
            df.append(temp)
        df = pd.concat(df)
    else:
        df = []
        for f in files:
            temp = pd.read_csv(f).dropna()
            temp[['probeFrames_raw','visible.keys_raw']]
            probeFrame = []
            for ii,row in temp.iterrows():
                if int(re.findall(r'\d',row['visible.keys_raw'])[0]) == 1:
                    probeFrame.append(row['probeFrames_raw'] - 2)
                elif int(re.findall(r'\d',row['visible.keys_raw'])[0]) == 2:
                    probeFrame.append(row['probeFrames_raw'] - 1)
                elif int(re.findall(r'\d',row['visible.keys_raw'])[0]) == 3:
                    probeFrame.append(row['probeFrames_raw'] + 1)
                elif int(re.findall(r'\d',row['visible.keys_raw'])[0]) == 4:
                    probeFrame.append(row['probeFrames_raw'] + 2)
            temp['probeFrames'] = probeFrame
            df.append(temp)
        df = pd.concat(df)
    df['probeFrames'] = df['probeFrames'].apply(str2int)
    results = []
    for vis,df_sub in df.groupby(['visible.keys_raw']):
        corrects = df_sub['response.corr_raw'].sum() / df_sub.shape[0]
        try:
            print(f"vis = {vis},N = {df_sub.shape[0]},mean frames = {np.mean(df_sub['probeFrames']):.2f} +/- {np.std(df_sub['probeFrames']):.2f}\np(correct) = {corrects:.4f}")
            empty_temp += f"vis = {vis},N = {df_sub.shape[0]},mean frames = {np.mean(df_sub['probeFrames']):.2f} +/- {np.std(df_sub['probeFrames']):.2f}\np(correct) = {corrects:.4f}\n"
            empty_temp += f"RT = {np.mean(df_sub['visible.rt_raw']):.3f} +/- {np.std(df_sub['visible.rt_raw']):.3f}\n"
        except:
            print("vis = {},mean frames = {:.2f} +/- {:.2f}".format(
                    vis,np.mean(df_sub['probeFrames']),np.std(df_sub['probeFrames'])))
        results.append([vis,np.mean(df_sub['probeFrames']),np.std(df_sub['probeFrames'])])
    return results,empty_temp

def preprocess_behavioral_file(f):
    df = read_behavorial_file(f)
    
    for col in ['probeFrames_raw',
                'response.keys_raw',
                'visible.keys_raw']:
        df[col] = df[col].apply(str2int)
    
    df = df.sort_values(['order'])
    return df

def read_behavorial_file(f):
    temp = pd.read_csv(f).iloc[:-12,:]
    return temp

def preload(f):
    temp = pd.read_csv(f).iloc[-12:,:2]
    return temp

def extract(x):
    try:
        return int(re.findall(r'\d',x)[0])
    except:
        return int(99)
#def extract_session_run_from_MRI(x):
#    temp = re.findall(r'\d+',x)
#    session = temp[1]
#    if int(session) == 7:
#        session = '1'
#    run = temp[-1]
#    return session,run
#def check_behaviral_data_session_block(x):
#    temp = preload(x)
#    temp.index = temp['category']
#    temp = temp.T
#    session = int(temp['session'].values[-1])
#    block = int(temp['block'].values[-1])
#    return session,block
#def compare_match(behavorial_file_name,session,block):
#    behav_session,behav_block = check_behaviral_data_session_block(behavorial_file_name)
#    if np.logical_and(behav_session == session, behav_block == block):
#        return True
#    else:
#        return False
def add_track(df_sub):
    n_rows = df_sub.shape[0]
    temp = '+'.join(str(item + 10) for item in df_sub['index'].values)
    df_sub = df_sub.iloc[1,:].to_frame().T
    df_sub['n_volume'] = n_rows
    df_sub['time_indices'] = temp
    return df_sub
def groupby_average(fmri,df,groupby = ['trials']):
    BOLD_average = np.array([np.mean(fmri[df_sub.index],0) for _,df_sub in df.groupby(groupby)])
    df_average = pd.concat([add_track(df_sub) for ii,df_sub in df.groupby(groupby)])
    return BOLD_average,df_average

def get_brightness_threshold(thresh):
    return [0.75 * val for val in thresh]

def get_brightness_threshold_double(thresh):
    return [2 * 0.75 * val for val in thresh]

def cartesian_product(fwhms, in_files, usans, btthresh):
    from nipype.utils.filemanip import ensure_list
    # ensure all inputs are lists
    in_files                = ensure_list(in_files)
    fwhms                   = [fwhms] if isinstance(fwhms, (int, float)) else fwhms
    # create cartesian product lists (s_<name> = single element of list)
    cart_in_file            = [
            s_in_file for s_in_file in in_files for s_fwhm in fwhms
                                ]
    cart_fwhm               = [
            s_fwhm for s_in_file in in_files for s_fwhm in fwhms
                                ]
    cart_usans              = [
            s_usans for s_usans in usans for s_fwhm in fwhms
                                ]
    cart_btthresh           = [
            s_btthresh for s_btthresh in btthresh for s_fwhm in fwhms
                                ]
    return cart_in_file, cart_fwhm, cart_usans, cart_btthresh

def getusans(x):
    return [[tuple([val[0], 0.5 * val[1]])] for val in x]

def create_fsl_FEAT_workflow_func(whichrun          = 0,
                                  whichvol          = 'middle',
                                  workflow_name     = 'nipype_mimic_FEAT',
                                  first_run         = True,
                                  func_data_file    = 'temp',
                                  fwhm              = 3):
    from nipype.workflows.fmri.fsl             import preprocess
    from nipype.interfaces                     import fsl
    from nipype.interfaces                     import utility as util
    from nipype.pipeline                       import engine as pe
    """
    Setup some functions and hyperparameters
    """
    fsl.FSLCommand.set_default_output_type('NIFTI_GZ')
    pickrun             = preprocess.pickrun
    pickvol             = preprocess.pickvol
    getthreshop         = preprocess.getthreshop
    getmeanscale        = preprocess.getmeanscale
#    chooseindex         = preprocess.chooseindex
    
    """
    Start constructing the workflow graph
    """
    preproc             = pe.Workflow(name = workflow_name)
    """
    Initialize the input and output spaces
    """
    inputnode           = pe.Node(
                        interface   = util.IdentityInterface(fields = ['func',
                                                                       'fwhm',
                                                                       'anat']),
                        name        = 'inputspec')
    outputnode          = pe.Node(
                        interface   = util.IdentityInterface(fields = ['reference',
                                                                       'motion_parameters',
                                                                       'realigned_files',
                                                                       'motion_plots',
                                                                       'mask',
                                                                       'smoothed_files',
                                                                       'mean']),
                        name        = 'outputspec')
    """
    first step: convert Images to float values
    """
    img2float           = pe.MapNode(
                        interface   = fsl.ImageMaths(
                                        out_data_type   = 'float',
                                        op_string       = '',
                                        suffix          = '_dtype'),
                        iterfield   = ['in_file'],
                        name        = 'img2float')
    preproc.connect(inputnode,'func',
                    img2float,'in_file')
    """
    delete first 10 volumes
    """
    develVolume         = pe.MapNode(
                        interface   = fsl.ExtractROI(t_min  = 10,
                                                     t_size = 508),
                        iterfield   = ['in_file'],
                        name        = 'remove_volumes')
    preproc.connect(img2float,      'out_file',
                    develVolume,    'in_file')
    if first_run == True:
        """ 
        extract example fMRI volume: middle one
        """
        extract_ref     = pe.MapNode(
                        interface   = fsl.ExtractROI(t_size = 1,),
                        iterfield   = ['in_file'],
                        name        = 'extractref')
        # connect to the deleteVolume node to get the data
        preproc.connect(develVolume,'roi_file',
                        extract_ref,'in_file')
        # connect to the deleteVolume node again to perform the extraction
        preproc.connect(develVolume,('roi_file',pickvol,0,whichvol),
                        extract_ref,'t_min')
        # connect to the output node to save the reference volume
        preproc.connect(extract_ref,'roi_file',
                        outputnode, 'reference')
    if first_run == True:
        """
        Realign the functional runs to the reference (`whichvol` volume of first run)
        """
        motion_correct  = pe.MapNode(
                        interface   = fsl.MCFLIRT(save_mats     = True,
                                                  save_plots    = True,
                                                  save_rms      = True,
                                                  stats_imgs    = True,
                                                  interpolation = 'spline'),
                        iterfield   = ['in_file','ref_file'],
                        name        = 'MCFlirt',
                                                  )
        # connect to the develVolume node to get the input data
        preproc.connect(develVolume,    'roi_file',
                        motion_correct, 'in_file',)
        ######################################################################################
        #################  the part where we replace the actual reference image if exists ####
        ######################################################################################
        # connect to the develVolume node to get the reference
        preproc.connect(extract_ref,    'roi_file', 
                        motion_correct, 'ref_file')
        ######################################################################################
        # connect to the output node to save the motion correction parameters
        preproc.connect(motion_correct, 'par_file',
                        outputnode,     'motion_parameters')
        # connect to the output node to save the other files
        preproc.connect(motion_correct, 'out_file',
                        outputnode,     'realigned_files')
    else:
        """
        Realign the functional runs to the reference (`whichvol` volume of first run)
        """
        motion_correct      = pe.MapNode(
                            interface   = fsl.MCFLIRT(ref_file      = first_run,
                                                      save_mats     = True,
                                                      save_plots    = True,
                                                      save_rms      = True,
                                                      stats_imgs    = True,
                                                      interpolation = 'spline'),
                            iterfield   = ['in_file','ref_file'],
                            name        = 'MCFlirt',
                        )
        # connect to the develVolume node to get the input data
        preproc.connect(develVolume,    'roi_file',
                        motion_correct, 'in_file',)
        # connect to the output node to save the motion correction parameters
        preproc.connect(motion_correct, 'par_file',
                        outputnode,     'motion_parameters')
        # connect to the output node to save the other files
        preproc.connect(motion_correct, 'out_file',
                        outputnode,     'realigned_files')
    """
    plot the estimated motion parameters
    """
    plot_motion             = pe.MapNode(
                            interface   = fsl.PlotMotionParams(in_source = 'fsl'),
                            iterfield   = ['in_file'],
                            name        = 'plot_motion',
            )
    plot_motion.iterables = ('plot_type',['rotations',
                                          'translations',
                                          'displacement'])
    preproc.connect(motion_correct, 'par_file',
                    plot_motion,    'in_file')
    preproc.connect(plot_motion,    'out_file',
                    outputnode,     'motion_plots')
    """
    extract the mean volume of the first functional run
    """
    meanfunc                = pe.Node(
                            interface  = fsl.ImageMaths(op_string   = '-Tmean',
                                                        suffix      = '_mean',),
                            name        = 'meanfunc')
    preproc.connect(motion_correct, ('out_file',pickrun,whichrun),
                    meanfunc,       'in_file')
    """
    strip the skull from the mean functional to generate a mask
    """
    meanfuncmask            = pe.Node(
                            interface   = fsl.BET(mask        = True,
                                                  no_output   = True,
                                                  frac        = 0.3,
                                                  surfaces    = True,),
                            name        = 'bet2_mean_func')
    preproc.connect(meanfunc,       'out_file',
                    meanfuncmask,   'in_file')
    """
    Mask the motion corrected functional data with the mask to create the masked (bet) motion corrected functional data
    """
    maskfunc                = pe.MapNode(
                            interface   = fsl.ImageMaths(suffix = '_bet',
                                                         op_string = '-mas'),
                            iterfield   = ['in_file'],
                            name        = 'maskfunc')
    preproc.connect(motion_correct, 'out_file',
                    maskfunc,       'in_file')
    preproc.connect(meanfuncmask,   'mask_file',
                    maskfunc,       'in_file2')
    """
    determine the 2nd and 98th percentiles of each functional run
    """
    getthreshold            = pe.MapNode(
                            interface   = fsl.ImageStats(op_string = '-p 2 -p 98'),
                            iterfield   = ['in_file'],
                            name        = 'getthreshold')
    preproc.connect(maskfunc,       'out_file',
                    getthreshold,   'in_file')
    """
    threshold the functional data at 10% of the 98th percentile
    """
    threshold               = pe.MapNode(
                            interface   = fsl.ImageMaths(out_data_type  = 'char',
                                                         suffix         = '_thresh',
                                                         op_string      = '-Tmin -bin'),
                            iterfield   = ['in_file','op_string'],
                            name        = 'tresholding')
    preproc.connect(maskfunc, 'out_file',
                    threshold,'in_file')
    """
    define a function to get 10% of the intensity
    """
    preproc.connect(getthreshold,('out_stat',getthreshop),
                    threshold,    'op_string')
    """
    Determine the median value of the functional runs using the mask
    """
    medianval               = pe.MapNode(
                            interface   = fsl.ImageStats(op_string = '-k %s -p 50'),
                            iterfield   = ['in_file','mask_file'],
                            name        = 'cal_intensity_scale_factor')
    preproc.connect(motion_correct,     'out_file',
                    medianval,          'in_file')
    preproc.connect(threshold,          'out_file',
                    medianval,          'mask_file')
    """
    dilate the mask
    """
    dilatemask              = pe.MapNode(
                            interface   = fsl.ImageMaths(suffix = '_dil',
                                                         op_string = '-dilF'),
                            iterfield   = ['in_file'],
                            name        = 'dilatemask')
    preproc.connect(threshold,  'out_file',
                    dilatemask, 'in_file')
    preproc.connect(dilatemask, 'out_file',
                    outputnode, 'mask')
    """
    mask the motion corrected functional runs with the dilated mask
    """
    dilateMask_MCed         = pe.MapNode(
                            interface   = fsl.ImageMaths(suffix     = '_mask',
                                                         op_string  = '-mas'),
                            iterfield   = ['in_file','in_file2'],
                            name        = 'dilateMask_MCed')
    preproc.connect(motion_correct,     'out_file',
                    dilateMask_MCed,    'in_file',)
    preproc.connect(dilatemask,         'out_file',
                    dilateMask_MCed,    'in_file2')
    """
    We now take this functional data that is motion corrected, high pass filtered, and
    create a "mean_func" image that is the mean across time (Tmean)
    """
    meanfunc2               = pe.MapNode(
                            interface   = fsl.ImageMaths(suffix     = '_mean',
                                                         op_string  = '-Tmean',),
                            iterfield   = ['in_file'],
                            name        = 'meanfunc2')
    preproc.connect(dilateMask_MCed,    'out_file',
                    meanfunc2,          'in_file')
    """
    smooth each run using SUSAN with the brightness threshold set to 
    75% of the median value for each run and a mask constituing the 
    mean functional
    """
    merge                   = pe.Node(
                            interface   = util.Merge(2, axis = 'hstack'), 
                            name        = 'merge')
    preproc.connect(meanfunc2,  'out_file', 
                    merge,      'in1')
    preproc.connect(medianval,('out_stat',get_brightness_threshold_double), 
                    merge,      'in2')
    smooth                  = pe.MapNode(
                            interface   = fsl.SUSAN(dimension   = 3,
                                                    use_median  = True),
                            iterfield   = ['in_file',
                                           'brightness_threshold',
                                           'fwhm',
                                           'usans'],
                            name        = 'susan_smooth')
    preproc.connect(dilateMask_MCed,    'out_file', 
                    smooth,             'in_file')
    preproc.connect(medianval,         ('out_stat',get_brightness_threshold),
                    smooth,             'brightness_threshold')
    preproc.connect(inputnode,          'fwhm', 
                    smooth,             'fwhm')
    preproc.connect(merge,              ('out',getusans),
                    smooth,             'usans')
    """
    mask the smoothed data with the dilated mask
    """
    maskfunc3               = pe.MapNode(
                            interface   = fsl.ImageMaths(suffix     = '_mask',
                                                         op_string  = '-mas'),
                            iterfield   = ['in_file','in_file2'],
                            name        = 'dilateMask_smoothed')
    # connect the output of the susam smooth component to the maskfunc3 node
    preproc.connect(smooth,     'smoothed_file',
                    maskfunc3,  'in_file')
    # connect the output of the dilated mask to the maskfunc3 node
    preproc.connect(dilatemask, 'out_file',
                    maskfunc3,  'in_file2')
    """
    scale the median value of the run is set to 10000
    """
    meanscale               = pe.MapNode(
                            interface   = fsl.ImageMaths(suffix = '_intnorm'),
                            iterfield   = ['in_file','op_string'],
                            name        = 'meanscale')
    preproc.connect(maskfunc3, 'out_file',
                    meanscale, 'in_file')
    preproc.connect(meanscale, 'out_file',
                    outputnode,'smoothed_files')
    """
    define a function to get the scaling factor for intensity normalization
    """
    preproc.connect(medianval,('out_stat',getmeanscale),
                    meanscale,'op_string')
    """
    generate a mean functional image from the first run
    should this be the 'mean.nii.gz' we will use in the future?
    """
    meanfunc3               = pe.MapNode(
                            interface   = fsl.ImageMaths(suffix     = '_mean',
                                                         op_string  = '-Tmean',),
                            iterfield   = ['in_file'],
                            name        = 'gen_mean_func_img')
    preproc.connect(meanscale, 'out_file',
                    meanfunc3, 'in_file')
    preproc.connect(meanfunc3, 'out_file',
                    outputnode,'mean')
    
    
    # initialize some of the input files
    preproc.inputs.inputspec.func       = os.path.abspath(func_data_file)
    preproc.inputs.inputspec.fwhm       = 3
    preproc.base_dir                    = os.path.abspath('/'.join(
                                        func_data_file.split('/')[:-1]))
    
    output_dir                          = os.path.abspath(os.path.join(
                                        preproc.base_dir,
                                        'outputs',
                                        'func'))
    MC_dir                              = os.path.join(output_dir,'MC')
    for directories in [output_dir,MC_dir]:
        if not os.path.exists(directories):
            os.makedirs(directories)
    
    # initialize all the output files
    if first_run == True:
        preproc.inputs.extractref.roi_file      = os.path.abspath(os.path.join(
                output_dir,'example_func.nii.gz'))
    
    preproc.inputs.dilatemask.out_file          = os.path.abspath(os.path.join(
                output_dir,'mask.nii.gz'))
    preproc.inputs.meanscale.out_file           = os.path.abspath(os.path.join(
                output_dir,'prefiltered_func.nii.gz'))
    preproc.inputs.gen_mean_func_img.out_file   = os.path.abspath(os.path.join(
                output_dir,'mean_func.nii.gz'))
    
    return preproc,MC_dir,output_dir


def create_registration_workflow(
                                 anat_brain,
                                 anat_head,
                                 func_ref,
                                 standard_brain,
                                 standard_head,
                                 standard_mask,
                                 workflow_name = 'registration',
                                 output_dir = 'temp'):
    from nipype.interfaces          import fsl
    from nipype.interfaces         import utility as util
    from nipype.pipeline           import engine as pe
    fsl.FSLCommand.set_default_output_type('NIFTI_GZ')
    """
    Start constructing the workflow graph
    """
    registration                    = pe.Workflow(name = 'registration')
    """
    Initialize the input and output spaces
    """
    inputnode                       = pe.Node(
                                    interface   = util.IdentityInterface(
                                    fields      = [
                                            'anat_brain',
                                            'anat_head',
                                            'func_ref',
                                            'standard_brain',
                                            'standard_head',
                                            'standard_mask'
                                            ]),
                                    name        = 'inputspec')
    outputnode                      = pe.Node(
                                    interface   = util.IdentityInterface(
                                    fields      = [
                                            'example2highres_FLIRT_mat',
                                            'example2highres_FLIRT_log',
                                            'example2highres_FLIRT_out_file',
                                            'highres2example_func_mat',
                                            'highres2standard_FLIRT_out_file',
                                            'highres2standard_FLIRT_mat',
                                            'highres2standard_FLIRT_log',
                                            'highres2standard_warp',
                                            'highres2standard_gz',
                                            'highres2highres_jac',
                                            'highres2highres_log',
                                            'highres2standard_mat',
                                            'example_func2standard_mat',
                                            'example_func2standard_warp',
                                            'example_func2standard',
                                            'standard2example_func_mat',
                                            ]),
                                    name        = 'outputspec')
    """
    /opt/fsl/fsl-5.0.10/fsl/bin/flirt 
        -in example_func 
        -ref highres 
        -out example_func2highres 
        -omat example_func2highres.mat 
        -cost corratio 
        -dof 7 
        -searchrx -180 180 
        -searchry -180 180 
        -searchrz -180 180 
        -interp trilinear 
    """
    example2highres_flirt           = pe.MapNode(
                                    interface   = fsl.FLIRT(cost          = 'corratio',
                                                            interp        = 'trilinear',
                                                            dof           = 7,
                                                            save_log      = True,
                                                            searchr_x     = [-180, 180],
                                                            searchr_y     = [-180, 180],
                                                            searchr_z     = [-180, 180],),
                                    iterfield   = ['in_file','reference'],
                                    name        = 'example2highres_flirt')
    registration.connect(inputnode,             'func_ref',
                         example2highres_flirt, 'in_file')
    registration.connect(inputnode,             'anat_brain',
                         example2highres_flirt, 'reference')
    registration.connect(example2highres_flirt, 'out_file',
                         outputnode,            'example2highres_FLIRT_out_file')
    registration.connect(example2highres_flirt, 'out_matrix_file',
                         outputnode,            'example2highres_FLIRT_mat')
    registration.connect(example2highres_flirt, 'out_log',
                         outputnode,            'example2highres_FLIRT_log')
    
    """
    /opt/fsl/fsl-5.0.10/fsl/bin/convert_xfm 
        -inverse -omat highres2example_func.mat example_func2highres.mat
    """
    inverse_example2highres         = pe.MapNode(
                                    interface   = fsl.ConvertXFM(invert_xfm = True),
                                    iterfield   = ['in_file',],
                                    name        = 'inverse_example2highres')
    registration.connect(example2highres_flirt,  'out_matrix_file',
                         inverse_example2highres,'in_file')
    registration.connect(inverse_example2highres,'out_file',
                         outputnode,             'highres2example_func_mat')
    """
    /opt/fsl/fsl-5.0.10/fsl/bin/flirt 
        -in highres 
        -ref standard 
        -out highres2standard 
        -omat highres2standard.mat 
        -cost corratio 
        -dof 12 
        -searchrx -180 180 
        -searchry -180 180 
        -searchrz -180 180 
        -interp trilinear 
    """
    highres2standard_flirt                  = pe.MapNode(
                                            interface   = fsl.FLIRT(cost        = 'corratio',
                                                                    interp      = 'trilinear',
                                                                    dof         = 12,
                                                                    save_log    = True,
                                                                    searchr_x   = [-180, 180],
                                                                    searchr_y   = [-180, 180],
                                                                    searchr_z   = [-180, 180],),
                                            iterfield   = ['in_file','reference'],
                                            name        = 'highres2standard_flirt')
    registration.connect(inputnode,                 'anat_brain',
                         highres2standard_flirt,    'in_file')
    registration.connect(inputnode,                 'standard_brain',
                         highres2standard_flirt,    'reference')
    registration.connect(highres2standard_flirt,    'out_file',
                         outputnode,                'highres2standard_FLIRT_out_file')
    registration.connect(highres2standard_flirt,    'out_matrix_file',
                         outputnode,                'highres2standard_FLIRT_mat')
    registration.connect(highres2standard_flirt,    'out_log',
                         outputnode,                'highres2standard_FLIRT_log')
    """
    /opt/fsl/fsl-5.0.10/fsl/bin/fnirt 
        --iout=highres2standard_head 
        --in=highres_head 
        --aff=highres2standard.mat 
        --cout=highres2standard_warp 
        --iout=highres2standard 
        --jout=highres2highres_jac 
        --config=T1_2_MNI152_2mm 
        --ref=standard_head 
        --refmask=standard_mask 
        --warpres=10,10,10
    """
    highres2standard_fnirt                  = pe.MapNode(
                                            interface   = fsl.FNIRT(warp_resolution = (10,10,10),
                                                                    config_file     = 'T1_2_MNI152_2mm'),
                                            iterfield   = ['in_file','affine_file',],
                                            name        = 'highres2standard_fnirt')
    registration.connect(inputnode,             'anat_head',
                         highres2standard_fnirt,'in_file') # <- nonlinear
    registration.connect(inputnode,             'standard_head',
                         highres2standard_fnirt,'ref_file',) # <- nonlinear
    registration.connect(highres2standard_flirt,'out_matrix_file', # <- linear
                         highres2standard_fnirt,'affine_file') # <- nonlinear
    registration.connect(highres2standard_fnirt,'fieldcoeff_file', # <- nonlinear
                         outputnode,            'highres2standard_warp')
    registration.connect(highres2standard_fnirt,'warped_file', # <- nonlinear
                         outputnode,            'highres2standard_gz_flirt')
    registration.connect(highres2standard_fnirt,'jacobian_file', # <- nonlinear
                         outputnode,            'highres2highres_jac')
    registration.connect(inputnode,             'standard_mask',
                         highres2standard_fnirt,'refmask_file') # <- nonlinear
    registration.connect(highres2standard_fnirt,'log_file', # <- nonlinear
                         outputnode,            'highres2standard_log')
    """
    /opt/fsl/fsl-5.0.10/fsl/bin/applywarp 
        -i highres 
        -r standard 
        -o highres2standard 
        -w highres2standard_warp
    """
    warp_anat_brain                         = pe.MapNode(
                                            interface   = fsl.ApplyWarp(),
                                            iterfield   = ['in_file',
                                                           'ref_file',
                                                           'field_file'],
                                            name        = 'warp_anat_brain')
    registration.connect(inputnode,             'anat_brain',
                         warp_anat_brain,       'in_file')
    registration.connect(inputnode,             'standard_brain',
                         warp_anat_brain,       'ref_file',)
    registration.connect(highres2standard_fnirt,'fieldcoeff_file',
                         warp_anat_brain,       'field_file')
    registration.connect(warp_anat_brain,       'out_file',
                         outputnode,            'highres2standard_gz_warp')
    
    """
    /opt/fsl/fsl-5.0.10/fsl/bin/convert_xfm 
        -inverse -omat standard2highres.mat highres2standard.mat
    """
    inverse_highres2standard            = pe.MapNode(
                                        interface   = fsl.ConvertXFM(invert_xfm = True),
                                        iterfield   = ['in_file',],
                                        name        = 'inverse_highres2standard')
    registration.connect(highres2standard_flirt,    'out_matrix_file',
                         inverse_highres2standard,  'in_file')
    registration.connect(inverse_example2highres,   'out_file',
                         outputnode,                'highres2standard_mat')
    """
    /opt/fsl/fsl-5.0.10/fsl/bin/convert_xfm 
        -omat example_func2standard.mat -concat highres2standard.mat example_func2highres.mat
    """
    example2standard                    = pe.MapNode(
                                        interface   = fsl.ConvertXFM(concat_xfm = True),
                                        iterfield   = ['in_file','in_file2',],
                                        name        = 'example2standard')
    registration.connect(highres2standard_flirt,    'out_matrix_file',
                         example2standard,          'in_file2')
    registration.connect(example2highres_flirt,     'out_matrix_file',
                         example2standard,          'in_file')
    registration.connect(example2standard,          'out_file',
                         outputnode,                'example_func2standard_mat')
    """
    /opt/fsl/fsl-5.0.10/fsl/bin/convertwarp 
        --ref=standard 
        --premat=example_func2highres.mat 
        --warp1=highres2standard_warp 
        --out=example_func2standard_warp
    """
    convertwarp_standard_brain          = pe.MapNode(
                                        interface   = fsl.ConvertWarp(),
                                        iterfield   = ['reference',
                                                       'premat',
                                                       'warp1'],
                                        name        = 'convertwarp_standard_brain')
    registration.connect(inputnode,                 'standard_brain',
                         convertwarp_standard_brain,'reference')
    registration.connect(example2highres_flirt,     'out_matrix_file',
                         convertwarp_standard_brain,'premat')
    registration.connect(highres2standard_fnirt,    'fieldcoeff_file',
                         convertwarp_standard_brain,'warp1')
    registration.connect(convertwarp_standard_brain,'out_file',
                         outputnode,                'example_func2standard_warp')
    """
    /opt/fsl/fsl-5.0.10/fsl/bin/applywarp 
        --ref=standard 
        --in=example_func 
        --out=example_func2standard 
        --warp=example_func2standard_warp
    """
    warp_standard_brain                 = pe.MapNode(
                                        interface   = fsl.ApplyWarp(),
                                        iterfield   = ['in_file','ref_file','field_file'],
                                        name        = 'warp_standard_brain')
    registration.connect(inputnode,                 'standard_brain',
                         warp_standard_brain,       'ref_file')
    registration.connect(inputnode,                 'func_ref',
                         warp_standard_brain,       'in_file')
    registration.connect(warp_standard_brain,       'out_file',
                         outputnode,                'example_func2standard')
    registration.connect(convertwarp_standard_brain,'out_file',
                         warp_standard_brain,       'field_file')
    """
    /opt/fsl/fsl-5.0.10/fsl/bin/convert_xfm 
        -inverse -omat standard2example_func.mat example_func2standard.mat
    """
    inverse_example2standard            = pe.MapNode(
                                        interface   = fsl.ConvertXFM(invert_xfm = True),
                                        iterfield   = ['in_file',],
                                        name        = 'inverse_example2standard')
    registration.connect(example2standard,          'out_file',
                         inverse_example2standard,  'in_file')
    registration.connect(inverse_example2standard,  'out_file',
                         outputnode,                'standard2example_func_mat')
    
    # initialize some of the input files with the directory
    registration.base_dir                                       = os.path.abspath(output_dir)
    registration.inputs.inputspec.anat_brain                    = anat_brain
    registration.inputs.inputspec.anat_head                     = anat_head
    registration.inputs.inputspec.func_ref                      = func_ref
    registration.inputs.inputspec.standard_brain                = standard_brain
    registration.inputs.inputspec.standard_head                 = standard_head
    registration.inputs.inputspec.standard_mask                 = standard_mask
    
    # define all the oupput file names with the directory
    registration.inputs.example2highres_flirt.out_file          = os.path.abspath(os.path.join(output_dir,'example_func2highres.nii.gz'))
    registration.inputs.example2highres_flirt.out_matrix_file   = os.path.abspath(os.path.join(output_dir,'example_func2highres.mat'))
    registration.inputs.example2highres_flirt.out_log           = os.path.abspath(os.path.join(output_dir,'example_func2highres.log'))
    registration.inputs.inverse_example2highres.out_file        = os.path.abspath(os.path.join(output_dir,'highres2example_func.mat'))
    registration.inputs.highres2standard_flirt.out_file         = os.path.abspath(os.path.join(output_dir,'highres2standard.nii.gz'))
    registration.inputs.highres2standard_flirt.out_matrix_file  = os.path.abspath(os.path.join(output_dir,'highres2standard.mat'))
    registration.inputs.highres2standard_flirt.out_log          = os.path.abspath(os.path.join(output_dir,'highres2standard.log'))
    registration.inputs.highres2standard_fnirt.fieldcoeff_file  = os.path.abspath(os.path.join(output_dir,'highres2standard_warp.nii.gz'))
    registration.inputs.highres2standard_fnirt.jacobian_file    = os.path.abspath(os.path.join(output_dir,'highres2highres_jac.nii.gz'))
    registration.inputs.warp_anat_brain.out_file                = os.path.abspath(os.path.join(output_dir,'highres2standard.nii.gz'))
    registration.inputs.inverse_highres2standard.out_file       = os.path.abspath(os.path.join(output_dir,'standard2highres.mat'))
    registration.inputs.example2standard.out_file               = os.path.abspath(os.path.join(output_dir,'example_func2standard.mat'))
    registration.inputs.convertwarp_standard_brain.out_file     = os.path.abspath(os.path.join(output_dir,'example_func2standard_warp.nii.gz'))
    registration.inputs.inverse_example2standard.out_file       = os.path.abspath(os.path.join(output_dir,'standard2example_func.mat'))
    #registration.inputs.highres2standard_fnirt.warped_file = os.path.abspath(os.path.join(output_dir,
    #                                                               'highres2standard.nii.gz'))
    #registration.inputs.warp_standard_brain.out_file = os.path.abspath(os.path.join(output_dir,
    #                                                  "example_func2standard.nii.gz"))
    return registration

def create_simple_struc2BOLD(roi,
                             roi_name,
                             preprocessed_functional_dir,
                             output_dir):
    from nipype.interfaces            import fsl
    from nipype.pipeline              import engine as pe
    from nipype.interfaces            import utility as util
    fsl.FSLCommand.set_default_output_type('NIFTI_GZ')
    
    simple_workflow         = pe.Workflow(name  = 'struc2BOLD')
    
    inputnode               = pe.Node(interface = util.IdentityInterface(
                                      fields    = ['flt_in_file',
                                                   'flt_in_matrix',
                                                   'flt_reference',
                                                   'mask']),
                                      name      = 'inputspec')
    outputnode              = pe.Node(interface = util.IdentityInterface(
                                      fields    = ['BODL_mask']),
                                      name      = 'outputspec')
    """
     flirt 
 -in /export/home/dsoto/dsoto/fmri/$s/sess2/label/$i 
 -ref /export/home/dsoto/dsoto/fmri/$s/sess2/run1_prepro1.feat/example_func.nii.gz  
 -applyxfm 
 -init /export/home/dsoto/dsoto/fmri/$s/sess2/run1_prepro1.feat/reg/highres2example_func.mat 
 -out  /export/home/dsoto/dsoto/fmri/$s/label/BOLD${i}
    """
    flirt_convert           = pe.MapNode(
                                    interface   = fsl.FLIRT(apply_xfm = True),
                                    iterfield   = ['in_file',
                                                   'reference',
                                                   'in_matrix_file'],
                                    name        = 'flirt_convert')
    simple_workflow.connect(inputnode,      'flt_in_file',
                            flirt_convert,  'in_file')
    simple_workflow.connect(inputnode,      'flt_reference',
                            flirt_convert,  'reference')
    simple_workflow.connect(inputnode,      'flt_in_matrix',
                            flirt_convert,  'in_matrix_file')
    
    """
     fslmaths /export/home/dsoto/dsoto/fmri/$s/label/BOLD${i} -mul 2 
     -thr `fslstats /export/home/dsoto/dsoto/fmri/$s/label/BOLD${i} -p 99.6` 
    -bin /export/home/dsoto/dsoto/fmri/$s/label/BOLD${i}
    """
    def getthreshop(thresh):
        return ['-mul 2 -thr %.10f -bin' % (val) for val in thresh]
    getthreshold            = pe.MapNode(
                                    interface   = fsl.ImageStats(op_string='-p 99.6'),
                                    iterfield   = ['in_file','mask_file'],
                                    name        = 'getthreshold')
    simple_workflow.connect(flirt_convert,  'out_file',
                            getthreshold,   'in_file')
    simple_workflow.connect(inputnode,      'mask',
                            getthreshold,   'mask_file')
    
    threshold               = pe.MapNode(
                                    interface   = fsl.ImageMaths(
                                            suffix      = '_thresh',
                                            op_string   = '-mul 2 -bin'),
                                    iterfield   = ['in_file','op_string'],
                                    name        = 'thresholding')
    simple_workflow.connect(flirt_convert,  'out_file',
                            threshold,      'in_file')
    simple_workflow.connect(getthreshold,   ('out_stat',getthreshop),
                            threshold,      'op_string')
#    simple_workflow.connect(threshold,'out_file',outputnode,'BOLD_mask')
    
    bound_by_mask           = pe.MapNode(
                                    interface   = fsl.ImageMaths(
                                            suffix      = '_mask',
                                            op_string   = '-mas'),
                                    iterfield   = ['in_file','in_file2'],
                                    name        = 'bound_by_mask')
    simple_workflow.connect(threshold,      'out_file',
                            bound_by_mask,  'in_file')
    simple_workflow.connect(inputnode,      'mask',
                            bound_by_mask,  'in_file2')
    simple_workflow.connect(bound_by_mask,  'out_file',
                            outputnode,     'BOLD_mask')
    
    # setup inputspecs 
    simple_workflow.inputs.inputspec.flt_in_file    = roi
    simple_workflow.inputs.inputspec.flt_in_matrix  = os.path.abspath(os.path.join(preprocessed_functional_dir,
                                                        'reg',
                                                        'highres2example_func.mat'))
    simple_workflow.inputs.inputspec.flt_reference  = os.path.abspath(os.path.join(preprocessed_functional_dir,
                                                        'func',
                                                        'example_func.nii.gz'))
    simple_workflow.inputs.inputspec.mask           = os.path.abspath(os.path.join(preprocessed_functional_dir,
                                                        'func',
                                                        'mask.nii.gz'))
    simple_workflow.inputs.bound_by_mask.out_file   = os.path.abspath(os.path.join(output_dir,
                                                         roi_name.replace('_fsl.nii.gz',
                                                                          '_BOLD.nii.gz')))
    return simple_workflow

def registration_plotting(output_dir,
                          anat_brain,
                          standard_brain):
    ######################
    ###### plotting ######
    try:
        example_func2highres    = os.path.abspath(os.path.join(output_dir,
                                                'example_func2highres'))
        example_func2standard   = os.path.abspath(os.path.join(output_dir,
                                                 'example_func2standard_warp'))
        highres2standard        = os.path.abspath(os.path.join(output_dir,
                                                 'highres2standard'))
        highres                 = os.path.abspath(anat_brain)
        standard                = os.path.abspath(standard_brain)
        
        plot_example_func2highres   = f"""
        /opt/fsl/fsl-5.0.10/fsl/bin/slicer {example_func2highres} {highres} -s 2 -x 0.35 sla.png -x 0.45 slb.png -x 0.55 slc.png -x 0.65 sld.png -y 0.35 sle.png -y 0.45 slf.png -y 0.55 slg.png -y 0.65 slh.png -z 0.35 sli.png -z 0.45 slj.png -z 0.55 slk.png -z 0.65 sll.png ; 
        /opt/fsl/fsl-5.0.10/fsl/bin/pngappend sla.png + slb.png + slc.png + sld.png + sle.png + slf.png + slg.png + slh.png + sli.png + slj.png + slk.png + sll.png {example_func2highres}1.png ; 
        /opt/fsl/fsl-5.0.10/fsl/bin/slicer {highres} {example_func2highres} -s 2 -x 0.35 sla.png -x 0.45 slb.png -x 0.55 slc.png -x 0.65 sld.png -y 0.35 sle.png -y 0.45 slf.png -y 0.55 slg.png -y 0.65 slh.png -z 0.35 sli.png -z 0.45 slj.png -z 0.55 slk.png -z 0.65 sll.png ; 
        /opt/fsl/fsl-5.0.10/fsl/bin/pngappend sla.png + slb.png + slc.png + sld.png + sle.png + slf.png + slg.png + slh.png + sli.png + slj.png + slk.png + sll.png {example_func2highres}2.png ; 
        /opt/fsl/fsl-5.0.10/fsl/bin/pngappend {example_func2highres}1.png - {example_func2highres}2.png {example_func2highres}.png; 
        /bin/rm -f sl?.png {example_func2highres}2.png
        /bin/rm {example_func2highres}1.png
        """
        
        plot_highres2standard       = f"""
        /opt/fsl/fsl-5.0.10/fsl/bin/slicer {highres2standard} {standard} -s 2 -x 0.35 sla.png -x 0.45 slb.png -x 0.55 slc.png -x 0.65 sld.png -y 0.35 sle.png -y 0.45 slf.png -y 0.55 slg.png -y 0.65 slh.png -z 0.35 sli.png -z 0.45 slj.png -z 0.55 slk.png -z 0.65 sll.png ; 
        /opt/fsl/fsl-5.0.10/fsl/bin/pngappend sla.png + slb.png + slc.png + sld.png + sle.png + slf.png + slg.png + slh.png + sli.png + slj.png + slk.png + sll.png {highres2standard}1.png ; 
        /opt/fsl/fsl-5.0.10/fsl/bin/slicer {standard} {highres2standard} -s 2 -x 0.35 sla.png -x 0.45 slb.png -x 0.55 slc.png -x 0.65 sld.png -y 0.35 sle.png -y 0.45 slf.png -y 0.55 slg.png -y 0.65 slh.png -z 0.35 sli.png -z 0.45 slj.png -z 0.55 slk.png -z 0.65 sll.png ; 
        /opt/fsl/fsl-5.0.10/fsl/bin/pngappend sla.png + slb.png + slc.png + sld.png + sle.png + slf.png + slg.png + slh.png + sli.png + slj.png + slk.png + sll.png {highres2standard}2.png ; 
        /opt/fsl/fsl-5.0.10/fsl/bin/pngappend {highres2standard}1.png - {highres2standard}2.png {highres2standard}.png; 
        /bin/rm -f sl?.png {highres2standard}2.png
        /bin/rm {highres2standard}1.png
        """
        
        plot_example_func2standard  = f"""
        /opt/fsl/fsl-5.0.10/fsl/bin/slicer {example_func2standard} {standard} -s 2 -x 0.35 sla.png -x 0.45 slb.png -x 0.55 slc.png -x 0.65 sld.png -y 0.35 sle.png -y 0.45 slf.png -y 0.55 slg.png -y 0.65 slh.png -z 0.35 sli.png -z 0.45 slj.png -z 0.55 slk.png -z 0.65 sll.png ; 
        /opt/fsl/fsl-5.0.10/fsl/bin/pngappend sla.png + slb.png + slc.png + sld.png + sle.png + slf.png + slg.png + slh.png + sli.png + slj.png + slk.png + sll.png {example_func2standard}1.png ; 
        /opt/fsl/fsl-5.0.10/fsl/bin/slicer {standard} {example_func2standard} -s 2 -x 0.35 sla.png -x 0.45 slb.png -x 0.55 slc.png -x 0.65 sld.png -y 0.35 sle.png -y 0.45 slf.png -y 0.55 slg.png -y 0.65 slh.png -z 0.35 sli.png -z 0.45 slj.png -z 0.55 slk.png -z 0.65 sll.png ; 
        /opt/fsl/fsl-5.0.10/fsl/bin/pngappend sla.png + slb.png + slc.png + sld.png + sle.png + slf.png + slg.png + slh.png + sli.png + slj.png + slk.png + sll.png {example_func2standard}2.png ; 
        /opt/fsl/fsl-5.0.10/fsl/bin/pngappend {example_func2standard}1.png - {example_func2standard}2.png {example_func2standard}.png; 
        /bin/rm -f sl?.png {example_func2standard}2.png
        """
        for cmdline in [plot_example_func2highres,
                        plot_example_func2standard,
                        plot_highres2standard]:
            os.system(cmdline)
    except:
        print('you should not use python 2.7, update your python!!')

def create_highpass_filter_workflow(workflow_name = 'highpassfiler',
                                    HP_freq = 60,
                                    TR = 0.85):
    from nipype.workflows.fmri.fsl    import preprocess
    from nipype.interfaces            import fsl
    from nipype.pipeline              import engine as pe
    from nipype.interfaces            import utility as util
    fsl.FSLCommand.set_default_output_type('NIFTI_GZ')
    getthreshop         = preprocess.getthreshop
    getmeanscale        = preprocess.getmeanscale
    highpass_workflow = pe.Workflow(name = workflow_name)
    
    inputnode               = pe.Node(interface = util.IdentityInterface(
                                      fields    = ['ICAed_file',]),
                                      name      = 'inputspec')
    outputnode              = pe.Node(interface = util.IdentityInterface(
                                      fields    = ['filtered_file']),
                                      name      = 'outputspec')
    
    img2float = pe.MapNode(interface    = fsl.ImageMaths(out_data_type     = 'float',
                                                         op_string         = '',
                                                         suffix            = '_dtype'),
                           iterfield    = ['in_file'],
                           name         = 'img2float')
    highpass_workflow.connect(inputnode,'ICAed_file',
                              img2float,'in_file')
    
    getthreshold = pe.MapNode(interface     = fsl.ImageStats(op_string = '-p 2 -p 98'),
                              iterfield     = ['in_file'],
                              name          = 'getthreshold')
    highpass_workflow.connect(img2float,    'out_file',
                              getthreshold, 'in_file')
    thresholding = pe.MapNode(interface     = fsl.ImageMaths(out_data_type  = 'char',
                                                             suffix         = '_thresh',
                                                             op_string      = '-Tmin -bin'),
                                iterfield   = ['in_file','op_string'],
                                name        = 'thresholding')
    highpass_workflow.connect(img2float,    'out_file',
                              thresholding, 'in_file')
    highpass_workflow.connect(getthreshold,('out_stat',getthreshop),
                              thresholding,'op_string')
    
    dilatemask = pe.MapNode(interface   = fsl.ImageMaths(suffix     = '_dil',
                                                         op_string  = '-dilF'),
                            iterfield   = ['in_file'],
                            name        = 'dilatemask')
    highpass_workflow.connect(thresholding,'out_file',
                              dilatemask,'in_file')
    
    maskfunc = pe.MapNode(interface     = fsl.ImageMaths(suffix     = '_mask',
                                                         op_string  = '-mas'),
                          iterfield     = ['in_file','in_file2'],
                          name          = 'apply_dilatemask')
    highpass_workflow.connect(img2float,    'out_file',
                              maskfunc,     'in_file')
    highpass_workflow.connect(dilatemask,   'out_file',
                              maskfunc,     'in_file2')
    
    medianval = pe.MapNode(interface    = fsl.ImageStats(op_string = '-k %s -p 50'),
                           iterfield    = ['in_file','mask_file'],
                           name         = 'cal_intensity_scale_factor')
    highpass_workflow.connect(img2float,    'out_file',
                              medianval,    'in_file')
    highpass_workflow.connect(thresholding, 'out_file',
                              medianval,    'mask_file')
    
    meanscale = pe.MapNode(interface    = fsl.ImageMaths(suffix = '_intnorm'),
                           iterfield    = ['in_file','op_string'],
                           name         = 'meanscale')
    highpass_workflow.connect(maskfunc,     'out_file',
                              meanscale,    'in_file')
    highpass_workflow.connect(medianval,    ('out_stat',getmeanscale),
                              meanscale,    'op_string')
    
    meanfunc = pe.MapNode(interface     = fsl.ImageMaths(suffix     = '_mean',
                                                         op_string  = '-Tmean'),
                           iterfield    = ['in_file'],
                           name         = 'meanfunc')
    highpass_workflow.connect(meanscale, 'out_file',
                              meanfunc,  'in_file')
    
    
    hpf = pe.MapNode(interface  = fsl.ImageMaths(suffix     = '_tempfilt',
                                                 op_string  = '-bptf %.10f -1' % (HP_freq/2/TR)),
                     iterfield  = ['in_file'],
                     name       = 'highpass_filering')
    highpass_workflow.connect(meanscale,'out_file',
                              hpf,      'in_file',)
    
    addMean = pe.MapNode(interface  = fsl.BinaryMaths(operation = 'add'),
                         iterfield  = ['in_file','operand_file'],
                         name       = 'addmean')
    highpass_workflow.connect(hpf,      'out_file',
                              addMean,  'in_file')
    highpass_workflow.connect(meanfunc, 'out_file',
                              addMean,  'operand_file')
    
    highpass_workflow.connect(addMean,      'out_file',
                              outputnode,   'filtered_file')
    
    return highpass_workflow

def load_csv(f,print_ = False):
    temp = re.findall(r'\d+',f)
    n_session = int(temp[-2])
    n_run = int(temp[-1])
    if print_:
        print(n_session,n_run)
    df = pd.read_csv(f)
    df['session'] = n_session
    df['run'] = n_run
    df['id'] = df['session'] * 1000 + df['run'] * 100 + df['trials']
    return df

def build_model_dictionary(print_train = False,
                           class_weight = 'balanced',
                           remove_invariant = False,
                           n_jobs = 1):
    np.random.seed(12345)
    svm = LinearSVC(penalty = 'l2', # default
                    dual = True, # default
                    tol = 1e-3, # not default
                    random_state = 12345, # not default
                    max_iter = int(1e3), # default
                    class_weight = class_weight, # not default
                    )
    svm = CalibratedClassifierCV(base_estimator = svm,
                                 method = 'sigmoid',
                                 cv = 8)
    xgb = XGBClassifier(
                        learning_rate                           = 1e-3, # not default
                        max_depth                               = 10, # not default
                        n_estimators                            = 100, # not default
                        objective                               = 'binary:logistic', # default
                        booster                                 = 'gbtree', # default
                        subsample                               = 0.9, # not default
                        colsample_bytree                        = 0.9, # not default
                        reg_alpha                               = 0, # default
                        reg_lambda                              = 1, # default
                        random_state                            = 12345, # not default
                        importance_type                         = 'gain', # default
                        n_jobs                                  = n_jobs,# default to be 1
                                              )
    bagging = BaggingClassifier(base_estimator                  = svm,
                                 n_estimators                   = 30, # not default
                                 max_features                   = 0.9, # not default
                                 max_samples                    = 0.9, # not default
                                 bootstrap                      = True, # default
                                 bootstrap_features             = True, # default
                                 random_state                   = 12345, # not default
                                                 )
    RF = SelectFromModel(xgb,
                        prefit                                  = False,
                        threshold                               = 'median' # induce sparsity
                        )
    uni = SelectPercentile(mutual_info_classif,50) # so annoying that I cannot control the random state
    knn = KNeighborsClassifier()
    tree = DecisionTreeClassifier(random_state = 12345,
                                  class_weight = class_weight)
    dummy = DummyClassifier(strategy = 'uniform',random_state = 12345,)
    if remove_invariant:
        RI = VarianceThreshold()
        models = OrderedDict([
                ['None + Dummy',                     make_pipeline(RI,MinMaxScaler(),
                                                                   dummy,)],
                ['None + Linear-SVM',                make_pipeline(RI,MinMaxScaler(),
                                                                  svm,)],
                ['None + Ensemble-SVMs',             make_pipeline(RI,MinMaxScaler(),
                                                                  bagging,)],
                ['None + KNN',                       make_pipeline(RI,MinMaxScaler(),
                                                                  knn,)],
                ['None + Tree',                      make_pipeline(RI,MinMaxScaler(),
                                                                  tree,)],
                ['PCA + Dummy',                      make_pipeline(RI,MinMaxScaler(),
                                                                   PCA(),
                                                                   dummy,)],
                ['PCA + Linear-SVM',                 make_pipeline(RI,MinMaxScaler(),
                                                                  PCA(),
                                                                  svm,)],
                ['PCA + Ensemble-SVMs',              make_pipeline(RI,MinMaxScaler(),
                                                                  PCA(),
                                                                  bagging,)],
                ['PCA + KNN',                        make_pipeline(RI,MinMaxScaler(),
                                                                  PCA(),
                                                                  knn,)],
                ['PCA + Tree',                       make_pipeline(RI,MinMaxScaler(),
                                                                  PCA(),
                                                                  tree,)],
                ['Mutual + Dummy',                   make_pipeline(RI,MinMaxScaler(),
                                                                   uni,
                                                                   dummy,)],
                ['Mutual + Linear-SVM',              make_pipeline(RI,MinMaxScaler(),
                                                                  uni,
                                                                  svm,)],
                ['Mutual + Ensemble-SVMs',           make_pipeline(RI,MinMaxScaler(),
                                                                  uni,
                                                                  bagging,)],
                ['Mutual + KNN',                     make_pipeline(RI,MinMaxScaler(),
                                                                  uni,
                                                                  knn,)],
                ['Mutual + Tree',                    make_pipeline(RI,MinMaxScaler(),
                                                                  uni,
                                                                  tree,)],
                ['RandomForest + Dummy',             make_pipeline(RI,MinMaxScaler(),
                                                                   RF,
                                                                   dummy,)],
                ['RandomForest + Linear-SVM',        make_pipeline(RI,MinMaxScaler(),
                                                                  RF,
                                                                  svm,)],
                ['RandomForest + Ensemble-SVMs',     make_pipeline(RI,MinMaxScaler(),
                                                                  RF,
                                                                  bagging,)],
                ['RandomForest + KNN',               make_pipeline(RI,MinMaxScaler(),
                                                                  RF,
                                                                  knn,)],
                ['RandomForest + Tree',              make_pipeline(RI,MinMaxScaler(),
                                                                  RF,
                                                                  tree,)],]
                )
    else:
        models = OrderedDict([
                ['None + Dummy',                     make_pipeline(MinMaxScaler(),
                                                                   dummy,)],
                ['None + Linear-SVM',                make_pipeline(MinMaxScaler(),
                                                                  svm,)],
                ['None + Ensemble-SVMs',             make_pipeline(MinMaxScaler(),
                                                                  bagging,)],
                ['None + KNN',                       make_pipeline(MinMaxScaler(),
                                                                  knn,)],
                ['None + Tree',                      make_pipeline(MinMaxScaler(),
                                                                  tree,)],
                ['PCA + Dummy',                      make_pipeline(MinMaxScaler(),
                                                                   PCA(),
                                                                   dummy,)],
                ['PCA + Linear-SVM',                 make_pipeline(MinMaxScaler(),
                                                                  PCA(),
                                                                  svm,)],
                ['PCA + Ensemble-SVMs',              make_pipeline(MinMaxScaler(),
                                                                  PCA(),
                                                                  bagging,)],
                ['PCA + KNN',                        make_pipeline(MinMaxScaler(),
                                                                  PCA(),
                                                                  knn,)],
                ['PCA + Tree',                       make_pipeline(MinMaxScaler(),
                                                                  PCA(),
                                                                  tree,)],
                ['Mutual + Dummy',                   make_pipeline(MinMaxScaler(),
                                                                   uni,
                                                                   dummy,)],
                ['Mutual + Linear-SVM',              make_pipeline(MinMaxScaler(),
                                                                  uni,
                                                                  svm,)],
                ['Mutual + Ensemble-SVMs',           make_pipeline(MinMaxScaler(),
                                                                  uni,
                                                                  bagging,)],
                ['Mutual + KNN',                     make_pipeline(MinMaxScaler(),
                                                                  uni,
                                                                  knn,)],
                ['Mutual + Tree',                    make_pipeline(MinMaxScaler(),
                                                                  uni,
                                                                  tree,)],
                ['RandomForest + Dummy',             make_pipeline(MinMaxScaler(),
                                                                   RF,
                                                                   dummy,)],
                ['RandomForest + Linear-SVM',        make_pipeline(MinMaxScaler(),
                                                                  RF,
                                                                  svm,)],
                ['RandomForest + Ensemble-SVMs',     make_pipeline(MinMaxScaler(),
                                                                  RF,
                                                                  bagging,)],
                ['RandomForest + KNN',               make_pipeline(MinMaxScaler(),
                                                                  RF,
                                                                  knn,)],
                ['RandomForest + Tree',              make_pipeline(MinMaxScaler(),
                                                                  RF,
                                                                  tree,)],]
                )
    return models

def get_blocks(df__,label_map,):
    ids = df__['id'].values
    chunks = df__['session'].values
    words = df__['labels'].values
    labels = np.array([label_map[item] for item in df__['targets'].values])[:,-1]
    sample_indecies = np.arange(len(labels))
    blocks = [np.array([ids[ids == target],
                        chunks[ids == target],
                        words[ids == target],
                        labels[ids == target],
                        sample_indecies[ids == target]
                       ]) for target in np.unique(ids)
                ]
    block_labels = np.array([np.unique(ll[-2]) for ll in blocks]).ravel()
    return blocks,block_labels

def make_unique_class_target(df_data):
    make_class = {name:[] for name in pd.unique(df_data['targets'])}
    for ii,df_sub in df_data.groupby(['labels']):
        target = pd.unique(df_sub['targets'])
        label = pd.unique(df_sub['labels'])
        make_class[target[0]].append(label[0])
    return make_class

def Find_Optimal_Cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------     
    list type, with optimal cutoff value

    """
    fpr, tpr, threshold         = roc_curve(target, predicted)
    i                           = np.arange(len(tpr)) 
    roc                         = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t                       = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold']) 
def customized_partition(df_data,groupby_column = ['id','labels'],n_splits = 100,):
    """
    modified for unaveraged volumes
    """
    idx_object = dict(ids = [],idx = [],labels = [])
    for label,df_sub in df_data.groupby(groupby_column):
        idx_object['ids'].append(label[0])
        idx_object['idx'].append(df_sub.index.tolist())
        idx_object['labels'].append(label[-1])
    df_object = pd.DataFrame(idx_object)
    idxs_test       = []
    for counter in range(int(1e4)):
        idx_test = [np.random.choice(item['idx'].values) for ii,item in df_object.groupby(groupby_column[-1])]
        if counter >= n_splits:
            return [np.concatenate(item) for item in idxs_test]
            break
        if counter > 0:
            temp = []
            for used in idxs_test:
                used_temp = [','.join(str(ii) for ii in item) for item in used]
                idx_test_temp = [','.join(str(ii) for ii in item) for item in idx_test]
                a = set(used_temp)
                b = set(idx_test_temp)
                temp.append(len(a.intersection(b)) != len(idx_test))
            if all(temp) == True:
                idxs_test.append(idx_test)
        else:
            idxs_test.append(idx_test)
def check_train_test_splits(idxs_test):
    temp = []
    for ii,item1 in enumerate(idxs_test):
        for jj,item2 in enumerate(idxs_test):
            if not ii == jj:
                if len(item1) == len(item2):
                    sample1 = np.sort(item1)
                    sample2 = np.sort(item2)
                    
                    temp.append(len(set(sample1).intersection(set(sample2))) == len(sample1))
    temp = np.array(temp)
    return any(temp)
def check_train_balance(df,idx_train,keys):
    Counts = dict(Counter(df.iloc[idx_train]['targets'].values))
    if np.abs(Counts[keys[0]] - Counts[keys[1]]) > 2:
        if Counts[keys[0]] > Counts[keys[1]]:
            key_major = keys[0]
            key_minor = keys[1]
        else:
            key_major = keys[1]
            key_minor = keys[0]
            
        ids_major = df.iloc[idx_train]['id'][df.iloc[idx_train]['targets'] == key_major]
        
        idx_train_new = idx_train.copy()
        for n in range(len(idx_train_new)):
            random_pick = np.random.choice(np.unique(ids_major),size = 1)[0]
            # print(random_pick,np.unique(ids_major))
            idx_train_new = np.array([item for item,id_temp in zip(idx_train_new,df.iloc[idx_train_new]['id']) if (id_temp != random_pick)])
            ids_major = np.array([item for item in ids_major if (item != random_pick)])
            new_counts = dict(Counter(df.iloc[idx_train_new]['targets']))
            if np.abs(new_counts[keys[0]] - new_counts[keys[1]]) > 3:
                if new_counts[keys[0]] > new_counts[keys[1]]:
                    key_major = keys[0]
                    key_minor = keys[1]
                else:
                    key_major = keys[1]
                    key_minor = keys[0]
                
                ids_major = df.iloc[idx_train_new]['id'][df.iloc[idx_train_new]['targets'] == key_major]
            elif np.abs(new_counts[keys[0]] - new_counts[keys[1]]) < 3:
                break
        return idx_train_new
    else:
        return idx_train


def LOO_partition(data,df_data):
    temp = {'targets':[],'labels':[]}
    for (targets,labels),df_sub in df_data.groupby(['targets','labels']):
        temp['targets'].append(targets)
        temp['labels'].append(labels)
    temp = pd.DataFrame(temp)
    temp = temp.sort_values(['targets','labels'])
    living = temp[temp['targets'] == 'Living_Things']['labels'].values
    nonliving = temp[temp['targets'] == 'Nonliving_Things']['labels'].values
    test_pairs = [[a,b] for a in living for b in nonliving]
    idxs_train,idxs_test = [],[]
    for test_pair in test_pairs:
        idx_test = np.logical_or(df_data['labels'] == test_pair[0],
                                 df_data['labels'] == test_pair[1])
        idx_train = np.invert(idx_test)
        idxs_train.append(np.where(idx_train == True)[0])
        idxs_test.append(np.where(idx_test == True)[0])
    return idxs_train,idxs_test
def resample_ttest(x,baseline = 0.5,n_ps = 100,n_permutation = 5000,one_tail = False):
    """
    http://www.stat.ucla.edu/~rgould/110as02/bshypothesis.pdf
    Inputs:
    ----------
    x: numpy array vector, the data that is to be compared
    baseline: the single point that we compare the data with
    n_ps: number of p values we want to estimate
    n_permutation: number of permutation we want to perform, the more the further it could detect the strong effects, but it is so unnecessary
    one_tail: whether to perform one-tailed comparison
    """
    import numpy as np
    experiment      = np.mean(x) # the mean of the observations in the experiment
    experiment_diff = x - np.mean(x) + baseline # shift the mean to the baseline but keep the distribution
    # newexperiment = np.mean(experiment_diff) # just look at the new mean and make sure it is at the baseline
    # simulate/bootstrap null hypothesis distribution
    # 1st-D := number of sample same as the experiment
    # 2nd-D := within one permutation resamping, we perform resampling same as the experimental samples,
    # but also repeat this one sampling n_permutation times
    # 3rd-D := repeat 2nd-D n_ps times to obtain a distribution of p values later
    temp            = np.random.choice(experiment_diff,size=(x.shape[0],n_permutation,n_ps),replace=True)
    temp            = temp.mean(0)# take the mean over the sames because we only care about the mean of the null distribution
    # along each row of the matrix (n_row = n_permutation), we count instances that are greater than the observed mean of the experiment
    # compute the proportion, and we get our p values
    
    if one_tail:
        ps = (np.sum(temp >= experiment,axis=0)+1.) / (n_permutation + 1.)
    else:
        ps = (np.sum(np.abs(temp) >= np.abs(experiment),axis=0)+1.) / (n_permutation + 1.)
    return ps
def resample_ttest_2sample(a,b,n_ps=100,n_permutation=5000,one_tail=False,match_sample_size = True,):
    # when the N is matched just simply test the pairwise difference against 0
    # which is a one sample comparison problem
    if match_sample_size:
        difference  = a - b
        ps          = resample_ttest(difference,baseline=0,n_ps=n_ps,n_permutation=n_permutation,one_tail=one_tail)
        return ps
    else: # when the N is not matched
        difference              = np.mean(a) - np.mean(b)
        concatenated            = np.concatenate([a,b])
        np.random.shuffle(concatenated)
        temp                    = np.zeros((n_permutation,n_ps))
        # the next part of the code is to estimate the "randomized situation" under the given data's distribution
        # by randomized the items in each group (a and b), we can compute the chance level differences
        # and then we estimate the probability of the chance level exceeds the true difference 
        # as to represent the "p value"
        try:
            iterator            = tqdm(range(n_ps),desc='ps')
        except:
            iterator            = range(n_ps)
        for n_p in iterator:
            for n_permu in range(n_permutation):
                idx_a           = np.random.choice(a    = [0,1],
                                                   size = (len(a)+len(b)),
                                                   p    = [float(len(a))/(len(a)+len(b)),
                                                           float(len(b))/(len(a)+len(b))]
                                                   ).astype(np.bool)
                idx_b           = np.logical_not(idx_a)
                d               = np.mean(concatenated[idx_a]) - np.mean(concatenated[idx_b])
                if np.isnan(d):
                    idx_a       = np.random.choice(a        = [0,1],
                                                   size     = (len(a)+len(b)),
                                                   p        = [float(len(a))/(len(a)+len(b)),
                                                               float(len(b))/(len(a)+len(b))]
                                                   ).astype(np.bool)
                    idx_b       = np.logical_not(idx_a)
                    d           = np.mean(concatenated[idx_a]) - np.mean(concatenated[idx_b])
                temp[n_permu,n_p] = d
        if one_tail:
            ps = (np.sum(temp >= difference,axis=0)+1.) / (n_permutation + 1.)
        else:
            ps = (np.sum(np.abs(temp) >= np.abs(difference),axis=0)+1.) / (n_permutation + 1.)
        return ps

class MCPConverter(object):
    import statsmodels as sms
    """
    https://gist.github.com/naturale0/3915e2def589553e91dce99e69d138cc
    https://en.wikipedia.org/wiki/Holm%E2%80%93Bonferroni_method
    input: array of p-values.
    * convert p-value into adjusted p-value (or q-value)
    """
    def __init__(self, pvals, zscores = None):
        self.pvals                    = pvals
        self.zscores                  = zscores
        self.len                      = len(pvals)
        if zscores is not None:
            srted                     = np.array(sorted(zip(pvals.copy(), zscores.copy())))
            self.sorted_pvals         = srted[:, 0]
            self.sorted_zscores       = srted[:, 1]
        else:
            self.sorted_pvals         = np.array(sorted(pvals.copy()))
        self.order                    = sorted(range(len(pvals)), key=lambda x: pvals[x])
    
    def adjust(self, method           = "holm"):
        import statsmodels as sms
        """
        methods = ["bonferroni", "holm", "bh", "lfdr"]
         (local FDR method needs 'statsmodels' package)
        """
        if method is "bonferroni":
            return [np.min([1, i]) for i in self.sorted_pvals * self.len]
        elif method is "holm":
            return [np.min([1, i]) for i in (self.sorted_pvals * (self.len - np.arange(1, self.len+1) + 1))]
        elif method is "bh":
            p_times_m_i = self.sorted_pvals * self.len / np.arange(1, self.len+1)
            return [np.min([p, p_times_m_i[i+1]]) if i < self.len-1 else p for i, p in enumerate(p_times_m_i)]
        elif method is "lfdr":
            if self.zscores is None:
                raise ValueError("Z-scores were not provided.")
            return sms.stats.multitest.local_fdr(abs(self.sorted_zscores))
        else:
            raise ValueError("invalid method entered: '{}'".format(method))
            
    def adjust_many(self, methods = ["bonferroni", "holm", "bh", "lfdr"]):
        if self.zscores is not None:
            df = pd.DataFrame(np.c_[self.sorted_pvals, self.sorted_zscores], columns=["p_values", "z_scores"])
            for method in methods:
                df[method] = self.adjust(method)
        else:
            df = pd.DataFrame(self.sorted_pvals, columns=["p_values"])
            for method in methods:
                if method is not "lfdr":
                    df[method] = self.adjust(method)
        return df
def define_roi_category():
    roi_dict = {'fusiform':'Visual',
                'parahippocampal':'Visual',
                'pericalcarine':'Visual',
                'precuneus':'Visual',
                'superiorparietal':'Visual',
                'inferiortemporal':'Working Memory',
                'inferiortemporal':'Working Memory',
                'lateraloccipital':'Working Memory',
                'lingual':'Working Memory',
                'rostralmiddlefrontal':'Working Memory',
                'superiorfrontal':'Working Memory',
                'ventrolateralPFC':'Working Memory',
                }
    
    return roi_dict




















#def _create_fsl_FEAT_workflow_func(whichrun = 0,
#                                   whichvol = 'middle',
#                                   workflow_name = 'nipype_mimic_FEAT',
#                                   first_run = True):
#    from nipype.workflows.fmri.fsl import preprocess
#    from nipype.interfaces import fsl
#    from nipype.interfaces import utility as util
#    from nipype.pipeline import engine as pe
#    
#    """
#    Setup some functions and hyperparameters
#    """
#    fsl.FSLCommand.set_default_output_type('NIFTI_GZ')
#    pickrun = preprocess.pickrun
#    pickvol = preprocess.pickvol
#    getthreshop = preprocess.getthreshop
#    getmeanscale = preprocess.getmeanscale
#    create_susan_smooth = preprocess.create_susan_smooth
##    chooseindex = preprocess.chooseindex
#    
#    """
#    Start constructing the workflow graph
#    """
#    preproc = pe.Workflow(name = workflow_name)
#    """
#    Initialize the input and output spaces
#    """
#    inputnode = pe.Node(
#            interface = util.IdentityInterface(
#                    fields=['func','fwhm','anat']),
#                    name = 'inputspec')
#    outputnode = pe.Node(
#            interface = util.IdentityInterface(
#                    fields = [
#                            'reference',
#                            'motion_parameters',
#                            'realigned_files',
#                            'motion_plots',
#                            'mask',
#                            'smoothed_files',
#                            'mean']),
#                    name = 'outputspec')
#    """
#    first step: convert Images to float values
#    """
#    img2float = pe.MapNode(
#            interface=fsl.ImageMaths(
#                    out_data_type='float',op_string='',suffix='_dtype'),
#                    iterfield=['in_file'],
#                    name = 'img2float')
#    preproc.connect(inputnode,'func',img2float,'in_file')
#    """
#    delete first 10 volumes
#    """
#    develVolume = pe.MapNode(
#            interface = fsl.ExtractROI(t_min = 10,t_size = 508),
#            iterfield = ['in_file'],
#            name = 'remove_volumes')
#    preproc.connect(img2float,'out_file',develVolume,'in_file')
#    """ 
#    extract example fMRI volume: middle one
#    """
#    extract_ref = pe.MapNode(interface=fsl.ExtractROI(t_size=1,),
#                          iterfield=['in_file'],
#                          name = 'extractref')
#    # connect to the deleteVolume node to get the data
#    preproc.connect(develVolume,'roi_file',
#                    extract_ref,'in_file')
#    # connect to the deleteVolume node again to perform the extraction
#    preproc.connect(develVolume,('roi_file',pickvol,0,whichvol),
#                    extract_ref,'t_min')
#    # connect to the output node to save the reference volume
#    preproc.connect(extract_ref,'roi_file',
#                    outputnode,'reference')
#    if first_run == True:
#        """
#        Realign the functional runs to the reference (`whichvol` volume of first run)
#        """
#        motion_correct = pe.MapNode(
#                interface = fsl.MCFLIRT(
#                        save_mats = True,
#                        save_plots = True,
#                        save_rms = True,
#                        stats_imgs = True,
#                        interpolation = 'spline'),
#                        iterfield = ['in_file','ref_file'],
#                        name = 'MCFlirt',
#                        )
#        # connect to the develVolume node to get the input data
#        preproc.connect(develVolume,'roi_file',
#                        motion_correct,'in_file',)
#        ######################################################################################
#        #################  the part where we replace the actual reference image if exists ####
#        ######################################################################################
#        # connect to the develVolume node to get the reference
#        preproc.connect(extract_ref, 'roi_file', 
#                        motion_correct,'ref_file')
#        ######################################################################################
#        # connect to the output node to save the motion correction parameters
#        preproc.connect(motion_correct,'par_file',
#                        outputnode,'motion_parameters')
#        # connect to the output node to save the other files
#        preproc.connect(motion_correct,'out_file',
#                        outputnode,'realigned_files')
#    else:
#        """
#        Realign the functional runs to the reference (`whichvol` volume of first run)
#        """
#        motion_correct = pe.MapNode(
#                interface = fsl.MCFLIRT(
#                        ref_file = first_run,
#                        save_mats = True,
#                        save_plots = True,
#                        save_rms = True,
#                        stats_imgs = True,
#                        interpolation = 'spline'),
#                        iterfield = ['in_file','ref_file'],
#                        name = 'MCFlirt',
#                        )
#        # connect to the develVolume node to get the input data
#        preproc.connect(develVolume,'roi_file',
#                        motion_correct,'in_file',)
#        # connect to the output node to save the motion correction parameters
#        preproc.connect(motion_correct,'par_file',
#                        outputnode,'motion_parameters')
#        # connect to the output node to save the other files
#        preproc.connect(motion_correct,'out_file',
#                        outputnode,'realigned_files')
#    """
#    plot the estimated motion parameters
#    """
#    plot_motion = pe.MapNode(
#            interface=fsl.PlotMotionParams(in_source='fsl'),
#            iterfield = ['in_file'],
#            name = 'plot_motion',
#            )
#    plot_motion.iterables = ('plot_type',['rotations','translations'])
#    preproc.connect(motion_correct,'par_file',
#                    plot_motion,'in_file')
#    preproc.connect(plot_motion,'out_file',
#                    outputnode,'motion_plots')
#    """
#    extract the mean volume of the first functional run
#    """
#    meanfunc = pe.Node(
#            interface=fsl.ImageMaths(op_string = '-Tmean',
#                                     suffix='_mean',
#                                     ),
#            name = 'meanfunc')
#    preproc.connect(motion_correct,('out_file',pickrun,whichrun),
#                    meanfunc,'in_file')
#    """
#    strip the skull from the mean functional to generate a mask
#    """
#    meanfuncmask = pe.Node(
#            interface=fsl.BET(mask=True,
#                              no_output=True,
#                              frac=0.3,
##                              Robust=True,
#                              ),
#            name='bet2_mean_func')
#    preproc.connect(meanfunc,'out_file',
#                    meanfuncmask,'in_file')
#    """
#    mask the motion corrected functional runs with the extracted mask
#    """
#    maskfunc = pe.MapNode(
#            interface=fsl.ImageMaths(suffix='_bet',op_string='-mas'),
#            iterfield=['in_file'],
#            name='maskfunc')
#    preproc.connect(motion_correct,'out_file',
#                    maskfunc,'in_file')
#    preproc.connect(meanfuncmask,'mask_file',
#                    maskfunc,'in_file2')
#    """
#    determine the 2nd and 98th percentiles of each functional run
#    """
#    getthreshold = pe.MapNode(
#            interface=fsl.ImageStats(op_string='-p 2 -p 98'),
#            iterfield = ['in_file'],
#            name='getthreshold')
#    preproc.connect(maskfunc,'out_file',getthreshold,'in_file')
#    """
#    threshold the first run of the functional data at 10% of the 98th percentile
#    """
#    threshold = pe.MapNode(
#            interface=fsl.ImageMaths(out_data_type='char',suffix='_thresh',
#                                     op_string = '-Tmin -bin'),
#            iterfield=['in_file','op_string'],
#            name='tresholding')
#    preproc.connect(maskfunc,'out_file',threshold,'in_file')
#    """
#    define a function to get 10% of the intensity
#    """
#    preproc.connect(getthreshold,('out_stat',getthreshop),threshold,
#                    'op_string')
#    """
#    Determine the median value of the functional runs using the mask
#    """
#    medianval = pe.MapNode(
#            interface = fsl.ImageStats(op_string = '-k %s -p 50'),
#            iterfield = ['in_file','mask_file'],
#            name='cal_intensity_scale_factor')
#    preproc.connect(motion_correct,'out_file',
#                    medianval,'in_file')
#    preproc.connect(maskfunc,'out_file',
#                    medianval,'mask_file')
#    """
#    dilate the mask
#    """
#    dilatemask = pe.MapNode(
#            interface = fsl.ImageMaths(suffix='_dil',op_string='-dilF'),
#            iterfield=['in_file'],
#            name = 'dilatemask')
#    preproc.connect(threshold,'out_file',dilatemask,'in_file')
#    preproc.connect(dilatemask,'out_file',outputnode,'mask')
#    """
#    mask the motion corrected functional runs with the dilated mask
#    """
#    maskfunc2 = pe.MapNode(
#            interface = fsl.ImageMaths(suffix='_mask',op_string='-mas'),
#            iterfield=['in_file','in_file2'],
#            name='dilateMask_MCed')
#    preproc.connect(motion_correct,'out_file',maskfunc2,'in_file',)
#    preproc.connect(dilatemask,'out_file',maskfunc2,'in_file2')
#    """
#    smooth each run using SUSAN with the brightness threshold set to 
#    75% of the median value for each run and a mask constituing the 
#    mean functional
#    """
#    smooth = create_susan_smooth()
#    preproc.connect(inputnode,'fwhm',smooth,'inputnode.fwhm')
#    preproc.connect(maskfunc2,'out_file',smooth,'inputnode.in_files')
#    preproc.connect(dilatemask,'out_file',smooth,'inputnode.mask_file')
#    """
#    mask the smoothed data with the dilated mask
#    """
#    maskfunc3 = pe.MapNode(
#            interface = fsl.ImageMaths(suffix='_mask',op_string='-mas'),
#            iterfield = ['in_file','in_file2'],
#            name='dilateMask_smoothed')
#    # connect the output of the susam smooth component to the maskfunc3 node
#    preproc.connect(smooth,'outputnode.smoothed_files',
#                    maskfunc3,'in_file')
#    # connect the output of the dilated mask to the maskfunc3 node
#    preproc.connect(dilatemask,'out_file',
#                    maskfunc3,'in_file2')
#    """
#    scale the median value of the run is set to 10000
#    """
#    meanscale = pe.MapNode(
#            interface = fsl.ImageMaths(suffix='_gms'),
#            iterfield = ['in_file','op_string'],
#            name = 'meanscale')
#    preproc.connect(maskfunc3,'out_file',
#                    meanscale,'in_file')
#    """
#    define a function to get the scaling factor for intensity normalization
#    """
#    preproc.connect(medianval,('out_stat',getmeanscale),
#                    meanscale,'op_string')
#    """
#    generate a mean functional image from the first run
#    should this be the 'mask.nii.gz' we will use in the future?
#    """
#    meanfunc3 = pe.MapNode(
#            interface = fsl.ImageMaths(suffix='_mean',
#                                       op_string='-Tmean',),
#            iterfield = ['in_file'],
#            name='gen_mean_func_img')
#    preproc.connect(meanscale,'out_file',meanfunc3,'in_file')
#    preproc.connect(meanfunc3,'out_file',outputnode,'mean')
#    
#    return preproc
#def customized_partition(df,label_map):
#    """
#    To customize the random partitioning, this function would randomly select instance of volumes
#    that correspond to unique words used in an experiment from different scanning blocks to form
#    the test set. 
#    By doing so, we create a quasi-leave-one-block-out cross-validation
#    """
#    unique_labels = pd.unique(df['labels'])
#    unique_chunks = pd.unique(df['session'])
#    labels = np.array([label_map[item] for item in df['targets'].values])[:,-1]
#    labels = df['labels'].values
#    chunks = df['session'].values
#    blocks,block_labels = get_blocks(df,label_map)
#    sample_indecies = np.arange(len(labels))
#    test,check = [],[]
##    for n in tqdm(range(int(2e3))):
##    pbar = tqdm(total = 96)
#    while len(check) != len(unique_labels):
##        print(counter_i,end=', ')
##        counter_i += 1
#        # randomly pick on of the scanning sessions/chunks
#        random_chunk = np.random.choice(unique_chunks,size=1,replace=False)[0]
#        # get indecies of the labels in this block from the whole dataset
#        working_labels = labels[chunks == random_chunk]
#        # variable "block" is a list. Each item contains: id, chunk, label, target, sample indecies
#        # and we only need to access the "chunk"
#        working_block = [block for block in blocks if (int(np.unique(block[1])[0]) == random_chunk)]
#        # pick a label randomly from the labels in the picked block
#        random_label = np.random.choice(working_labels,size=1,replace=False)[0]
#        if random_label not in check: # we need to check if the label has been selected before
#            for block in working_block:
#                if (np.unique(block[2])[0] == random_label) and (random_label not in check):
#                    test.append(block[-1].astype(int))
#                    check.append(block[2][0])
##                    pbar.update(1)
##                if len(check) == len(unique_labels):
##                    break
##            if len(check) == len(unique_labels):
##                break
##        if len(check) == len(unique_labels):
##            break
#    test = np.concatenate(test,0).flatten()
#    train = np.array([idx for idx in sample_indecies if (idx not in test)])
##    print(pd.unique(df.iloc[test]['labels']).shape)
#    return train,test
#
#def get_train_test_splits(df,label_map,n_splits):
#    idxs_train,idxs_test = [],[]
#    np.random.seed(12345)
#    used_test = []
#    fold = -1
#    for abc in range(int(1000)):
##        print('paritioning ...')
#        idx_train,idx_test = customized_partition(df,label_map,)
#        current_sample = np.sort(idx_test)
#        candidates = [np.sort(item) for item in used_test if (len(item) == len(idx_test))]
#        if any([np.sum(current_sample == item) == len(current_sample) for item in candidates]):
#            pass
#        else:
#            fold += 1
#            used_test.append(idx_test)
#            idxs_train.append(idx_train)
#            idxs_test.append(idx_test)
#            print('done, get fold {}'.format(fold))
#            if fold == n_splits - 1:
#                break
#    return idxs_train,idxs_test
#def partioning_preload(df_name,label_map,
#                       conscious_state = 'unconscious',
#                       n_splits = 100):
#    df_event        = pd.read_csv(df_name)
#    train_test_split = {}
#    idx_unconscious = df_event['visibility'] == conscious_state
#    df_data         = df_event[idx_unconscious].reset_index()
#    df_data['id']   = df_data['session'] * 1000 + df_data['run'] * 100 + df_data['trials']
#    print('partioning {}...'.format(conscious_state))
#    idxs_train,idxs_test = get_train_test_splits(df_data,label_map,n_splits)
#    while check_train_test_splits(idxs_test):
#        idxs_train,idxs_test = get_train_test_splits(df_data,label_map,n_splits)
#    train_test_split['train'],train_test_split['test'] = idxs_train,idxs_test
#    return train_test_split
#
