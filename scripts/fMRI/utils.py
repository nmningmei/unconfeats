#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 10:09:21 2019

@author: nmei
"""
try:
    from autoreject import (AutoReject,get_rejection_threshold)
except Exception as e:
    print(e)
try:
    import mne
except Exception as e:
    print(e)
from glob import glob
import re
import os

import numpy as np
import pandas as pd
import pickle
#import faster # https://gist.github.com/wmvanvliet/d883c3fe1402c7ced6fc

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
# from sklearn.ensemble.forest                       import _generate_unsampled_indices
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
# from sklearn.base                                  import clone
from sklearn.neighbors                             import KNeighborsClassifier
from sklearn.tree                                  import DecisionTreeClassifier
from collections                                   import OrderedDict

from scipy                                         import stats
from collections                                   import Counter
from mpl_toolkits.axes_grid1                       import make_axes_locatable
from matplotlib                                    import pyplot as plt
from matplotlib.pyplot                             import cm
from nilearn.plotting.img_plotting                 import (_load_anat,
                                                           _utils,
                                                           _plot_img_with_bg,
                                                           _get_colorbar_and_data_ranges,
                                                           _safe_get_data)
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
    from tqdm import tqdm
except:
    print('why is tqdm not installed?')

def find_outliers(X, threshold=3.0, max_iter=2):
    """Find outliers based on iterated Z-scoring.
 
    This procedure compares the absolute z-score against the threshold.
    After excluding local outliers, the comparison is repeated until no
    local outlier is present any more.
    
    ########ATTENTION ATTENTION ATTENTION#####
    # This function if removed from MNE-python code base

    Parameters
    ----------
    X : np.ndarray of float, shape (n_elemenets,)
        The scores for which to find outliers.
    threshold : float
        The value above which a feature is classified as outlier.
    max_iter : int
        The maximum number of iterations.
 
    Returns
    -------
    bad_idx : np.ndarray of int, shape (n_features)
        The outlier indices.
    """
    from scipy.stats import zscore
    my_mask = np.zeros(len(X), dtype=np.bool)
    for _ in range(max_iter):
        X = np.ma.masked_array(X, my_mask)
        this_z = np.abs(zscore(X))
        local_bad = this_z > threshold
        my_mask = np.max([my_mask, local_bad], 0)
        if not np.any(local_bad):
            break
 
    bad_idx = np.where(my_mask)[0]
    return bad_idx

def preprocessing_conscious(raw,
                            events,
                            session,
                            tmin = -0,
                            tmax = 1,
                            notch_filter = 50,
                            event_id = {'living':1,'nonliving':2},
                            baseline = (None,None),
                            perform_ICA = False,
                            lowpass = None,
                            interpolate_bad_channels = True,):
    """
    0. re-reference - explicitly
    """
    raw_ref ,_  = mne.set_eeg_reference(raw,
                                        ref_channels     = 'average',
                                        projection       = True,)
    raw_ref.apply_proj() # it might tell you it already has been re-referenced, but do it anyway
    
    # everytime before filtering, explicitly pick the type of channels you want
    # to perform the filters
    picks = mne.pick_types(raw_ref.info,
                           meg = False, # No MEG
                           eeg = True,  # YES EEG
                           eog = perform_ICA,  # depends on ICA
                           )
    # regardless the bandpass filtering later, we should always filter
    # for wire artifacts and their oscillations
    raw_ref.notch_filter(np.arange(notch_filter,241,notch_filter),
                         picks = picks)
    if lowpass is not None:
        raw_ref.filter(None,lowpass,)
    epochs      = mne.Epochs(raw_ref,
                             events,    # numpy array
                             event_id,  # dictionary
                             tmin        = tmin,
                             tmax        = tmax,
                             baseline    = baseline, # range of time for computing the mean references for each channel and subtract these values from all the time points per channel
                             picks       = picks,
                             detrend     = 1, # detrend
                             preload     = True # must be true if we want to do further processing
                             )
    """
    1. if necessary, perform ICA
    """
    if perform_ICA:
        epochs_for_ICA = epochs.copy()
        epochs_for_ICA.filter(1,lowpass)
        picks       = mne.pick_types(epochs.info,
                               eeg          = True, # YES EEG
                               eog          = False # NO EOG
                               )
        if interpolate_bad_channels:
            interpolation_list = faster_bad_channels(epochs,picks=picks)
            for ch_name in interpolation_list:
                epochs_for_ICA.info['bads'].append(ch_name)
                epochs.info['bads'].append(ch_name)
            epochs_for_ICA = epochs_for_ICA.interpolate_bads()
            epochs = epochs.interpolate_bads()
#        ar          = AutoReject(
#                        picks               = picks,
#                        random_state        = 12345,
#                        )
#        ar.fit(epochs)
#        _,reject_log = ar.transform(epochs,return_log=True)
        # calculate the noise covariance of the epochs
        noise_cov   = mne.compute_covariance(epochs_for_ICA,#[~reject_log.bad_epochs],
                                             tmin                   = baseline[0],
                                             tmax                   = baseline[1],
                                             method                 = 'empirical',
                                             rank                   = None,)
        # define an ica function
        ica         = mne.preprocessing.ICA(n_components            = .99,
                                            n_pca_components        = .99,
                                            max_pca_components      = None,
                                            method                  = 'infomax',
                                            max_iter                = int(3e3),
                                            noise_cov               = noise_cov,
                                            random_state            = 12345,)
        picks       = mne.pick_types(epochs_for_ICA.info,
                                     eeg = True, # YES EEG
                                     eog = False # NO EOG
                                     ) 
        ica.fit(epochs_for_ICA,#[~reject_log.bad_epochs],
                picks   = picks,
                start   = tmin,
                stop    = tmax,
                decim   = 3,
                tstep   = 1. # Length of data chunks for artifact rejection in seconds. It only applies if inst is of type Raw.
                )
        # search for artificial ICAs automatically
        # most of these hyperparameters were used in a unrelated published study
        ica.detect_artifacts(epochs_for_ICA,#[~reject_log.bad_epochs],
                             eog_ch         = ['FT9','FT10','TP9','TP10'],
                             eog_criterion  = 0.4, # arbitary choice
                             skew_criterion = 1,   # arbitary choice
                             kurt_criterion = 1,   # arbitary choice
                             var_criterion  = 1,   # arbitary choice
                             )
        picks       = mne.pick_types(epochs_for_ICA.info,
                                     eeg = True, # YES EEG
                                     eog = False # NO EOG
                                     ) 
        epochs_ica  = ica.apply(epochs,#,[~reject_log.bad_epochs],
                                exclude    = ica.exclude,
                                )
        epochs = epochs_ica.copy()
    else:
        picks       = mne.pick_types(epochs.info,
                               eeg          = True, # YES EEG
                               eog          = False # NO EOG
                               )
        if interpolate_bad_channels:
            interpolation_list = faster_bad_channels(epochs,picks=picks)
            for ch_name in interpolation_list:
                epochs.info['bads'].append(ch_name)
            epochs = epochs.interpolate_bads()
    # pick the EEG channels for later use
    clean_epochs = epochs.pick_types(eeg = True, eog = False)
    
    return clean_epochs

def preprocessing_unconscious(raw,
                              events,
                              session,
                              tmin = -0,
                              tmax = 1,
                              notch_filter = 50,
                              event_id = {'living':1,'nonliving':2},
                              baseline = (None,None),
                              perform_ICA = False,
                              eog_chs = [],
                              ecg_chs = [],):
    # everytime before filtering, explicitly pick the type of channels you want
    # to perform the filters
    picks = mne.pick_types(raw.info,
                           meg = True,  # No MEG
                           eeg = False, # NO EEG
                           eog = True,  # YES EOG
                           ecg = True,  # YES ECG
                           )
    # regardless the bandpass filtering later, we should always filter
    # for wire artifacts and their oscillations
    if type(notch_filter) is list:
        for item in notch_filter:
            raw.notch_filter(np.arange(item,301,item),
                                 picks = picks)
    else:
        raw.notch_filter(np.arange(notch_filter,301,notch_filter),
                             picks = picks)
    # filter EOG and ECG channels
    picks = mne.pick_types(raw.info,
                           meg = False,
                           eeg = False,
                           eog = True,
                           ecg = True,)
    raw.filter(1,12,picks = picks,)
    # epoch the data
    picks = mne.pick_types(raw.info,
                           meg = True,
                           eog = True,
                           ecg = True,
                           )
    epochs      = mne.Epochs(raw,
                             events,    # numpy array
                             event_id,  # dictionary
                             tmin        = tmin,
                             tmax        = tmax,
                             baseline    = baseline, # range of time for computing the mean references for each channel and subtract these values from all the time points per channel
                             picks       = picks,
                             detrend     = 1, # detrend
                             preload     = True # must be true if we want to do further processing
                             )
    """
    1. if necessary, perform ICA
    """
    if perform_ICA:
        picks       = mne.pick_types(epochs.info,
                               meg          = True,  # YES MEG
                               eeg          = False, # NO EEG
                               eog          = False, # NO EOG
                               ecg          = False, # NO ECG
                               )
#        ar          = AutoReject(
#                        picks               = picks,
#                        random_state        = 12345,
#                        )
#        ar.fit(epochs)
#        _,reject_log = ar.transform(epochs,return_log=True)
        # calculate the noise covariance of the epochs
        noise_cov   = mne.compute_covariance(epochs,#[~reject_log.bad_epochs],
                                             tmin                   = tmin,
                                             tmax                   = 0,
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
        picks       = mne.pick_types(epochs.info,
                                     eeg = True, # YES EEG
                                     eog = False # NO EOG
                                     ) 
        ica.fit(epochs,#[~reject_log.bad_epochs],
                picks   = picks,
                start   = tmin,
                stop    = tmax,
                decim   = 3,
                tstep   = 1. # Length of data chunks for artifact rejection in seconds. It only applies if inst is of type Raw.
                )
        # search for artificial ICAs automatically
        # most of these hyperparameters were used in a unrelated published study
        ica.detect_artifacts(epochs,#[~reject_log.bad_epochs],
                             eog_ch         = eog_chs,
                             ecg_ch         = ecg_chs[0],
                             eog_criterion  = 0.4, # arbitary choice
                             ecg_criterion  = 0.1, # arbitary choice
                             skew_criterion = 1,   # arbitary choice
                             kurt_criterion = 1,   # arbitary choice
                             var_criterion  = 1,   # arbitary choice
                             )
        epochs_ica  = ica.apply(epochs,#,[~reject_log.bad_epochs],
                                exclude    = ica.exclude,
                                )
        epochs = epochs_ica.copy()
    # pick the EEG channels for later use
    clean_epochs = epochs.pick_types(meg = True, eeg = True, eog = False)
    
    return clean_epochs
def _preprocessing_conscious(
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
                  logging = None,
                  filtering = False,):
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
    if filtering:
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
                        baseline    = (tmin,- (1 / 60 * 20)), # range of time for computing the mean references for each channel and subtract these values from all the time points per channel
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
#        epochs.filter(None,
#                   low_pass,
#                   picks            = picks,
#                   filter_length    = 'auto',    # the filter length is chosen based on the size of the transition regions (6.6 times the reciprocal of the shortest transition band for fir_window=’hamming’ and fir_design=”firwin2”, and half that for “firwin”)
#                   method           = 'fir',     # overlap-add FIR filtering
#                   phase            = 'zero',    # the delay of this filter is compensated for
#                   fir_window       = 'hamming', # The window to use in FIR design
#                   fir_design       = 'firwin2',  # a time-domain design technique that generally gives improved attenuation using fewer samples than “firwin2”
#                   )
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
#        clean_epochs.filter(None,
#                   low_pass,
#                   picks            = picks,
#                   filter_length    = 'auto',    # the filter length is chosen based on the size of the transition regions (6.6 times the reciprocal of the shortest transition band for fir_window=’hamming’ and fir_design=”firwin2”, and half that for “firwin”)
#                   method           = 'fir',     # overlap-add FIR filtering
#                   phase            = 'zero',    # the delay of this filter is compensated for
#                   fir_window       = 'hamming', # The window to use in FIR design
#                   fir_design       = 'firwin2',  # a time-domain design technique that generally gives improved attenuation using fewer samples than “firwin2”
#                   )
        if logging is not None:
            for key in clean_epochs.event_id.keys():
                evoked = epochs[key].average()
                fig_ = evoked.plot_joint(title = key) 
                fig_.savefig(logging.replace('.png',f'_{key}_post.png'),
                             bbox_inches = 'tight')
                plt.close('all')
        return clean_epochs
def plot_temporal_decoding(times,
                           scores,
                           frames,
                           ii,
                           conscious_state,
                           plscores,
                           n_splits,
                           ylim = (0.2,0.8)):
    scores_mean = scores.mean(0)
    scores_se   = scores.std(0) / np.sqrt(n_splits)
    fig,ax = plt.subplots(figsize = (16,8))
    ax.plot(times,scores_mean,
            color = 'k',
            alpha = .9,
            label = f'Average across {n_splits} folds',
            )
    ax.fill_between(times,
                    scores_mean + scores_se,
                    scores_mean - scores_se,
                    color = 'red',
                    alpha = 0.4,
                    label = 'Standard Error',)
    ax.axhline(0.5,
               linestyle    = '--',
               color        = 'k',
               alpha        = 0.7,
               label        = 'Chance level')
    ax.axvline(0,
               linestyle    = '--',
               color        = 'blue',
               alpha        = 0.7,
               label        = 'Probe onset',)
    if ii is not None:
        ax.axvspan(frames[ii][1] * (1 / 100) - frames[ii][2] * (1 / 100),
                   frames[ii][1] * (1 / 100) + frames[ii][2] * (1 / 100),
                   color        = 'blue',
                   alpha        = 0.3,
                   label        = 'probe offset ave +/- std',)
    ax.set(xlim     = (times.min(),
                       times.max()),
           ylim     = ylim,#(0.4,0.6),
           title    = f'Temporal decoding of {conscious_state} = {plscores.mean():.3f}+/-{plscores.std():.3f}',
           )
    ax.legend()
    return fig,ax
def plot_temporal_generalization(scores_gen_,
                                 times,
                                 ii,
                                 conscious_state,
                                 frames,
                                 vmin = 0.4,
                                 vmax = 0.6):
    fig, ax = plt.subplots(figsize = (10,10))
    if len(scores_gen_.shape) > 2:
        scores_gen_ = scores_gen_.mean(0)
    im      = ax.imshow(
                        scores_gen_, 
                        interpolation       = 'hamming', 
                        origin              = 'lower', 
                        cmap                = 'RdBu_r',
                        extent              = times[[0, -1, 0, -1]], 
                        vmin                = vmin, 
                        vmax                = vmax,
                        )
    ax.set_xlabel('Testing Time (s)')
    ax.set_ylabel('Training Time (s)')
    ax.set_title(f'Temporal generalization of {conscious_state}')
    ax.axhline(0.,
               linestyle                    = '--',
               color                        = 'black',
               alpha                        = 0.7,
               label                        = 'Probe onset',)
    ax.axvline(0.,
               linestyle                    = '--',
               color                        = 'black',
               alpha                        = 0.7,
               )
    if ii is not None:
        ax.axhspan(frames[ii][1] * (1 / 100) - frames[ii][2] * (1 / 100),
                   frames[ii][1] * (1 / 100) + frames[ii][2] * (1 / 100),
                   color                        = 'black',
                   alpha                        = 0.2,
                   label                        = 'probe offset ave +/- std',)
        ax.axvspan(frames[ii][1] * (1 / 100) - frames[ii][2] * (1 / 100),
                   frames[ii][1] * (1 / 100) + frames[ii][2] * (1 / 100),
                   color                        = 'black',
                   alpha                        = 0.2,
                   )
    plt.colorbar(im, ax = ax)
    ax.legend()
    return fig,ax

def plot_t_stats(T_obs,
                 clusters,
                 cluster_p_values,
                 times,
                 ii,
                 conscious_state,
                 frames,):
    
    # since the p values of each cluster is corrected for multiple comparison, 
    # we could directly use 0.05 as the threshold to filter clusters
    T_obs_plot              = 0 * np.ones_like(T_obs)
    k = np.array([np.sum(c) for c in clusters])
    if np.max(k) > 1000:
        c_thresh = 1000
    elif 1000 > np.max(k) > 500:
        c_thresh = 500
    elif 500 > np.max(k) > 100:
        c_thresh = 100
    elif 100 > np.max(k) > 10:
        c_thresh = 10
    else:
        c_thresh = 0
    for c, p_val in zip(clusters, cluster_p_values):
        if (p_val <= 0.01) and (np.sum(c) >= c_thresh):# and (distance.cdist(np.where(c ==  True)[0].reshape(1,-1),np.where(c ==  True)[1].reshape(1,-1))[0][0] < 200):# and (np.sum(c) >= c_thresh):
            T_obs_plot[c]   = T_obs[c]
    # defind the range of the colorbar
    vmax = np.max(np.abs(T_obs))
    vmin = -vmax# - 2 * t_threshold
    plt.close('all')
    fig,ax = plt.subplots(figsize=(10,10))
    im      = ax.imshow(T_obs_plot,
                   origin                   = 'lower',
                   cmap                     = plt.cm.RdBu_r,# to emphasize the clusters
                   extent                   = times[[0, -1, 0, -1]],
                   vmin                     = vmin,
                   vmax                     = vmax,
                   interpolation            = 'lanczos',
                   )
    divider = make_axes_locatable(ax)
    cax     = divider.append_axes("right", 
                                  size      = "5%", 
                                  pad       = 0.2)
    cb      = plt.colorbar(im, 
                           cax              = cax,
                           ticks            = np.linspace(vmin,vmax,3))
    cb.ax.set(title = 'T Statistics')
    ax.plot([times[0],times[-1]],[times[0],times[-1]],
            linestyle                    = '--',
            color                        = 'black',
            alpha                        = 0.7,
            )
    ax.axhline(0.,
               linestyle                    = '--',
               color                        = 'black',
               alpha                        = 0.7,
               label                        = 'Probe onset',)
    ax.axvline(0.,
               linestyle                    = '--',
               color                        = 'black',
               alpha                        = 0.7,
               )
    if ii is not None:
        ax.axhspan(frames[ii][1] * (1 / 100) - frames[ii][2] * (1 / 100),
                   frames[ii][1] * (1 / 100) + frames[ii][2] * (1 / 100),
                   color                        = 'black',
                   alpha                        = 0.2,
                   label                        = 'probe offset ave +/- std',)
        ax.axvspan(frames[ii][1] * (1 / 100) - frames[ii][2] * (1 / 100),
                   frames[ii][1] * (1 / 100) + frames[ii][2] * (1 / 100),
                   color                        = 'black',
                   alpha                        = 0.2,
                   )
    ax.set(xlabel                           = 'Test time',
           ylabel                           = 'Train time',
           title                            = f'nonparametric t test of {conscious_state}')
    ax.legend()
    return fig,ax
def plot_p_values(times,
                  clusters,
                  cluster_p_values,
                  ii,
                  conscious_state,
                  frames):
    width = len(times)
    p_clust = np.ones((width, width))# * np.nan
    k = np.array([np.sum(c) for c in clusters])
    if np.max(k) > 1000:
        c_thresh = 1000
    elif 1000 > np.max(k) > 500:
        c_thresh = 500
    elif 500 > np.max(k) > 100:
        c_thresh = 100
    elif 100 > np.max(k) > 10:
        c_thresh = 10
    else:
        c_thresh = 0
    for c, p_val in zip(clusters, cluster_p_values):
        if (np.sum(c) >= c_thresh):
            p_val_ = p_val.copy()
            if p_val_ > 0.05:
                p_val_ = 1.
            p_clust[c] = p_val_
    
    # defind the range of the colorbar
    vmax = 1.
    vmin = 0.
    plt.close('all')
    fig,ax = plt.subplots(figsize = (10,10))
    im      = ax.imshow(p_clust,
                   origin                   = 'lower',
                   cmap                     = plt.cm.RdBu_r,# to emphasize the clusters
                   extent                   = times[[0, -1, 0, -1]],
                   vmin                     = vmin,
                   vmax                     = vmax,
                   interpolation            = 'hanning',
                   )
    divider = make_axes_locatable(ax)
    cax     = divider.append_axes("right", 
                                  size      = "5%", 
                                  pad       = 0.2)
    cb      = plt.colorbar(im, 
                           cax              = cax,
                           ticks            = [0,0.05,1])
    cb.ax.set(title = 'P values')
    ax.plot([times[0],times[-1]],[times[0],times[-1]],
            linestyle                       = '--',
            color                           = 'black',
            alpha                           = 0.7,
            )
    ax.axhline(0.,
               linestyle                    = '--',
               color                        = 'black',
               alpha                        = 0.7,
               label                        = 'Probe onset',)
    ax.axvline(0.,
               linestyle                    = '--',
               color                        = 'black',
               alpha                        = 0.7,
               )
    if ii is not None:
        ax.axhspan(frames[ii][1] * (1 / 100) - frames[ii][2] * (1 / 100),
                   frames[ii][1] * (1 / 100) + frames[ii][2] * (1 / 100),
                   color                        = 'black',
                   alpha                        = 0.2,
                   label                        = 'probe offset ave +/- std',)
        ax.axvspan(frames[ii][1] * (1 / 100) - frames[ii][2] * (1 / 100),
                   frames[ii][1] * (1 / 100) + frames[ii][2] * (1 / 100),
                   color                        = 'black',
                   alpha                        = 0.2,
                   )
    ax.set(xlabel                           = 'Test time',
           ylabel                           = 'Train time',
           title                            = f'p value map of {conscious_state}')
    ax.legend()
    return fig,ax
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
    if type(x) is str:
        return float(re.findall(r'\d+',x)[0])
    else:
        return x
def compute_A(h,f):
    if (.5 >= f) and (h >= .5):
        a = .75 + (h - f) / 4 - f * (1 - h)
    elif (h >= f) and (.5 >= h):
        a = .75 + (h - f) / 4 - f / (4 * h)
    else:
        a = .75 + (h - f) / 4 - (1 - h) / (4 * (1 - f))
    return a

def check_nan(temp):
    if np.isnan(temp[1]):
        return 0
    else:
        return temp[1]
def simple_load(f,idx):
    temp = f.split('/')
    sub = temp[-3]
    session = temp[-2]
    run = temp[-1].split('_')[-1].split('.')[0]
    df = pd.read_csv(f).dropna()
    df['sub'] = sub
    df['run'] = run
    df['session'] = session
    return df
def get_frames(directory,new = True,EEG = True):
    if EEG:
        files = glob(os.path.join(directory,'*trials.csv'))
#    elif EEG == 'fMRI':
#        files = glob(os.path.join(directory,'*trials.csv'))
    else:
        files = glob(os.path.join(directory,'*','*.csv'))
    df_stat = dict(
        conscious_state = [],
        prob_press_1 = [],
        prob_press_2 = [],
        correct = [],
        frame_mean = [],
        frame_std = [],
        RT_mean = [],
        RT_std = [],
        )
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
    try:
        for col in ['probeFrames_raw',
                    'response.keys_raw',
                    'visible.keys_raw']:
#            print(df[col])
            df[col] = df[col].apply(str2int)
            
    except:
        for col in ['probe_Frames_raw',
                    'response.keys_raw',
                    'visible.keys_raw']:
#            print(df[col])
            df[col] = df[col].apply(str2int)
            
        df["probeFrames_raw"] = df["probe_Frames_raw"]
    df = df[df['probeFrames_raw'] != 999]
    df = df.sort_values(['run','order'])
    
    for vis,df_sub in df.groupby(['visible.keys_raw']):
        df_press1 = df_sub[df_sub['response.keys_raw'] == 1]
        df_press2 = df_sub[df_sub['response.keys_raw'] == 2]
        prob1 = df_press1.shape[0] / df_sub.shape[0]
        prob2 = df_press2.shape[0] / df_sub.shape[0]
        
        df_stat['conscious_state'].append(vis)
        df_stat['prob_press_1'].append(prob1)
        df_stat['prob_press_2'].append(prob2)
        
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
        for kk,f in enumerate(files):
            temp = simple_load(f,kk).dropna()
            try:
                temp[['probeFrames_raw','visible.keys_raw']]
            except:
                temp['probeFrames_raw'] = temp['probe_Frames_raw']
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
    df = df[df['probeFrames'] != 999]
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
        
        df_stat['frame_mean'].append(np.mean(df_sub['probeFrames']))
        df_stat['frame_std'].append(np.std(df_sub['probeFrames']))
        df_stat['correct'].append(corrects)
        df_stat['RT_mean'].append(np.mean(df_sub['visible.rt_raw']))
        df_stat['RT_std'].append(np.std(df_sub['visible.rt_raw']))
    return results,empty_temp,pd.DataFrame(df_stat),df

def subj_map():
    temp = {'sub-01':'sub-01',
            'sub-03':'sub-02',
            'sub-04':'sub-03',
            'sub-05':'sub-04',
            'sub-02':'sub-05',
            'sub-06':'sub-06',
            'sub-07':'sub-07',}
    return temp


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
    if len(df_sub.index.values) > 1:
        temp = '+'.join(str(item + 10) for item in df_sub.index.values)
    else:
        temp = str(df_sub.index.values[0])
    df_sub = df_sub.iloc[0,:].to_frame().T # why did I use 1 instead of 0?
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
    """
    Works with fsl-5.0.9 and fsl-5.0.11, but not fsl-6.0.0
    """
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
                                 example_func,
                                 standard_brain,
                                 standard_head,
                                 standard_mask,
                                 workflow_name = 'registration',
                                 output_dir = 'temp'):
    from nipype.interfaces          import fsl
    from nipype.interfaces         import utility as util
    from nipype.pipeline           import engine as pe
    fsl.FSLCommand.set_default_output_type('NIFTI_GZ')
    registration                    = pe.Workflow(name = 'registration')
    inputnode                       = pe.Node(
                                        interface   = util.IdentityInterface(
                                        fields      = [
                                                'highres', # anat_brain
                                                'highres_head', # anat_head
                                                'example_func',
                                                'standard', # standard_brain
                                                'standard_head',
                                                'standard_mask'
                                                ]),
                                        name        = 'inputspec')
    outputnode                      = pe.Node(
                                    interface   = util.IdentityInterface(
                                    fields      = ['example_func2highres_nii_gz',
                                                   'example_func2highres_mat',
                                                   'linear_example_func2highres_log',
                                                   'highres2example_func_mat',
                                                   'highres2standard_linear_nii_gz',
                                                   'highres2standard_mat',
                                                   'linear_highres2standard_log',
                                                   'highres2standard_nii_gz',
                                                   'highres2standard_warp_nii_gz',
                                                   'highres2standard_head_nii_gz',
    #                                               'highres2standard_apply_warp_nii_gz',
                                                   'highres2highres_jac_nii_gz',
                                                   'nonlinear_highres2standard_log',
                                                   'highres2standard_nii_gz',
                                                   'standard2highres_mat',
                                                   'example_func2standard_mat',
                                                   'example_func2standard_warp_nii_gz',
                                                   'example_func2standard_nii_gz',
                                                   'standard2example_func_mat',
                                                   ]),
                                    name        = 'outputspec')
    """
    fslmaths /bcbl/home/public/Consciousness/uncon_feat/data/MRI/sub-01/anat/sub-01-T1W_mprage_sag_p2_1iso_MGH_day_6_nipy_brain highres
    fslmaths /bcbl/home/public/Consciousness/uncon_feat/data/MRI/sub-01/anat/sub-01-T1W_mprage_sag_p2_1iso_MGH_day_6_nipy_brain  highres_head
    fslmaths /opt/fsl/fsl-5.0.9/fsl/data/standard/MNI152_T1_2mm_brain standard
    fslmaths /opt/fsl/fsl-5.0.9/fsl/data/standard/MNI152_T1_2mm standard_head
    fslmaths /opt/fsl/fsl-5.0.9/fsl/data/standard/MNI152_T1_2mm_brain_mask_dil standard_mask
    """
    # skip
    
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
    linear_example_func2highres = pe.MapNode(
            interface   = fsl.FLIRT(cost = 'corratio',
                                    interp = 'trilinear',
                                    dof = 7,
                                    save_log = True,
                                    searchr_x = [-180, 180],
                                    searchr_y = [-180, 180],
                                    searchr_z = [-180, 180],),
            iterfield   = ['in_file','reference'],
            name        = 'linear_example_func2highres')
    registration.connect(inputnode, 'example_func',
                         linear_example_func2highres, 'in_file')
    registration.connect(inputnode, 'highres',
                         linear_example_func2highres, 'reference')
    registration.connect(linear_example_func2highres, 'out_file',
                         outputnode, 'example_func2highres_nii_gz')
    registration.connect(linear_example_func2highres, 'out_matrix_file',
                         outputnode, 'example_func2highres_mat')
    registration.connect(linear_example_func2highres, 'out_log',
                         outputnode, 'linear_example_func2highres_log')
    
    """
    /opt/fsl/fsl-5.0.10/fsl/bin/convert_xfm 
        -inverse -omat highres2example_func.mat example_func2highres.mat
    """
    get_highres2example_func = pe.MapNode(
            interface = fsl.ConvertXFM(invert_xfm = True),
            iterfield = ['in_file'],
            name = 'get_highres2example_func')
    registration.connect(linear_example_func2highres,'out_matrix_file',
                         get_highres2example_func,'in_file')
    registration.connect(get_highres2example_func,'out_file',
                         outputnode,'highres2example_func_mat')
    
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
    linear_highres2standard = pe.MapNode(
            interface = fsl.FLIRT(cost = 'corratio',
                                interp = 'trilinear',
                                dof = 12,
                                save_log = True,
                                searchr_x = [-180, 180],
                                searchr_y = [-180, 180],
                                searchr_z = [-180, 180],),
            iterfield = ['in_file','reference'],
            name = 'linear_highres2standard')
    registration.connect(inputnode,'highres',
                         linear_highres2standard,'in_file')
    registration.connect(inputnode,'standard',
                         linear_highres2standard,'reference',)
    registration.connect(linear_highres2standard,'out_file',
                         outputnode,'highres2standard_linear_nii_gz')
    registration.connect(linear_highres2standard,'out_matrix_file',
                         outputnode,'highres2standard_mat')
    registration.connect(linear_highres2standard,'out_log',
                         outputnode,'linear_highres2standard_log')
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
    nonlinear_highres2standard = pe.MapNode(
            interface = fsl.FNIRT(warp_resolution = (10,10,10),
                                  config_file = "T1_2_MNI152_2mm"),
            iterfield = ['in_file','ref_file','affine_file','refmask_file'],
            name = 'nonlinear_highres2standard')
    # -- iout
    registration.connect(nonlinear_highres2standard,'warped_file',
                         outputnode,'highres2standard_head_nii_gz')
    # --in
    registration.connect(inputnode,'highres',
                         nonlinear_highres2standard,'in_file')
    # --aff
    registration.connect(linear_highres2standard,'out_matrix_file',
                         nonlinear_highres2standard,'affine_file')
    # --cout
    registration.connect(nonlinear_highres2standard,'fieldcoeff_file',
                         outputnode,'highres2standard_warp_nii_gz')
    # --jout
    registration.connect(nonlinear_highres2standard,'jacobian_file',
                         outputnode,'highres2highres_jac_nii_gz')
    # --ref
    registration.connect(inputnode,'standard_head',
                         nonlinear_highres2standard,'ref_file',)
    # --refmask
    registration.connect(inputnode,'standard_mask',
                         nonlinear_highres2standard,'refmask_file')
    # log
    registration.connect(nonlinear_highres2standard,'log_file',
                         outputnode,'nonlinear_highres2standard_log')
    """
    /opt/fsl/fsl-5.0.10/fsl/bin/applywarp 
        -i highres 
        -r standard 
        -o highres2standard 
        -w highres2standard_warp
    """
    warp_highres2standard = pe.MapNode(
            interface = fsl.ApplyWarp(),
            iterfield = ['in_file','ref_file','field_file'],
            name = 'warp_highres2standard')
    registration.connect(inputnode,'highres',
                         warp_highres2standard,'in_file')
    registration.connect(inputnode,'standard',
                         warp_highres2standard,'ref_file')
    registration.connect(warp_highres2standard,'out_file',
                         outputnode,'highres2standard_nii_gz')
    registration.connect(nonlinear_highres2standard,'fieldcoeff_file',
                         warp_highres2standard,'field_file')
    """
    /opt/fsl/fsl-5.0.10/fsl/bin/convert_xfm 
        -inverse -omat standard2highres.mat highres2standard.mat
    """
    get_standard2highres = pe.MapNode(
            interface = fsl.ConvertXFM(invert_xfm = True),
            iterfield = ['in_file'],
            name = 'get_standard2highres')
    registration.connect(linear_highres2standard,'out_matrix_file',
                         get_standard2highres,'in_file')
    registration.connect(get_standard2highres,'out_file',
                         outputnode,'standard2highres_mat')
    """
    /opt/fsl/fsl-5.0.10/fsl/bin/convert_xfm 
        -omat example_func2standard.mat -concat highres2standard.mat example_func2highres.mat
    """
    get_exmaple_func2standard = pe.MapNode(
            interface = fsl.ConvertXFM(concat_xfm = True),
            iterfield = ['in_file','in_file2'],
            name = 'get_exmaple_func2standard')
    registration.connect(linear_example_func2highres, 'out_matrix_file',
                         get_exmaple_func2standard,'in_file')
    registration.connect(linear_highres2standard,'out_matrix_file',
                         get_exmaple_func2standard,'in_file2')
    registration.connect(get_exmaple_func2standard,'out_file',
                         outputnode,'example_func2standard_mat')
    """
    /opt/fsl/fsl-5.0.10/fsl/bin/convertwarp 
        --ref=standard 
        --premat=example_func2highres.mat 
        --warp1=highres2standard_warp 
        --out=example_func2standard_warp
    """
    convertwarp_example2standard = pe.MapNode(
            interface = fsl.ConvertWarp(),
            iterfield = ['reference','premat','warp1'],
            name = 'convertwarp_example2standard')
    registration.connect(inputnode,'standard',
                         convertwarp_example2standard,'reference')
    registration.connect(linear_example_func2highres,'out_matrix_file',
                         convertwarp_example2standard,'premat')
    registration.connect(nonlinear_highres2standard,'fieldcoeff_file',
                         convertwarp_example2standard,'warp1')
    registration.connect(convertwarp_example2standard,'out_file',
                         outputnode,'example_func2standard_warp_nii_gz')
    """
    /opt/fsl/fsl-5.0.10/fsl/bin/applywarp 
        --ref=standard 
        --in=example_func 
        --out=example_func2standard 
        --warp=example_func2standard_warp
    """
    warp_example2stand = pe.MapNode(
            interface = fsl.ApplyWarp(),
            iterfield = ['ref_file','in_file','field_file'],
            name = 'warp_example2stand')
    registration.connect(inputnode,'standard',
                         warp_example2stand,'ref_file')
    registration.connect(inputnode,'example_func',
                         warp_example2stand,'in_file')
    registration.connect(warp_example2stand,'out_file',
                         outputnode,'example_func2standard_nii_gz')
    registration.connect(convertwarp_example2standard,'out_file',
                         warp_example2stand,'field_file')
    """
    /opt/fsl/fsl-5.0.10/fsl/bin/convert_xfm 
        -inverse -omat standard2example_func.mat example_func2standard.mat
    """
    get_standard2example_func = pe.MapNode(
            interface = fsl.ConvertXFM(invert_xfm = True),
            iterfield = ['in_file'],
            name = 'get_standard2example_func')
    registration.connect(get_exmaple_func2standard,'out_file',
                         get_standard2example_func,'in_file')
    registration.connect(get_standard2example_func,'out_file',
                         outputnode,'standard2example_func_mat')
    
    registration.base_dir = output_dir
    
    registration.inputs.inputspec.highres = anat_brain
    registration.inputs.inputspec.highres_head= anat_head
    registration.inputs.inputspec.example_func = example_func
    registration.inputs.inputspec.standard = standard_brain
    registration.inputs.inputspec.standard_head = standard_head
    registration.inputs.inputspec.standard_mask = standard_mask
    
    # define all the oupput file names with the directory
    registration.inputs.linear_example_func2highres.out_file          = os.path.abspath(os.path.join(output_dir,
                            'example_func2highres.nii.gz'))
    registration.inputs.linear_example_func2highres.out_matrix_file   = os.path.abspath(os.path.join(output_dir,
                            'example_func2highres.mat'))
    registration.inputs.linear_example_func2highres.out_log           = os.path.abspath(os.path.join(output_dir,
                            'linear_example_func2highres.log'))
    registration.inputs.get_highres2example_func.out_file        = os.path.abspath(os.path.join(output_dir,
                            'highres2example_func.mat'))
    registration.inputs.linear_highres2standard.out_file         = os.path.abspath(os.path.join(output_dir,
                            'highres2standard_linear.nii.gz'))
    registration.inputs.linear_highres2standard.out_matrix_file  = os.path.abspath(os.path.join(output_dir,
                            'highres2standard.mat'))
    registration.inputs.linear_highres2standard.out_log          = os.path.abspath(os.path.join(output_dir,
                            'linear_highres2standard.log'))
    # --iout
    registration.inputs.nonlinear_highres2standard.warped_file  = os.path.abspath(os.path.join(output_dir,
                            'highres2standard.nii.gz'))
    # --cout
    registration.inputs.nonlinear_highres2standard.fieldcoeff_file    = os.path.abspath(os.path.join(output_dir,
                            'highres2standard_warp.nii.gz'))
    # --jout
    registration.inputs.nonlinear_highres2standard.jacobian_file      = os.path.abspath(os.path.join(output_dir,
                            'highres2highres_jac.nii.gz'))
    registration.inputs.nonlinear_highres2standard.log_file           = os.path.abspath(os.path.join(output_dir,
                            'nonlinear_highres2standard.log'))
    registration.inputs.warp_highres2standard.out_file                = os.path.abspath(os.path.join(output_dir,
                            'highres2standard.nii.gz'))
    registration.inputs.get_standard2highres.out_file       = os.path.abspath(os.path.join(output_dir,
                            'standard2highres.mat'))
    registration.inputs.get_exmaple_func2standard.out_file               = os.path.abspath(os.path.join(output_dir,
                            'example_func2standard.mat'))
    registration.inputs.convertwarp_example2standard.out_file     = os.path.abspath(os.path.join(output_dir,
                            'example_func2standard_warp.nii.gz'))
    registration.inputs.warp_example2stand.out_file       = os.path.abspath(os.path.join(output_dir,
                            'example_func2standard.nii.gz'))
    registration.inputs.get_standard2example_func.out_file       = os.path.abspath(os.path.join(output_dir,
                            'standard2example_func.mat'))
    return registration

def _create_registration_workflow(anat_brain,
                                 anat_head,
                                 func_ref,
                                 standard_brain,
                                 standard_head,
                                 standard_mask,
                                 output_dir = 'temp'):
    from nipype.interfaces          import fsl
    """
    fslmaths /bcbl/home/public/Consciousness/uncon_feat/data/MRI/sub-01/anat/sub-01-T1W_mprage_sag_p2_1iso_MGH_day_6_nipy_brain highres
    fslmaths /bcbl/home/public/Consciousness/uncon_feat/data/MRI/sub-01/anat/sub-01-T1W_mprage_sag_p2_1iso_MGH_day_6_nipy_brain  highres_head
    fslmaths /opt/fsl/fsl-5.0.9/fsl/data/standard/MNI152_T1_2mm_brain standard
    fslmaths /opt/fsl/fsl-5.0.9/fsl/data/standard/MNI152_T1_2mm standard_head
    fslmaths /opt/fsl/fsl-5.0.9/fsl/data/standard/MNI152_T1_2mm_brain_mask_dil standard_mask
    
    """
    fslmaths = fsl.ImageMaths()
    fslmaths.inputs.in_file = anat_brain
    fslmaths.inputs.out_file = os.path.abspath(os.path.join(output_dir,'highres.nii.gz'))
    fslmaths.cmdline
    fslmaths.run()
    
    fslmaths = fsl.ImageMaths()
    fslmaths.inputs.in_file = anat_head
    fslmaths.inputs.out_file = os.path.abspath(os.path.join(output_dir,'highres_head.nii.gz'))
    fslmaths.cmdline
    fslmaths.run()
    
    fslmaths = fsl.ImageMaths()
    fslmaths.inputs.in_file = standard_brain
    fslmaths.inputs.out_file = os.path.abspath(os.path.join(output_dir,'standard.nii.gz'))
    fslmaths.cmdline
    fslmaths.run()
    
    fslmaths = fsl.ImageMaths()
    fslmaths.inputs.in_file = standard_head
    fslmaths.inputs.out_file = os.path.abspath(os.path.join(output_dir,'standard_head.nii.gz'))
    fslmaths.cmdline
    fslmaths.run()
    
    fslmaths = fsl.ImageMaths()
    fslmaths.inputs.in_file = standard_mask
    fslmaths.inputs.out_file = os.path.abspath(os.path.join(output_dir,'standard_mask.nii.gz'))
    fslmaths.cmdline
    fslmaths.run()
    
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
    flt = fsl.FLIRT()
    flt.inputs.in_file = func_ref
    flt.inputs.reference = anat_brain
    flt.inputs.out_file = os.path.abspath(os.path.join(output_dir,'example_func2highres.nii.gz'))
    flt.inputs.out_matrix_file = os.path.abspath(os.path.join(output_dir,'example_func2highres.mat'))
    flt.inputs.out_log = os.path.abspath(os.path.join(output_dir,'example_func2highres.log'))
    flt.inputs.cost = 'corratio'
    flt.inputs.interp = 'trilinear'
    flt.inputs.searchr_x = [-180, 180]
    flt.inputs.searchr_y = [-180, 180]
    flt.inputs.searchr_z = [-180, 180]
    flt.inputs.dof = 7
    flt.inputs.save_log = True
    flt.cmdline
    flt.run()
    
    """
    /opt/fsl/fsl-5.0.10/fsl/bin/convert_xfm 
        -inverse -omat highres2example_func.mat example_func2highres.mat
    """
    inverse_transformer = fsl.ConvertXFM()
    inverse_transformer.inputs.in_file = os.path.abspath(os.path.join(output_dir,"example_func2highres.mat"))
    inverse_transformer.inputs.invert_xfm = True
    inverse_transformer.inputs.out_file = os.path.abspath(os.path.join(output_dir,'highres2example_func.mat'))
    inverse_transformer.cmdline
    inverse_transformer.run()
    
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
    flt = fsl.FLIRT()
    flt.inputs.in_file = anat_brain
    flt.inputs.reference = standard_brain
    flt.inputs.out_file = os.path.abspath(os.path.join(output_dir,'highres2standard_linear.nii.gz'))
    flt.inputs.out_matrix_file = os.path.abspath(os.path.join(output_dir,'highres2standard.mat'))
    flt.inputs.out_log = os.path.abspath(os.path.join(output_dir,'highres2standard.log'))
    flt.inputs.cost = 'corratio'
    flt.inputs.interp = 'trilinear'
    flt.inputs.searchr_x = [-180, 180]
    flt.inputs.searchr_y = [-180, 180]
    flt.inputs.searchr_z = [-180, 180]
    flt.inputs.dof = 12
    flt.inputs.save_log = True
    flt.cmdline
    flt.run()
    
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
    
    fnirt_mprage = fsl.FNIRT()
    fnirt_mprage.inputs.warp_resolution = (10, 10, 10)
    # --iout name of output image
    fnirt_mprage.inputs.warped_file = os.path.abspath(os.path.join(output_dir,
                                                                 'highres2standard.nii.gz'))
    # --in input image
    fnirt_mprage.inputs.in_file = anat_head
    # --aff affine transform
    fnirt_mprage.inputs.affine_file = os.path.abspath(os.path.join(output_dir,
                                                                   'highres2standard.mat'))
    # --cout output file with field coefficients
    fnirt_mprage.inputs.fieldcoeff_file = os.path.abspath(os.path.join(output_dir,
                                                                       'highres2standard_warp.nii.gz'))
    # --jout
    fnirt_mprage.inputs.jacobian_file = os.path.abspath(os.path.join(output_dir,
                                                                     'highres2highres_jac.nii.gz'))
    # --config
    fnirt_mprage.inputs.config_file = 'T1_2_MNI152_2mm'
    # --ref
    fnirt_mprage.inputs.ref_file = os.path.abspath(standard_head)
    # --refmask
    fnirt_mprage.inputs.refmask_file = os.path.abspath(standard_mask)
    # --warpres
    fnirt_mprage.inputs.log_file = os.path.abspath(os.path.join(output_dir,
                                                                'highres2standard.log'))
    fnirt_mprage.cmdline
    fnirt_mprage.run()
    
    """
    /opt/fsl/fsl-5.0.10/fsl/bin/applywarp 
        -i highres 
        -r standard 
        -o highres2standard 
        -w highres2standard_warp
    """
    aw = fsl.ApplyWarp()
    aw.inputs.in_file = anat_brain
    aw.inputs.ref_file = os.path.abspath(standard_brain)
    aw.inputs.out_file = os.path.abspath(os.path.join(output_dir,
                                                      'highres2standard.nii.gz'))
    aw.inputs.field_file = os.path.abspath(os.path.join(output_dir,
                                                        'highres2standard_warp.nii.gz'))
    aw.cmdline
    aw.run()
    
    """
    /opt/fsl/fsl-5.0.10/fsl/bin/convert_xfm 
        -inverse -omat standard2highres.mat highres2standard.mat
    """
    inverse_transformer = fsl.ConvertXFM()
    inverse_transformer.inputs.in_file = os.path.abspath(os.path.join(output_dir,"highres2standard.mat"))
    inverse_transformer.inputs.invert_xfm = True
    inverse_transformer.inputs.out_file = os.path.abspath(os.path.join(output_dir,'standard2highres.mat'))
    inverse_transformer.cmdline
    inverse_transformer.run()
    
    """
    /opt/fsl/fsl-5.0.10/fsl/bin/convert_xfm 
        -omat example_func2standard.mat -concat highres2standard.mat example_func2highres.mat
    """
    inverse_transformer = fsl.ConvertXFM()
    inverse_transformer.inputs.in_file2 = os.path.abspath(os.path.join(output_dir,"highres2standard.mat"))
    inverse_transformer.inputs.in_file = os.path.abspath(os.path.join(output_dir,
                                                                       "example_func2highres.mat"))
    inverse_transformer.inputs.concat_xfm = True
    inverse_transformer.inputs.out_file = os.path.abspath(os.path.join(output_dir,'example_func2standard.mat'))
    inverse_transformer.cmdline
    inverse_transformer.run()
    
    """
    /opt/fsl/fsl-5.0.10/fsl/bin/convertwarp 
        --ref=standard 
        --premat=example_func2highres.mat 
        --warp1=highres2standard_warp 
        --out=example_func2standard_warp
    """
    warputils = fsl.ConvertWarp()
    warputils.inputs.reference = os.path.abspath(standard_brain)
    warputils.inputs.premat = os.path.abspath(os.path.join(output_dir,
                                                           "example_func2highres.mat"))
    warputils.inputs.warp1 = os.path.abspath(os.path.join(output_dir,
                                                          "highres2standard_warp.nii.gz"))
    warputils.inputs.out_file = os.path.abspath(os.path.join(output_dir,
                                                             "example_func2standard_warp.nii.gz"))
    warputils.cmdline
    warputils.run()
    
    """
    /opt/fsl/fsl-5.0.10/fsl/bin/applywarp 
        --ref=standard 
        --in=example_func 
        --out=example_func2standard 
        --warp=example_func2standard_warp
    """
    aw = fsl.ApplyWarp()
    aw.inputs.ref_file = os.path.abspath(standard_brain)
    aw.inputs.in_file = os.path.abspath(func_ref)
    aw.inputs.out_file = os.path.abspath(os.path.join(output_dir,
                                                      "example_func2standard.nii.gz"))
    aw.inputs.field_file = os.path.abspath(os.path.join(output_dir,
                                                        "example_func2standard_warp.nii.gz"))
    aw.run()
    """
    /opt/fsl/fsl-5.0.10/fsl/bin/convert_xfm 
        -inverse -omat standard2example_func.mat example_func2standard.mat
    """
    inverse_transformer = fsl.ConvertXFM()
    inverse_transformer.inputs.in_file = os.path.abspath(os.path.join(output_dir,
                                                               "example_func2standard.mat"))
    inverse_transformer.inputs.out_file = os.path.abspath(os.path.join(output_dir,
                                                                "standard2example_func.mat"))
    inverse_transformer.inputs.invert_xfm = True
    inverse_transformer.cmdline
    inverse_transformer.run()
    ######################
    ###### plotting ######
    example_func2highres = os.path.abspath(os.path.join(output_dir,
                                                        'example_func2highres'))
    example_func2standard = os.path.abspath(os.path.join(output_dir,
                                                         "example_func2standard"))
    highres2standard = os.path.abspath(os.path.join(output_dir,
                                                    'highres2standard'))
    highres = os.path.abspath(anat_brain)
    standard = os.path.abspath(standard_brain)
    
    plot_example_func2highres = f"""
    /opt/fsl/fsl-5.0.10/fsl/bin/slicer {example_func2highres} {highres} -s 2 -x 0.35 sla.png -x 0.45 slb.png -x 0.55 slc.png -x 0.65 sld.png -y 0.35 sle.png -y 0.45 slf.png -y 0.55 slg.png -y 0.65 slh.png -z 0.35 sli.png -z 0.45 slj.png -z 0.55 slk.png -z 0.65 sll.png ; 
    /opt/fsl/fsl-5.0.10/fsl/bin/pngappend sla.png + slb.png + slc.png + sld.png + sle.png + slf.png + slg.png + slh.png + sli.png + slj.png + slk.png + sll.png {example_func2highres}1.png ; 
    /opt/fsl/fsl-5.0.10/fsl/bin/slicer {highres} {example_func2highres} -s 2 -x 0.35 sla.png -x 0.45 slb.png -x 0.55 slc.png -x 0.65 sld.png -y 0.35 sle.png -y 0.45 slf.png -y 0.55 slg.png -y 0.65 slh.png -z 0.35 sli.png -z 0.45 slj.png -z 0.55 slk.png -z 0.65 sll.png ; 
    /opt/fsl/fsl-5.0.10/fsl/bin/pngappend sla.png + slb.png + slc.png + sld.png + sle.png + slf.png + slg.png + slh.png + sli.png + slj.png + slk.png + sll.png {example_func2highres}2.png ; 
    /opt/fsl/fsl-5.0.10/fsl/bin/pngappend {example_func2highres}1.png - {example_func2highres}2.png {example_func2highres}.png; 
    /bin/rm -f sl?.png {example_func2highres}2.png
    /bin/rm {example_func2highres}1.png
    """.replace("\n"," ")
    
    plot_highres2standard = f"""
    /opt/fsl/fsl-5.0.10/fsl/bin/slicer {highres2standard} {standard} -s 2 -x 0.35 sla.png -x 0.45 slb.png -x 0.55 slc.png -x 0.65 sld.png -y 0.35 sle.png -y 0.45 slf.png -y 0.55 slg.png -y 0.65 slh.png -z 0.35 sli.png -z 0.45 slj.png -z 0.55 slk.png -z 0.65 sll.png ; 
    /opt/fsl/fsl-5.0.10/fsl/bin/pngappend sla.png + slb.png + slc.png + sld.png + sle.png + slf.png + slg.png + slh.png + sli.png + slj.png + slk.png + sll.png {highres2standard}1.png ; 
    /opt/fsl/fsl-5.0.10/fsl/bin/slicer {standard} {highres2standard} -s 2 -x 0.35 sla.png -x 0.45 slb.png -x 0.55 slc.png -x 0.65 sld.png -y 0.35 sle.png -y 0.45 slf.png -y 0.55 slg.png -y 0.65 slh.png -z 0.35 sli.png -z 0.45 slj.png -z 0.55 slk.png -z 0.65 sll.png ; 
    /opt/fsl/fsl-5.0.10/fsl/bin/pngappend sla.png + slb.png + slc.png + sld.png + sle.png + slf.png + slg.png + slh.png + sli.png + slj.png + slk.png + sll.png {highres2standard}2.png ; 
    /opt/fsl/fsl-5.0.10/fsl/bin/pngappend {highres2standard}1.png - {highres2standard}2.png {highres2standard}.png; 
    /bin/rm -f sl?.png {highres2standard}2.png
    /bin/rm {highres2standard}1.png
    """.replace("\n"," ")
    
    plot_example_func2standard = f"""
    /opt/fsl/fsl-5.0.10/fsl/bin/slicer {example_func2standard} {standard} -s 2 -x 0.35 sla.png -x 0.45 slb.png -x 0.55 slc.png -x 0.65 sld.png -y 0.35 sle.png -y 0.45 slf.png -y 0.55 slg.png -y 0.65 slh.png -z 0.35 sli.png -z 0.45 slj.png -z 0.55 slk.png -z 0.65 sll.png ; 
    /opt/fsl/fsl-5.0.10/fsl/bin/pngappend sla.png + slb.png + slc.png + sld.png + sle.png + slf.png + slg.png + slh.png + sli.png + slj.png + slk.png + sll.png {example_func2standard}1.png ; 
    /opt/fsl/fsl-5.0.10/fsl/bin/slicer {standard} {example_func2standard} -s 2 -x 0.35 sla.png -x 0.45 slb.png -x 0.55 slc.png -x 0.65 sld.png -y 0.35 sle.png -y 0.45 slf.png -y 0.55 slg.png -y 0.65 slh.png -z 0.35 sli.png -z 0.45 slj.png -z 0.55 slk.png -z 0.65 sll.png ; 
    /opt/fsl/fsl-5.0.10/fsl/bin/pngappend sla.png + slb.png + slc.png + sld.png + sle.png + slf.png + slg.png + slh.png + sli.png + slj.png + slk.png + sll.png {example_func2standard}2.png ; 
    /opt/fsl/fsl-5.0.10/fsl/bin/pngappend {example_func2standard}1.png - {example_func2standard}2.png {example_func2standard}.png; 
    /bin/rm -f sl?.png {example_func2standard}2.png
    """.replace("\n"," ")
    for cmdline in [plot_example_func2highres,plot_example_func2standard,plot_highres2standard]:
        os.system(cmdline)

def create_simple_highres2standard(roi,
                             roi_name,
                             preprocessed_functional_dir,
                             output_dir):
    from nipype.interfaces            import fsl
    from nipype.pipeline              import engine as pe
    from nipype.interfaces            import utility as util
    fsl.FSLCommand.set_default_output_type('NIFTI_GZ')
    
    simple_workflow         = pe.Workflow(name  = 'highres2standard')
    
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
#                                                        'reg',
                                                        'highres2standard.mat'))
    simple_workflow.inputs.inputspec.flt_reference  = os.path.abspath(os.path.join(preprocessed_functional_dir,
#                                                        'reg',
                                                        'MNI152_T1_2mm_brain.nii.gz'))
    simple_workflow.inputs.inputspec.mask           = os.path.abspath(os.path.join(preprocessed_functional_dir,
#                                                        'reg',
                                                        'MNI152_T1_2mm_brain_mask_dil.nii.gz'))
    simple_workflow.inputs.bound_by_mask.out_file   = os.path.abspath(os.path.join(output_dir,
                                                         roi_name.replace('_fsl.nii.gz',
                                                                          '_standard.nii.gz')))
    return simple_workflow

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
        
        plot_example_func2highres = f"""
/opt/fsl/fsl-5.0.10/fsl/bin/slicer {example_func2highres} {highres} -s 2 -x 0.35 sla.png -x 0.45 slb.png -x 0.55 slc.png -x 0.65 sld.png -y 0.35 sle.png -y 0.45 slf.png -y 0.55 slg.png -y 0.65 slh.png -z 0.35 sli.png -z 0.45 slj.png -z 0.55 slk.png -z 0.65 sll.png ; 
/opt/fsl/fsl-5.0.10/fsl/bin/pngappend sla.png + slb.png + slc.png + sld.png + sle.png + slf.png + slg.png + slh.png + sli.png + slj.png + slk.png + sll.png {example_func2highres}1.png ; 
/opt/fsl/fsl-5.0.10/fsl/bin/slicer {highres} {example_func2highres} -s 2 -x 0.35 sla.png -x 0.45 slb.png -x 0.55 slc.png -x 0.65 sld.png -y 0.35 sle.png -y 0.45 slf.png -y 0.55 slg.png -y 0.65 slh.png -z 0.35 sli.png -z 0.45 slj.png -z 0.55 slk.png -z 0.65 sll.png ; 
/opt/fsl/fsl-5.0.10/fsl/bin/pngappend sla.png + slb.png + slc.png + sld.png + sle.png + slf.png + slg.png + slh.png + sli.png + slj.png + slk.png + sll.png {example_func2highres}2.png ; 
/opt/fsl/fsl-5.0.10/fsl/bin/pngappend {example_func2highres}1.png - {example_func2highres}2.png {example_func2highres}.png; 
/bin/rm -f sl?.png {example_func2highres}2.png
/bin/rm {example_func2highres}1.png
        """.replace("\n"," ")
        
        plot_highres2standard = f"""
/opt/fsl/fsl-5.0.10/fsl/bin/slicer {highres2standard} {standard} -s 2 -x 0.35 sla.png -x 0.45 slb.png -x 0.55 slc.png -x 0.65 sld.png -y 0.35 sle.png -y 0.45 slf.png -y 0.55 slg.png -y 0.65 slh.png -z 0.35 sli.png -z 0.45 slj.png -z 0.55 slk.png -z 0.65 sll.png ; 
/opt/fsl/fsl-5.0.10/fsl/bin/pngappend sla.png + slb.png + slc.png + sld.png + sle.png + slf.png + slg.png + slh.png + sli.png + slj.png + slk.png + sll.png {highres2standard}1.png ; 
/opt/fsl/fsl-5.0.10/fsl/bin/slicer {standard} {highres2standard} -s 2 -x 0.35 sla.png -x 0.45 slb.png -x 0.55 slc.png -x 0.65 sld.png -y 0.35 sle.png -y 0.45 slf.png -y 0.55 slg.png -y 0.65 slh.png -z 0.35 sli.png -z 0.45 slj.png -z 0.55 slk.png -z 0.65 sll.png ; 
/opt/fsl/fsl-5.0.10/fsl/bin/pngappend sla.png + slb.png + slc.png + sld.png + sle.png + slf.png + slg.png + slh.png + sli.png + slj.png + slk.png + sll.png {highres2standard}2.png ; 
/opt/fsl/fsl-5.0.10/fsl/bin/pngappend {highres2standard}1.png - {highres2standard}2.png {highres2standard}.png; 
/bin/rm -f sl?.png {highres2standard}2.png
/bin/rm {highres2standard}1.png
        """.replace("\n"," ")
        
        plot_example_func2standard = f"""
/opt/fsl/fsl-5.0.10/fsl/bin/slicer {example_func2standard} {standard} -s 2 -x 0.35 sla.png -x 0.45 slb.png -x 0.55 slc.png -x 0.65 sld.png -y 0.35 sle.png -y 0.45 slf.png -y 0.55 slg.png -y 0.65 slh.png -z 0.35 sli.png -z 0.45 slj.png -z 0.55 slk.png -z 0.65 sll.png ; 
/opt/fsl/fsl-5.0.10/fsl/bin/pngappend sla.png + slb.png + slc.png + sld.png + sle.png + slf.png + slg.png + slh.png + sli.png + slj.png + slk.png + sll.png {example_func2standard}1.png ; 
/opt/fsl/fsl-5.0.10/fsl/bin/slicer {standard} {example_func2standard} -s 2 -x 0.35 sla.png -x 0.45 slb.png -x 0.55 slc.png -x 0.65 sld.png -y 0.35 sle.png -y 0.45 slf.png -y 0.55 slg.png -y 0.65 slh.png -z 0.35 sli.png -z 0.45 slj.png -z 0.55 slk.png -z 0.65 sll.png ; 
/opt/fsl/fsl-5.0.10/fsl/bin/pngappend sla.png + slb.png + slc.png + sld.png + sle.png + slf.png + slg.png + slh.png + sli.png + slj.png + slk.png + sll.png {example_func2standard}2.png ; 
/opt/fsl/fsl-5.0.10/fsl/bin/pngappend {example_func2standard}1.png - {example_func2standard}2.png {example_func2standard}.png; 
/bin/rm -f sl?.png {example_func2standard}2.png
        """.replace("\n"," ")
        for cmdline in [plot_example_func2highres,
                        plot_example_func2standard,
                        plot_highres2standard]:
            os.system(cmdline)
    except:
        print('you should not use python 2.7, update your python!!')

def create_highpass_filter_workflow(workflow_name   = 'highpassfiler',
                                    HP_freq         = 60,
                                    TR              = 0.85):
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
    
    img2float               = pe.MapNode(interface     = fsl.ImageMaths(out_data_type     = 'float',
                                                                        op_string         = '',
                                                                        suffix            = '_dtype'),
                                         iterfield     = ['in_file'],
                                         name          = 'img2float')
    highpass_workflow.connect(inputnode,'ICAed_file',
                              img2float,'in_file')
    
    getthreshold            = pe.MapNode(interface     = fsl.ImageStats(op_string = '-p 2 -p 98'),
                                         iterfield     = ['in_file'],
                                         name          = 'getthreshold')
    highpass_workflow.connect(img2float,    'out_file',
                              getthreshold, 'in_file')
    thresholding            = pe.MapNode(interface    = fsl.ImageMaths(out_data_type  = 'char',
                                                                       suffix         = '_thresh',
                                                                       op_string      = '-Tmin -bin'),
                                         iterfield    = ['in_file','op_string'],
                                         name         = 'thresholding')
    highpass_workflow.connect(img2float,    'out_file',
                              thresholding, 'in_file')
    highpass_workflow.connect(getthreshold,('out_stat',getthreshop),
                              thresholding,'op_string')
    
    dilatemask              = pe.MapNode(interface    = fsl.ImageMaths(suffix     = '_dil',
                                                                       op_string  = '-dilF'),
                                         iterfield    = ['in_file'],
                                         name         = 'dilatemask')
    highpass_workflow.connect(thresholding,'out_file',
                              dilatemask,'in_file')
    
    maskfunc                = pe.MapNode(interface    = fsl.ImageMaths(suffix     = '_mask',
                                                                       op_string  = '-mas'),
                                         iterfield    = ['in_file','in_file2'],
                                         name         = 'apply_dilatemask')
    highpass_workflow.connect(img2float,    'out_file',
                              maskfunc,     'in_file')
    highpass_workflow.connect(dilatemask,   'out_file',
                              maskfunc,     'in_file2')
    
    medianval               = pe.MapNode(interface    = fsl.ImageStats(op_string = '-k %s -p 50'),
                                         iterfield    = ['in_file','mask_file'],
                                         name         = 'cal_intensity_scale_factor')
    highpass_workflow.connect(img2float,    'out_file',
                              medianval,    'in_file')
    highpass_workflow.connect(thresholding, 'out_file',
                              medianval,    'mask_file')
    
    meanscale               = pe.MapNode(interface    = fsl.ImageMaths(suffix    = '_intnorm'),
                                         iterfield    = ['in_file','op_string'],
                                         name         = 'meanscale')
    highpass_workflow.connect(maskfunc,     'out_file',
                              meanscale,    'in_file')
    highpass_workflow.connect(medianval,    ('out_stat',getmeanscale),
                              meanscale,    'op_string')
    
    meanfunc                = pe.MapNode(interface    = fsl.ImageMaths(suffix     = '_mean',
                                                                       op_string  = '-Tmean'),
                                         iterfield    = ['in_file'],
                                         name         = 'meanfunc')
    highpass_workflow.connect(meanscale, 'out_file',
                              meanfunc,  'in_file')
    
    
    hpf                     = pe.MapNode(interface    = fsl.ImageMaths(suffix     = '_tempfilt',
                                                                       op_string  = '-bptf %.10f -1' % (HP_freq/2/TR)),
                                         iterfield    = ['in_file'],
                                         name         = 'highpass_filering')
    highpass_workflow.connect(meanscale,'out_file',
                              hpf,      'in_file',)
    
    addMean                 = pe.MapNode(interface    = fsl.BinaryMaths(operation = 'add'),
                                         iterfield    = ['in_file','operand_file'],
                                         name         = 'addmean')
    highpass_workflow.connect(hpf,      'out_file',
                              addMean,  'in_file')
    highpass_workflow.connect(meanfunc, 'out_file',
                              addMean,  'operand_file')
    
    highpass_workflow.connect(addMean,   'out_file',
                              outputnode,'filtered_file')
    
    return highpass_workflow

def load_csv(f,print_ = False,sub = None):
    temp = re.findall(r'\d+',f)
    n_session = int(temp[-2])
    n_run = int(temp[-1])
    if print_:
        print(n_session,n_run)
    df = pd.read_csv(f)
    df['session'] = n_session
    df['run'] = n_run
    df['id'] = df['session'] * 1000 + df['run'] * 100 + df['trials']
    if sub is not None:
        df['sub'] = sub
    return df

def build_model_dictionary(print_train = False,
                           class_weight = 'balanced',
                           remove_invariant = True,
                           l1 = False,
                           n_jobs = 1,
                           C = 1,
                           tol = 1e-2):
    np.random.seed(12345)
    if l1:
        svm = LinearSVC(penalty = 'l1', # not default
                        dual = False, # not default
                        tol = tol, # not default
                        random_state = 12345, # not default
                        max_iter = int(1e4), # default
                        class_weight = class_weight, # not default
                        C = C,
                        )
    else:
        svm = LinearSVC(penalty = 'l2', # default
                        dual = True, # default
                        tol = tol, # not default
                        random_state = 12345, # not default
                        max_iter = int(1e4), # default
                        class_weight = class_weight, # not default
                        C = C,
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
    """
    check if we get repeated test sets
    """
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
    """
    check the balance of the training set.
    if only one of the classes has more 2 instances than the other
    we will randomly take out those 'extra instances' from the major
    class
    """
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

def load_roi_array_data(df_event,
                        conscious_state,
                        BOLD,
                        label_map,
                        feature_dir,
                        model_name,
                        scaling_func = None,
                        ):
    idx_unconscious  = df_event['visibility'] == conscious_state
    data             = BOLD[idx_unconscious]
    df_data          = df_event[idx_unconscious].reset_index(drop=True)
    df_data['id']    = df_data['session'] * 1000 + df_data['run'] * 100 + df_data['trials']
    targets          = np.array([label_map[item] for item in df_data['targets'].values])#[:,-1]
    if scaling_func is not None:
        scaler       = scaling_func.fit(data)
        data         = scaler.transform(data)
    else:
        scaler       = None
    images           = df_data['paths'].apply(lambda x: x.split('.')[0] + '.npy').values
    CNN_feature      = np.array([np.load(os.path.join(feature_dir,
                                                     model_name,
                                                     item)) for item in images])
    groups           = df_data['labels'].values
    return data,df_data,targets,scaler,CNN_feature,groups

def LOO_partition(df_data,target_column = 'labels'):
    temp = {'targets':[],target_column:[]}
    for (targets,labels),df_sub in df_data.groupby(['targets',target_column]):
        temp['targets'].append(targets)
        temp[target_column].append(labels)
    temp = pd.DataFrame(temp)
    temp = temp.sort_values(['targets',target_column])
    living = temp[temp['targets'] == 'Living_Things'][target_column].values
    nonliving = temp[temp['targets'] == 'Nonliving_Things'][target_column].values
    test_pairs = [[a,b] for a in living for b in nonliving]
    idxs_train,idxs_test = [],[]
    for test_pair in test_pairs:
        idx_test = np.logical_or(df_data[target_column] == test_pair[0],
                                 df_data[target_column] == test_pair[1])
        idx_train = np.invert(idx_test)
        idxs_train.append(np.where(idx_train == True)[0])
        idxs_test.append(np.where(idx_test == True)[0])
    return idxs_train,idxs_test

def check_LOO_cv(idxs_test_target,df_data_target,df_data_source):
    from tqdm import tqdm
    cv_warning                  = False
    idxs_train_source           = []
    idxs_test_source            = []
    for idx_test_target in tqdm(idxs_test_target):
        df_data_target_sub      = df_data_target.iloc[idx_test_target]
        unique_subcategories    = pd.unique(df_data_target_sub['labels'])
        # category check:
        # print(Counter(df_data_target_sub['targets']))
        idx_train_source        = []
        idx_test_source         = []
        for subcategory,df_data_source_sub in df_data_source.groupby(['labels']):
            if subcategory not in unique_subcategories:
                idx_train_source.append(list(df_data_source_sub.index))
            else:
                idx_test_source.append(list(df_data_source_sub.index))
        idx_train_source        = np.concatenate(idx_train_source)
        idx_test_source         = idx_train_source.copy()
        
        # check if the training and testing have subcategory overlapping
        target_set              = set(pd.unique(df_data_target.iloc[idx_test_target]['labels']))
        source_set              = set(pd.unique(df_data_source.iloc[idx_train_source]['labels']))
        overlapping             = target_set.intersection(source_set)
        # print(f'overlapped subcategories: {overlapping}')
        if len(overlapping) > 0:
            cv_warning          = True
        idxs_train_source.append(idx_train_source)
        # the testing set for the source does NOT matter since we don't care its performance
        idxs_test_source.append(idx_test_source)
    return cv_warning,idxs_train_source,idxs_test_source

def resample_ttest(x,
                   baseline         = 0.5,
                   n_ps             = 100,
                   n_permutation    = 10000,
                   one_tail         = False,
                   n_jobs           = 12, 
                   verbose          = 0,
                   full_size        = True
                   ):
    """
    http://www.stat.ucla.edu/~rgould/110as02/bshypothesis.pdf
    https://www.tau.ac.il/~saharon/StatisticsSeminar_files/Hypothesis.pdf
    Inputs:
    ----------
    x: numpy array vector, the data that is to be compared
    baseline: the single point that we compare the data with
    n_ps: number of p values we want to estimate
    one_tail: whether to perform one-tailed comparison
    """
    import numpy as np
    import gc
    from joblib import Parallel,delayed
    # statistics with the original data distribution
    t_experiment    = np.mean(x)
    null            = x - np.mean(x) + baseline # shift the mean to the baseline but keep the distribution
    
    if null.shape[0] > int(1e4): # catch for big data
        full_size   = False
    if not full_size:
        size        = int(1e3)
    else:
        size = null.shape[0]
    
    
    gc.collect()
    def t_statistics(null,size,):
        """
        null: shifted data distribution
        size: tuple of 2 integers (n_for_averaging,n_permutation)
        """
        null_dist   = np.random.choice(null,size = size,replace = True)
        t_null      = np.mean(null_dist,0)
        if one_tail:
            return ((np.sum(t_null >= t_experiment)) + 1) / (size[1] + 1)
        else:
            return ((np.sum(np.abs(t_null) >= np.abs(t_experiment))) + 1) / (size[1] + 1) /2
    if n_ps == 1:
        ps = t_statistics(null, size)
    else:
        ps = Parallel(n_jobs = n_jobs,verbose = verbose)(delayed(t_statistics)(**{
                        'null':null,
                        'size':(size,int(n_permutation)),}) for i in range(n_ps))
        ps = np.array(ps)
    return ps
def resample_ttest_2sample(a,b,
                           n_ps                 = 100,
                           n_permutation        = 10000,
                           one_tail             = False,
                           match_sample_size    = True,
                           n_jobs               = 6,
                           verbose              = 0):
    from joblib import Parallel,delayed
    import gc
    # when the samples are dependent just simply test the pairwise difference against 0
    # which is a one sample comparison problem
    if match_sample_size:
        difference  = a - b
        ps          = resample_ttest(difference,
                                     baseline       = 0,
                                     n_ps           = n_ps,
                                     n_permutation  = n_permutation,
                                     one_tail       = one_tail,
                                     n_jobs         = n_jobs,
                                     verbose        = verbose,)
        return ps
    else: # when the samples are independent
        t_experiment        = np.mean(a) - np.mean(b)
        if not one_tail:
            t_experiment    = np.abs(t_experiment)
            
        def t_statistics(a,b):
            group           = np.concatenate([a,b])
            np.random.shuffle(group)
            new_a           = group[:a.shape[0]]
            new_b           = group[a.shape[0]:]
            t_null          = np.mean(new_a) - np.mean(new_b)
            if not one_tail:
                t_null      = np.abs(t_null)
            return t_null
        
        gc.collect()
        ps = np.zeros(n_ps)
        for ii in range(n_ps):
            t_null_null = Parallel(n_jobs = n_jobs,verbose = verbose)(delayed(t_statistics)(**{
                            'a':a,
                            'b':b}) for i in range(n_permutation))
            if one_tail:
                ps[ii] = ((np.sum(t_null_null >= t_experiment)) + 1) / (n_permutation + 1)
            else:
                ps[ii] = ((np.sum(np.abs(t_null_null) >= np.abs(t_experiment))) + 1) / (n_permutation + 1) / 2
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
        if method == "bonferroni":
            return [np.min([1, i]) for i in self.sorted_pvals * self.len]
        elif method == "holm":
            return [np.min([1, i]) for i in (self.sorted_pvals * (self.len - np.arange(1, self.len+1) + 1))]
        elif method == "bh":
            p_times_m_i = self.sorted_pvals * self.len / np.arange(1, self.len+1)
            return [np.min([p, p_times_m_i[i+1]]) if i < self.len-1 else p for i, p in enumerate(p_times_m_i)]
        elif method == "lfdr":
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
                if method != "lfdr":
                    df[method] = self.adjust(method)
        return df
def define_roi_category():
    roi_dict = {'fusiform':'Visual',
                'parahippocampal':'Visual',
                'pericalcarine':'Visual',
                'precuneus':'Visual',
                'superiorparietal':'Working Memory',
                'inferiortemporal':'Visual',
                'lateraloccipital':'Visual',
                'lingual':'Visual',
                'rostralmiddlefrontal':'Working Memory',
                'superiorfrontal':'Working Memory',
                'ventrolateralPFC':'Working Memory',
                'inferiorparietal':'Visual',
                }
    
    return roi_dict

def rename_ROI_for_plotting():
    name_map = {
        'fusiform':'Fusiform gyrus',
        'inferiorparietal':'Inferior parietal lobe',
        'inferiortemporal':'Inferior temporal lobe',
        'lateraloccipital':'Lateral occipital cortex',
        'lingual':'Lingual',
        'middlefrontal':'Middle frontal gyrus',
        'rostralmiddlefrontal':'Middle fontal gyrus',
        'parahippocampal':'Parahippocampal gyrus',
        'pericalcarine':'Pericalcarine cortex',
        'precuneus':'Precuneus',
        'superiorfrontal':'Superior frontal gyrus',
        'superiorparietal':'Superior parietal gyrus',
        'ventrolateralPFC':'Inferior frontal gyrus',
        }
    return name_map

def stars(x):
    if x < 0.001:
        return '***'
    elif x < 0.01:
        return '**'
    elif x < 0.05:
        return '*'
    else:
        return 'n.s.'

def get_fs(x):
    return x.split(' + ')[0]
def get_clf(x):
    return x.split(' + ')[1]
def rename_roi(x):
    return x.split('-')[-1] + '-' + x.split('-')[1]

def strip_interaction_names(df_corrected):
    results = []
    for ii,row in df_corrected.iterrows():
        row['window'] = row['level1'].split('_')[0]
        try:
            row['attribute1']= row['level1'].split('_')[1]
            row['attribute2']= row['level2'].split('_')[1]
        except:
            row['attr1']= row['level1'].split('_')[1]
            row['attr2']= row['level2'].split('_')[1]
        results.append(row.to_frame().T)
    results = pd.concat(results)
    return results
def compute_xy(df_sub,position_map,hue_map):
    df_add = []
    for ii,row in df_sub.iterrows():
        xtick = int(row['window']) - 1
        attribute1_x = xtick + position_map[hue_map[row['attribute1']]]
        attribute2_x = xtick + position_map[hue_map[row['attribute2']]]
        row['x1'] = attribute1_x
        row['x2'] = attribute2_x
        df_add.append(row.to_frame().T)
    df_add = pd.concat(df_add)
    return df_add

def split_probe_path(x,idx):
    temp = x.split('/')
    return temp[idx]

def standard_MNI_coordinate_for_plot():
    return {
        'lh-fusiform':(-47,-52,-12),
        'rh-fusiform':(47,-51,-14),
        'lh-inferiorparietal':(-46,-60,33),
        'rh-inferiorparietal':(46,-59,31),
        'lh-inferiortemporal':(-47,-14,-34),
        'rh-inferiortemporal':(48,-17,-31),
        'lh-lateraloccipital':(-46,-58,-8),
        'rh-lateraloccipital':(40,-78,12),
        'lh-lingual':(-11,-81,7),
        'rh-lingual':(11,-78,9),
        'lh-rostralmiddlefrontal':(-30,50,24),
        'rh-rostralmiddlefrontal':(4,58,30),
        'lh-parahippocampal':(-25,-22,-22),
        'rh-parahippocampal':(27,-19,-25),
        'lh-pericalcarine':(-24,-66,8),
        'rh-pericalcarine':(26,-68,12),
        'lh-precuneus':None,
        'rh-precuneus':None,
        'lh-superiorfrontal':(-23,24,44),
        'rh-superiorfrontal':(22,26,45),
        'lh-superiorparietal':(-18,-61,55),
        'rh-superiorparietal':(27,-60,45),
        'lh-ventrolateralPFC':(-32,54,-4),
        'rh-ventrolateralPFC':(42,46,0)}

def bootstrap_behavioral_estimation(df_sub,n_bootstrap = int(1e2)):
    scores,chance = [],[]
    responses = df_sub['response.keys_raw'].values - 1
    answers = df_sub['correctAns_raw'].values - 1
    np.random.seed(12345)
    for n_ in tqdm(range(n_bootstrap)):
        idx = np.random.choice(np.arange(responses.shape[0]),
                               size = responses.shape[0],
                               replace = True)
        
        response_ = responses[idx]
        answer_ = answers[idx]
        score_ = roc_auc_score(answer_,response_)
        scores.append(score_)
    scores = np.array(scores)
    # chance
    # by keeping the answers in order but shuffle the response, 
    # we can estimate the chance
    # level accuracy
    idx = np.random.choice(np.arange(responses.shape[0]),
                           size = responses.shape[0],
                           replace = True)
    
    response_ = responses[idx]
    answer_ = answers[idx]
    chance = np.array([roc_auc_score(answer_,shuffle(response_))\
                       for _ in tqdm(range(n_bootstrap))])
    
    pvals = resample_ttest_2sample(scores,chance,one_tail = True,
                                   match_sample_size = True)
    return pvals,scores,chance

def get_label_category_mapping():
    return {'Chest-of-drawers': 'Nonliving_Things',
 'armadillo': 'Living_Things',
 'armchair': 'Nonliving_Things',
 'axe': 'Nonliving_Things',
 'barn-owl': 'Living_Things',
 'bed': 'Nonliving_Things',
 'bedside-table': 'Nonliving_Things',
 'boat': 'Nonliving_Things',
 'bookcase': 'Nonliving_Things',
 'bus': 'Nonliving_Things',
 'butterfly': 'Living_Things',
 'car': 'Nonliving_Things',
 'castle': 'Nonliving_Things',
 'cat': 'Living_Things',
 'cathedral': 'Nonliving_Things',
 'chair': 'Nonliving_Things',
 'cheetah': 'Living_Things',
 'church': 'Nonliving_Things',
 'coking-pot': 'Nonliving_Things',
 'couch': 'Nonliving_Things',
 'cow': 'Living_Things',
 'crab': 'Living_Things',
 'cup': 'Nonliving_Things',
 'dolphin': 'Living_Things',
 'dragonfly': 'Living_Things',
 'drum': 'Nonliving_Things',
 'duck': 'Living_Things',
 'elephant': 'Living_Things',
 'factory': 'Nonliving_Things',
 'filling-cabinet': 'Nonliving_Things',
 'fondue': 'Nonliving_Things',
 'frying-pan': 'Nonliving_Things',
 'giraffe': 'Living_Things',
 'goldfinch': 'Living_Things',
 'goose': 'Living_Things',
 'granary': 'Nonliving_Things',
 'guitar': 'Nonliving_Things',
 'hammer': 'Nonliving_Things',
 'hen': 'Living_Things',
 'hippopotamus': 'Living_Things',
 'horse': 'Living_Things',
 'house': 'Nonliving_Things',
 'hummingbird': 'Living_Things',
 'killer-whale': 'Living_Things',
 'kiwi': 'Living_Things',
 'ladybird': 'Living_Things',
 'lamp': 'Nonliving_Things',
 'lectern': 'Nonliving_Things',
 'lioness': 'Living_Things',
 'lobster': 'Living_Things',
 'lynx': 'Living_Things',
 'magpie': 'Living_Things',
 'manatee': 'Living_Things',
 'mill': 'Nonliving_Things',
 'motorbike': 'Nonliving_Things',
 'narwhal': 'Living_Things',
 'ostrich': 'Living_Things',
 'owl': 'Living_Things',
 'palace': 'Nonliving_Things',
 'partridge': 'Living_Things',
 'pelican': 'Living_Things',
 'penguin': 'Living_Things',
 'piano': 'Nonliving_Things',
 'pigeon': 'Living_Things',
 'plane': 'Nonliving_Things',
 'pomfret': 'Living_Things',
 'pot': 'Nonliving_Things',
 'raven': 'Living_Things',
 'rhino': 'Living_Things',
 'rocking-chair': 'Nonliving_Things',
 'rooster': 'Living_Things',
 'saucepan': 'Nonliving_Things',
 'saxophone': 'Nonliving_Things',
 'scorpion': 'Living_Things',
 'seagull': 'Living_Things',
 'shark': 'Living_Things',
 'ship': 'Nonliving_Things',
 'small-saucepan': 'Nonliving_Things',
 'sofa': 'Nonliving_Things',
 'sparrow': 'Living_Things',
 'sperm-whale': 'Living_Things',
 'table': 'Nonliving_Things',
 'tapir': 'Living_Things',
 'teapot': 'Nonliving_Things',
 'tiger': 'Living_Things',
 'toucan': 'Living_Things',
 'tractor': 'Nonliving_Things',
 'train': 'Nonliving_Things',
 'trumpet': 'Nonliving_Things',
 'tuba': 'Nonliving_Things',
 'turtle': 'Living_Things',
 'van': 'Nonliving_Things',
 'violin': 'Nonliving_Things',
 'wardrobe': 'Nonliving_Things',
 'whale': 'Living_Things',
 'zebra': 'Living_Things'}
def get_label_subcategory_mapping():
    return {'Chest-of-drawers': 'Furniture',
 'armadillo': 'Animals',
 'armchair': 'Furniture',
 'axe': 'Tools',
 'barn-owl': 'Birds',
 'bed': 'Furniture',
 'bedside-table': 'Furniture',
 'boat': 'Vehicles',
 'bookcase': 'Furniture',
 'bus': 'Vehicles',
 'butterfly': 'Insects',
 'car': 'Vehicles',
 'castle': 'Buildings',
 'cat': 'Animals',
 'cathedral': 'Buildings',
 'chair': 'Furniture',
 'cheetah': 'Animals',
 'church': 'Buildings',
 'coking-pot': 'Kitchen_Uten',
 'couch': 'Furniture',
 'cow': 'Animals',
 'crab': 'Marine_creatures',
 'cup': 'Kitchen_Uten',
 'dolphin': 'Marine_creatures',
 'dragonfly': 'Insects',
 'drum': 'Musical_Inst',
 'duck': 'Birds',
 'elephant': 'Animals',
 'factory': 'Buildings',
 'filling-cabinet': 'Furniture',
 'fondue': 'Kitchen_Uten',
 'frying-pan': 'Kitchen_Uten',
 'giraffe': 'Animals',
 'goldfinch': 'Birds',
 'goose': 'Birds',
 'granary': 'Buildings',
 'guitar': 'Musical_Inst',
 'hammer': 'Tools',
 'hen': 'Birds',
 'hippopotamus': 'Animals',
 'horse': 'Animals',
 'house': 'Buildings',
 'hummingbird': 'Birds',
 'killer-whale': 'Marine_creatures',
 'kiwi': 'Birds',
 'ladybird': 'Insects',
 'lamp': 'Furniture',
 'lectern': 'Furniture',
 'lioness': 'Animals',
 'lobster': 'Marine_creatures',
 'lynx': 'Animals',
 'magpie': 'Birds',
 'manatee': 'Marine_creatures',
 'mill': 'Buildings',
 'motorbike': 'Vehicles',
 'narwhal': 'Marine_creatures',
 'ostrich': 'Birds',
 'owl': 'Birds',
 'palace': 'Buildings',
 'partridge': 'Birds',
 'pelican': 'Birds',
 'penguin': 'Birds',
 'piano': 'Musical_Inst',
 'pigeon': 'Birds',
 'plane': 'Vehicles',
 'pomfret': 'Marine_creatures',
 'pot': 'Kitchen_Uten',
 'raven': 'Birds',
 'rhino': 'Animals',
 'rocking-chair': 'Furniture',
 'rooster': 'Birds',
 'saucepan': 'Kitchen_Uten',
 'saxophone': 'Musical_Inst',
 'scorpion': 'Insects',
 'seagull': 'Birds',
 'shark': 'Marine_creatures',
 'ship': 'Vehicles',
 'small-saucepan': 'Kitchen_Uten',
 'sofa': 'Furniture',
 'sparrow': 'Birds',
 'sperm-whale': 'Marine_creatures',
 'table': 'Furniture',
 'tapir': 'Animals',
 'teapot': 'Kitchen_Uten',
 'tiger': 'Animals',
 'toucan': 'Birds',
 'tractor': 'Vehicles',
 'train': 'Vehicles',
 'trumpet': 'Musical_Inst',
 'tuba': 'Musical_Inst',
 'turtle': 'Animals',
 'van': 'Vehicles',
 'violin': 'Musical_Inst',
 'wardrobe': 'Furniture',
 'whale': 'Marine_creatures',
 'zebra': 'Animals'}
    
def make_df_axis(df_data):
    label_category_map = get_label_category_mapping()
    label_subcategory_map = get_label_subcategory_mapping()
    df_axis = pd.DataFrame({'labels':pd.unique(df_data['labels'])})
    df_axis['category'] = df_axis['labels'].map(label_category_map)
    df_axis['subcategory'] = df_axis['labels'].map(label_subcategory_map)
    df_axis = df_axis.sort_values(['category','subcategory','labels'])
    return df_axis

def load_same_same(sub,target_folder = 'decoding',target_file = '*None*csv'):
    working_dir = '../../../../results/MRI/nilearn/{}/{}'.format(sub,target_folder)
    working_data = glob(os.path.join(working_dir,target_file))
    
    df = pd.concat([pd.read_csv(f) for f in working_data])
    if 'model_name' not in df.columns:
        df['model_name'] = df['model']
    df['feature_selector'] = df['model_name'].apply(get_fs)
    df['estimator'] = df['model_name'].apply(get_clf)
    if 'score' in df.columns:
        df['roc_auc'] = df['score']
    
    temp = np.array([item.split('-') for item in df['roi'].values])
    df['roi_name'] = temp[:,1]
    df['side'] = temp[:,0]
    return df

def plot_stat_map(stat_map_img, 
                  bg_img                    = '', 
                  cut_coords                = None,
                  output_file               = None, 
                  display_mode              = 'ortho', 
                  colorbar                  = True,
                  figure                    = None, 
                  axes                      = None, 
                  title                     = None, 
                  threshold                 = 1e-6,
                  annotate                  = True, 
                  draw_cross                = True, 
                  black_bg                  = 'auto',
                  cmap                      = cm.coolwarm, 
                  symmetric_cbar            = "auto",
                  dim                       = 'auto', 
                  vmin_                     = None,
                  vmax                      = None, 
                  resampling_interpolation  = 'continuous',
                  **kwargs):
    
    bg_img, black_bg, bg_vmin, bg_vmax      = _load_anat(
                  bg_img, 
                  dim                       = dim,
                  black_bg                  = black_bg)

    stat_map_img                            = _utils.check_niimg_3d(
                  stat_map_img, 
                  dtype                     = 'auto')

    cbar_vmin, cbar_vmax, vmin, vmax        = _get_colorbar_and_data_ranges(
                  _safe_get_data(
                          stat_map_img, 
                          ensure_finite     = True),
                          vmax,
                          symmetric_cbar,
                          kwargs)
    display                                 = _plot_img_with_bg(
                  img                       = stat_map_img, 
                  bg_img                    = bg_img, 
                  cut_coords                = cut_coords,
                  output_file               = output_file, 
                  display_mode              = display_mode,
                  figure                    = figure, 
                  axes                      = axes, 
                  title                     = title, 
                  annotate                  = annotate,
                  draw_cross                = draw_cross, 
                  black_bg                  = black_bg, 
                  threshold                 = threshold,
                  bg_vmin                   = bg_vmin, 
                  bg_vmax                   = bg_vmax, 
                  cmap                      = cmap, 
                  vmin                      = vmin_, 
                  vmax                      = vmax,
                  colorbar                  = colorbar, 
                  cbar_vmin                 = vmin_, 
                  cbar_vmax                 = cbar_vmax,
                  resampling_interpolation  = resampling_interpolation, 
                  **kwargs)

    return display

def load_whole_brain_data_with_mask(BOLD_file,csv_file,masker):
    masker.fit()
    df = pd.read_csv(csv_file)
    masker.sessions = df['session'].values
    BOLD = masker.transform(BOLD_file)
    return BOLD

def make_ridge_model_CV(perform_pca = True,alpha_space = [1,12],custom_scorer = None):
    from sklearn import linear_model,metrics
    from sklearn.decomposition import PCA
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import make_pipeline
    
    if custom_scorer == None:
        def score_func(y, y_pred,):
            temp        = metrics.r2_score(y,y_pred,multioutput = 'raw_values')
            if np.sum(temp > 0):
                return temp[temp > 0].mean()
            else:
                return 0
        custom_scorer      = metrics.make_scorer(score_func,greater_is_better = True)
        
    pca         = PCA(n_components = .99,random_state = 12345)
#    scaler      = StandardScaler()
    reg         = linear_model.Ridge(normalize      = True,
                                     alpha          = 1,
                                     random_state   = 12345)
    if perform_pca:
        reg         = GridSearchCV(make_pipeline(pca,reg),
                                   dict(ridge__alpha = np.logspace(alpha_space[0],alpha_space[1],alpha_space[1] - alpha_space[0] + 1),
                                        ),
                                   scoring  = custom_scorer,
                                   n_jobs   = 1,
                                   cv       = 10,
                                   )
    else:
        reg         = GridSearchCV(make_pipeline(reg),
                                   dict(ridge__alpha = np.logspace(alpha_space[0],alpha_space[1],alpha_space[1] - alpha_space[0] + 1),
                                        ),
                                   scoring  = custom_scorer,
                                   n_jobs   = 1,
                                   cv       = 10,
                                   )
    return reg
    
def cross_validation(feature_dir,
                     encoding_model,
                     custom_scorer,
                     BOLD_sc_source,
                     idxs_train_source,
                     idxs_test_source,
                     image_source,
                     image_target,):
    """
    Encoding pipeline
    """
    from sklearn.model_selection import cross_validate
    features_source         = np.array([np.load(os.path.join(feature_dir,
                                                            encoding_model,
                                                            item)) for item in image_source])
    features_target         = np.array([np.load(os.path.join(feature_dir,
                                                            encoding_model,
                                                            item)) for item in image_target])
    reg = make_ridge_model_CV(custom_scorer = custom_scorer)
    res = cross_validate(reg,
                         features_source,
                         BOLD_sc_source,
                         scoring  = 'r2',
                         cv = zip(idxs_train_source,idxs_test_source),
                         return_estimator = True,
                         n_jobs = -1,
                         verbose = 1,)
    return res,features_target,features_source

def fill_results(scores,
                 results,
                 n_splits,
                 conscious_source,
                 conscious_target,
                 roi_name,
                 BOLD_sc_source,
                 features_source,
                 corr,):
    mean_variance = scores.copy()
    mean_variance = np.array([item[item > 0].mean() for item in mean_variance])
    positive_voxels = np.array([np.sum(temp > 0) for temp in scores])
    positive_voxel_indices = [','.join(str(item) for item in np.where(row > 0.)[0]) for row in scores]
    
    scores_to_save = mean_variance.copy()
    scores_to_save = np.nan_to_num(scores_to_save,)
    results['mean_variance'] = scores.mean(1)#scores_to_save
    results['fold'] = np.arange(n_splits) + 1
    results['conscious_source'] = [conscious_source] * n_splits
    results['conscious_target'] = [conscious_target] * n_splits
    results['roi_name'] = [roi_name] * n_splits
    results['positive voxels'  ]= positive_voxels
    results['n_parameters'] = [BOLD_sc_source.shape[1] * features_source.shape[1]] * n_splits
    results['corr'] = corr
    results['positive_voxel_indices'] = positive_voxel_indices
    return scores.mean(1),results

def get_array_from_dataframe(df,column_name):
    return np.array([item for item in df[column_name].values[0].replace('[',
                     '').replace(']',
                        '').replace('\n',
                          '').replace('  ',
                            ' ').replace(',',' ').split(' ') if len(item) > 0],
                    dtype = 'float32')

def load_whole_brain_BOLD_csv_preprocessing(BOLD_file_name,csv_file_name,whole_brain_mask,
                                label_map = {'Nonliving_Things': [0, 1], 'Living_Things': [1, 0]},
                                preprocessing_steps = ['scale_data','clustering','permute_voxels'],
                                kernel_size = None):
    import gc
    import numpy as np
    import pandas as pd
    
    from sklearn import cluster
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.preprocessing import MinMaxScaler
    
    from nilearn.input_data import NiftiMasker
    
    masker           = NiftiMasker(whole_brain_mask)
    data             = masker.fit_transform(BOLD_file_name)
    df_data          = pd.read_csv(csv_file_name)
    df_data['id']    = df_data['session'] * 1000 + df_data['run'] * 100 + df_data['trials']
    targets          = np.array([label_map[item] for item in df_data['targets'].values])
    scaler           = make_pipeline(VarianceThreshold(),MinMaxScaler((-1,1)))
    if 'scale_data' in preprocessing_steps:
        print('scaling')
        data         = scaler.fit_transform(data)
    if 'clustering' in preprocessing_steps:
        print('clustering')
        gc.collect()
        if kernel_size == None:
            kernel_size = 8
        CLUSTER      = cluster.DBSCAN(min_samples   = kernel_size,
                                      metric        = 'correlation',
                                      n_jobs        = -1,)
        CLUSTER.fit(data.T)
        idx          = np.argsort(CLUSTER.labels_)
        data         = data[:,idx]
    if ('permute_voxels' in preprocessing_steps) and ('clustering' not in preprocessing_steps):
        np.random.seed(12345)
        print('shuffling')
        data         = np.random.shuffle(data.T).T
    return data,df_data,targets,scaler

def load_BOLD_csv_preprocessing(BOLD_file_name,csv_file_name,conscious_state,
                                label_map = {'Nonliving_Things': [0, 1], 'Living_Things': [1, 0]},
                                preprocessing_steps = ['scale_data','clustering','permute_voxels'],
                                kernel_size = None):
    import gc
    import numpy as np
    import pandas as pd
    
    from sklearn import cluster
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.preprocessing import MinMaxScaler
    
    BOLD             = np.load(BOLD_file_name)
    event            = pd.read_csv(csv_file_name)
    roi_name         = csv_file_name.split('/')[-1].split('_events')[0]
    idx_unconscious  = event['visibility'] == conscious_state
    data             = BOLD[idx_unconscious]
    df_data          = event[idx_unconscious].reset_index(drop=True)
    df_data['id']    = df_data['session'] * 1000 + df_data['run'] * 100 + df_data['trials']
    targets          = np.array([label_map[item] for item in df_data['targets'].values])
    scaler           = make_pipeline(VarianceThreshold(),MinMaxScaler((-1,1)))
    if 'scale_data' in preprocessing_steps:
        data = scaler.fit_transform(data)
    if 'clustering' in preprocessing_steps:
        gc.collect()
        if kernel_size == None:
            kernel_size = 8
        CLUSTER = cluster.DBSCAN(min_samples = kernel_size,
                                 metric = 'correlation',
                                 n_jobs = -1,)
        CLUSTER.fit(data.T)
        idx = np.argsort(CLUSTER.labels_)
        data = data[:,idx]
    if ('permute_voxels' in preprocessing_steps) and ('clustering' not in preprocessing_steps):
        np.random.seed(12345)
        data         = np.random.shuffle(data.T).T
    return data,df_data,targets,scaler,roi_name
###################################################################################
###################################################################################
import numpy as np
import scipy.signal
from scipy.stats import kurtosis
try:
    from mne.preprocessing import find_outliers
except:
    pass
from numpy import nanmean
from mne.utils import logger
#from mne.preprocessing.eog import _get_eog_channel_index


def hurst(x):
    """Estimate Hurst exponent on a timeseries.

    The estimation is based on the second order discrete derivative.

    Parameters
    ----------
    x : 1D numpy array
        The timeseries to estimate the Hurst exponent for.

    Returns
    -------
    h : float
        The estimation of the Hurst exponent for the given timeseries.
    """
    y = np.cumsum(np.diff(x, axis=1), axis=1)

    b1 = [1, -2, 1]
    b2 = [1,  0, -2, 0, 1]

    # second order derivative
    y1 = scipy.signal.lfilter(b1, 1, y, axis=1)
    y1 = y1[:, len(b1) - 1:-1]  # first values contain filter artifacts

    # wider second order derivative
    y2 = scipy.signal.lfilter(b2, 1, y, axis=1)
    y2 = y2[:, len(b2) - 1:-1]  # first values contain filter artifacts

    s1 = np.mean(y1 ** 2, axis=1)
    s2 = np.mean(y2 ** 2, axis=1)

    return 0.5 * np.log2(s2 / s1)

def _freqs_power(data, sfreq, freqs):
    fs, ps = scipy.signal.welch(data, sfreq,
                                nperseg=2 ** int(np.log2(10 * sfreq) + 1),
                                noverlap=0,
                                axis=-1)
    return np.sum([ps[..., np.searchsorted(fs, f)] for f in freqs], axis=0)

def faster_bad_channels(epochs, picks=None, thres=3, use_metrics=None):
    """Implements the first step of the FASTER algorithm.
    
    This function attempts to automatically mark bad EEG channels by performing
    outlier detection. It operated on epoched data, to make sure only relevant
    data is analyzed.

    Parameters
    ----------
    epochs : Instance of Epochs
        The epochs for which bad channels need to be marked
    picks : list of int | None
        Channels to operate on. Defaults to EEG channels.
    thres : float
        The threshold value, in standard deviations, to apply. A channel
        crossing this threshold value is marked as bad. Defaults to 3.
    use_metrics : list of str
        List of metrics to use. Can be any combination of:
            'variance', 'correlation', 'hurst', 'kurtosis', 'line_noise'
        Defaults to all of them.

    Returns
    -------
    bads : list of str
        The names of the bad EEG channels.
    """
    metrics = {
        'variance':    lambda x: np.var(x, axis=1),
        'correlation': lambda x: nanmean(
                           np.ma.masked_array(
                               np.corrcoef(x),
                               np.identity(len(x), dtype=bool)
                           ),
                           axis=0),
        'hurst':       lambda x: hurst(x),
        'kurtosis':    lambda x: kurtosis(x, axis=1),
        'line_noise':  lambda x: _freqs_power(x, epochs.info['sfreq'],
                                              [50, 60]),
    }

    if picks is None:
        picks = mne.pick_types(epochs.info, meg=False, eeg=True, exclude=[])
    if use_metrics is None:
        use_metrics = metrics.keys()

    # Concatenate epochs in time
    data = epochs.get_data()
    data = data.transpose(1, 0, 2).reshape(data.shape[1], -1)
    data = data[picks]

    # Find bad channels
    bads = []
    for m in use_metrics:
        s = metrics[m](data)
        b = [epochs.ch_names[picks[i]] for i in find_outliers(s, thres)]
        logger.info('Bad by %s:\n\t%s' % (m, b))
        bads.append(b)

    return np.unique(np.concatenate(bads)).tolist()

def _deviation(data):
    """Computes the deviation from mean for each channel in a set of epochs.

    This is not implemented as a lambda function, because the channel means
    should be cached during the computation.
    
    Parameters
    ----------
    data : 3D numpy array
        The epochs (#epochs x #channels x #samples).

    Returns
    -------
    dev : 1D numpy array
        For each epoch, the mean deviation of the channels.
    """
    ch_mean = np.mean(data, axis=2)
    return ch_mean - np.mean(ch_mean, axis=0)

def faster_bad_epochs(epochs, picks=None, thres=3, use_metrics=None):
    """Implements the second step of the FASTER algorithm.
    
    This function attempts to automatically mark bad epochs by performing
    outlier detection.

    Parameters
    ----------
    epochs : Instance of Epochs
        The epochs to analyze.
    picks : list of int | None
        Channels to operate on. Defaults to EEG channels.
    thres : float
        The threshold value, in standard deviations, to apply. An epoch
        crossing this threshold value is marked as bad. Defaults to 3.
    use_metrics : list of str
        List of metrics to use. Can be any combination of:
            'amplitude', 'variance', 'deviation'
        Defaults to all of them.

    Returns
    -------
    bads : list of int
        The indices of the bad epochs.
    """

    metrics = {
        'amplitude': lambda x: np.mean(np.ptp(x, axis=2), axis=1),
        'deviation': lambda x: np.mean(_deviation(x), axis=1),
        'variance':  lambda x: np.mean(np.var(x, axis=2), axis=1),
    }

    if picks is None:
        picks = mne.pick_types(epochs.info, meg=False, eeg=True,
                               exclude='bads')
    if use_metrics is None:
        use_metrics = metrics.keys()

    data = epochs.get_data()[:, picks, :]

    bads = []
    for m in use_metrics:
        s = metrics[m](data)
        b = find_outliers(s, thres)
        logger.info('Bad by %s:\n\t%s' % (m, b))
        bads.append(b)

    return np.unique(np.concatenate(bads)).tolist()

def _power_gradient(ica, source_data):
    # Compute power spectrum
    f, Ps = scipy.signal.welch(source_data, ica.info['sfreq'])

    # Limit power spectrum to upper frequencies
    Ps = Ps[:, np.searchsorted(f, 25):np.searchsorted(f, 45)]

    # Compute mean gradients
    return np.mean(np.diff(Ps), axis=1)


def faster_bad_components(ica, epochs, thres=3, use_metrics=None):
    """Implements the third step of the FASTER algorithm.
    
    This function attempts to automatically mark bad ICA components by
    performing outlier detection.

    Parameters
    ----------
    ica : Instance of ICA
        The ICA operator, already fitted to the supplied Epochs object.
    epochs : Instance of Epochs
        The untransformed epochs to analyze.
    thres : float
        The threshold value, in standard deviations, to apply. A component
        crossing this threshold value is marked as bad. Defaults to 3.
    use_metrics : list of str
        List of metrics to use. Can be any combination of:
            'eog_correlation', 'kurtosis', 'power_gradient', 'hurst',
            'median_gradient'
        Defaults to all of them.

    Returns
    -------
    bads : list of int
        The indices of the bad components.

    See also
    --------
    ICA.find_bads_ecg
    ICA.find_bads_eog
    """
    source_data = ica.get_sources(epochs).get_data().transpose(1,0,2)
    source_data = source_data.reshape(source_data.shape[0], -1)

    metrics = {
        'eog_correlation': lambda x: x.find_bads_eog(epochs)[1],
        'kurtosis':        lambda x: kurtosis(
                               np.dot(
                                   x.mixing_matrix_.T,
                                   x.pca_components_[:x.n_components_]),
                               axis=1),
        'power_gradient':  lambda x: _power_gradient(x, source_data),
        'hurst':           lambda x: hurst(source_data),
        'median_gradient': lambda x: np.median(np.abs(np.diff(source_data)),
                                               axis=1),
        'line_noise':  lambda x: _freqs_power(source_data,
                                              epochs.info['sfreq'], [50, 60]),
    }

    if use_metrics is None:
        use_metrics = metrics.keys()

    bads = []
    for m in use_metrics:
        scores = np.atleast_2d(metrics[m](ica))
        for s in scores:
            b = find_outliers(s, thres)
            logger.info('Bad by %s:\n\t%s' % (m, b))
            bads.append(b)

    return np.unique(np.concatenate(bads)).tolist()

def faster_bad_channels_in_epochs(epochs, picks=None, thres=3, use_metrics=None):
    """Implements the fourth step of the FASTER algorithm.
    
    This function attempts to automatically mark bad channels in each epochs by
    performing outlier detection.

    Parameters
    ----------
    epochs : Instance of Epochs
        The epochs to analyze.
    picks : list of int | None
        Channels to operate on. Defaults to EEG channels.
    thres : float
        The threshold value, in standard deviations, to apply. An epoch
        crossing this threshold value is marked as bad. Defaults to 3.
    use_metrics : list of str
        List of metrics to use. Can be any combination of:
            'amplitude', 'variance', 'deviation', 'median_gradient'
        Defaults to all of them.

    Returns
    -------
    bads : list of lists of int
        For each epoch, the indices of the bad channels.
    """

    metrics = {
        'amplitude':       lambda x: np.ptp(x, axis=2),
        'deviation':       lambda x: _deviation(x),
        'variance':        lambda x: np.var(x, axis=2),
        'median_gradient': lambda x: np.median(np.abs(np.diff(x)), axis=2),
        'line_noise':      lambda x: _freqs_power(x, epochs.info['sfreq'],
                                                  [50, 60]),
    }

    if picks is None:
        picks = mne.pick_types(epochs.info, meg=False, eeg=True,
                               exclude='bads')
    if use_metrics is None:
        use_metrics = metrics.keys()

    
    data = epochs.get_data()[:, picks, :]

    bads = [[] for i in range(len(epochs))]
    for m in use_metrics:
        s_epochs = metrics[m](data)
        for i, s in enumerate(s_epochs):
            b = [epochs.ch_names[picks[j]] for j in find_outliers(s, thres)]
            logger.info('Epoch %d, Bad by %s:\n\t%s' % (i, m, b))
            bads[i].append(b)

    for i, b in enumerate(bads):
        if len(b) > 0:
            bads[i] = np.unique(np.concatenate(b)).tolist()

    return bads

def run_faster(epochs, thres=3, copy=True):
    """Run the entire FASTER pipeline on the data.
    """
    if copy:
        epochs = epochs.copy()

    # Step one
    logger.info('Step 1: mark bad channels')
    epochs.info['bads'] += faster_bad_channels(epochs, thres=5)

    # Step two
    logger.info('Step 2: mark bad epochs')
    bad_epochs = faster_bad_epochs(epochs, thres=thres)
    good_epochs = list(set(range(len(epochs))).difference(set(bad_epochs)))
    epochs = epochs[good_epochs]

    # Step three (using the build-in MNE functionality for this)
    logger.info('Step 3: mark bad ICA components')
    picks = mne.pick_types(epochs.info, meg=False, eeg=True, eog=True, exclude='bads')
    ica = mne.preprocessing.run_ica(epochs, len(picks), picks=picks, eog_ch=['vEOG', 'hEOG'])
    print(ica.exclude)
    ica.apply(epochs)

    # Step four
    logger.info('Step 4: mark bad channels for each epoch')
    bad_channels_per_epoch = faster_bad_channels_in_epochs(epochs, thres=thres)
    for i, b in enumerate(bad_channels_per_epoch):
        if len(b) > 0:
            epoch = epochs[i]
            epoch.info['bads'] += b
            epoch.interpolate_bads_eeg()
            epochs._data[i, :, :] = epoch._data[0, :, :]

    # Now that the data is clean, apply average reference
    epochs.info['custom_ref_applied'] = False
    epochs, _ = mne.io.set_eeg_reference(epochs)
    epochs.apply_proj()

    # That's all for now
    return epochs
######################################################################################
######################################################################################




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
