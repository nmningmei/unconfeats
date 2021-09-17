#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy2 Experiment Builder (v1.85.3),
    on Mon 01 Apr 2019 04:02:24 PM CEST
If you publish work using this script please cite the PsychoPy publications:
    Peirce, JW (2007) PsychoPy - Psychophysics software in Python.
        Journal of Neuroscience Methods, 162(1-2), 8-13.
    Peirce, JW (2009) Generating stimuli for neuroscience using PsychoPy.
        Frontiers in Neuroinformatics, 2:10. doi: 10.3389/neuro.11.010.2008
"""

from __future__ import absolute_import, division
from psychopy import locale_setup, sound, gui, visual, core, data, event, logging
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)
import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle
import os  # handy system and path functions
import sys  # to get file system encoding

# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__)).decode(sys.getfilesystemencoding())
os.chdir(_thisDir)

# Store info about the experiment session
expName = u'Expt_DEF_calibration'  # from the Builder filename that created this script
expInfo = {u'n_square': u'32', u'probeFrames': u'5', u'participant': u'', u'session': u'001', u'image_size': u'256', u'premask_dur': u'20', u'postmask_dur': u'20'}
dlg = gui.DlgFromDict(dictionary=expInfo, title=expName)
if dlg.OK == False:
    core.quit()  # user pressed cancel
expInfo['date'] = data.getDateStr()  # add a simple timestamp
expInfo['expName'] = expName

# Data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
filename = _thisDir + os.sep + u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])

# An ExperimentHandler isn't essential but helps with data saving
thisExp = data.ExperimentHandler(name=expName, version='',
    extraInfo=expInfo, runtimeInfo=None,
    originPath=u'/bcbl/home/home_n-z/nmei/uncon_feat/experiment_scripts/Expt_DEF_calibration.psyexp',
    savePickle=True, saveWideText=True,
    dataFileName=filename)
# save a log file for detail verbose info
logFile = logging.LogFile(filename+'.log', level=logging.EXP)
logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

endExpNow = False  # flag for 'escape' or other condition => quit the exp

# Start Code - component code to be run before the window creation

# Setup the Window
win = visual.Window(
    size=(1920, 1080), fullscr=True, screen=0,
    allowGUI=False, allowStencil=False,
    monitor=u'testMonitor', color=u'black', colorSpace='rgb',
    blendMode='avg', useFBO=True)
# store frame rate of monitor if we can measure it
expInfo['frameRate'] = win.getActualFrameRate()
if expInfo['frameRate'] != None:
    frameDur = 1.0 / round(expInfo['frameRate'])
else:
    frameDur = 1.0 / 60.0  # could not measure, so guess

# Initialize components for Routine "setupTRIGetc"
setupTRIGetcClock = core.Clock()
curr=int(expInfo['probeFrames'])
count=0

n_total = 96
premask_dur = int(expInfo['premask_dur'])
postmask_dur = int(expInfo['postmask_dur'])
n_square = int(expInfo['n_square'])
image_size = int(expInfo['image_size'])



# Initialize components for Routine "introduction"
introductionClock = core.Clock()
text = visual.TextStim(win=win, name='text',
    text='Empezando',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1,
    depth=0.0);
text_2 = visual.TextStim(win=win, name='text_2',
    text='+',
    font='Arial',
    pos=[0, 0], height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1,
    depth=-1.0);

# Initialize components for Routine "premask"
premaskClock = core.Clock()
fixation = visual.TextStim(win=win, name='fixation',
    text='+',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1,
    depth=0.0);
blank = visual.TextStim(win=win, name='blank',
    text=None,
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1,
    depth=-1.0);
premask_1 = visual.GratingStim(
    win=win, name='premask_1',units='pix', 
    tex=np.random.rand(n_square,n_square) * 2 -1, mask=None,
    ori=0, pos=(0, 0), size=(image_size, image_size), sf=None, phase=1.0,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    texRes=128, interpolate=False, depth=-2.0)

# Initialize components for Routine "probe_routine"
probe_routineClock = core.Clock()

probe = visual.ImageStim(
    win=win, name='probe',units='pix', 
    image='sin', mask=None,
    ori=0, pos=(0, 0), size=(image_size, image_size),
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-1.0)

# Initialize components for Routine "postmask"
postmaskClock = core.Clock()
postmask_1 = visual.GratingStim(
    win=win, name='postmask_1',units='pix', 
    tex=np.random.rand(n_square,n_square) * 2 -1, mask=None,
    ori=0, pos=(0, 0), size=(image_size, image_size), sf=None, phase=1.0,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    texRes=128, interpolate=False, depth=0.0)

# Initialize components for Routine "jitter_delay"
jitter_delayClock = core.Clock()

delay_post_mask = visual.TextStim(win=win, name='delay_post_mask',
    text='+',
    font='Arial',
    pos=[0, 0], height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1,
    depth=-1.0);

# Initialize components for Routine "response_routine"
response_routineClock = core.Clock()

tell_response = visual.TextStim(win=win, name='tell_response',
    text='default text',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1,
    depth=-2.0);

# Initialize components for Routine "visibility"
visibilityClock = core.Clock()
tell_visible = visual.TextStim(win=win, name='tell_visible',
    text='?',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1,
    depth=-1.0);

response_no_experience = []

# Initialize components for Routine "post_trial_jitter"
post_trial_jitterClock = core.Clock()
post_fixation = visual.TextStim(win=win, name='post_fixation',
    text='+',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1,
    depth=0.0);


# Initialize components for Routine "show_message"
show_messageClock = core.Clock()

text_3 = visual.TextStim(win=win, name='text_3',
    text='default text',
    font='Arial',
    pos=[0, 0], height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1,
    depth=-1.0);

# Initialize components for Routine "End_experiment"
End_experimentClock = core.Clock()
The_End = visual.TextStim(win=win, name='The_End',
    text=None,
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1,
    depth=0.0);


# Create some handy timers
globalClock = core.Clock()  # to track the time since experiment started
routineTimer = core.CountdownTimer()  # to track time remaining of each (non-slip) routine 

# ------Prepare to start Routine "setupTRIGetc"-------
t = 0
setupTRIGetcClock.reset()  # clock
frameN = -1
continueRoutine = True
# update component parameters for each repeat

globalClock.reset()
startTime = globalClock.getTime() 
# keep track of which components have finished
setupTRIGetcComponents = []
for thisComponent in setupTRIGetcComponents:
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED

# -------Start Routine "setupTRIGetc"-------
while continueRoutine:
    # get current time
    t = setupTRIGetcClock.getTime()
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in setupTRIGetcComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # check for quit (the Esc key)
    if endExpNow or event.getKeys(keyList=["escape"]):
        core.quit()
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "setupTRIGetc"-------
for thisComponent in setupTRIGetcComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)

# the Routine "setupTRIGetc" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "introduction"-------
t = 0
introductionClock.reset()  # clock
frameN = -1
continueRoutine = True
routineTimer.add(3.000000)
# update component parameters for each repeat
# keep track of which components have finished
introductionComponents = [text, text_2]
for thisComponent in introductionComponents:
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED

# -------Start Routine "introduction"-------
while continueRoutine and routineTimer.getTime() > 0:
    # get current time
    t = introductionClock.getTime()
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *text* updates
    if t >= 0.0 and text.status == NOT_STARTED:
        # keep track of start time/frame for later
        text.tStart = t
        text.frameNStart = frameN  # exact frame index
        text.setAutoDraw(True)
    frameRemains = 0.0 + 1- win.monitorFramePeriod * 0.75  # most of one frame period left
    if text.status == STARTED and t >= frameRemains:
        text.setAutoDraw(False)
    
    # *text_2* updates
    if t >= 1 and text_2.status == NOT_STARTED:
        # keep track of start time/frame for later
        text_2.tStart = t
        text_2.frameNStart = frameN  # exact frame index
        text_2.setAutoDraw(True)
    frameRemains = 1 + 2- win.monitorFramePeriod * 0.75  # most of one frame period left
    if text_2.status == STARTED and t >= frameRemains:
        text_2.setAutoDraw(False)
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in introductionComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # check for quit (the Esc key)
    if endExpNow or event.getKeys(keyList=["escape"]):
        core.quit()
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "introduction"-------
for thisComponent in introductionComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)

# set up handler to look after randomisation of conditions etc
trials = data.TrialHandler(nReps=1, method='random', 
    extraInfo=expInfo, originPath=-1,
    trialList=data.importConditions(u'csvs/calibration.csv'),
    seed=None, name='trials')
thisExp.addLoop(trials)  # add the loop to the experiment
thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
if thisTrial != None:
    for paramName in thisTrial.keys():
        exec(paramName + '= thisTrial.' + paramName)

for thisTrial in trials:
    currentLoop = trials
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial.keys():
            exec(paramName + '= thisTrial.' + paramName)
    
    # ------Prepare to start Routine "premask"-------
    t = 0
    premaskClock.reset()  # clock
    frameN = -1
    continueRoutine = True
    # update component parameters for each repeat
    premask_1.setPhase(np.random.uniform(0,1,2).round(1))
    # keep track of which components have finished
    premaskComponents = [fixation, blank, premask_1]
    for thisComponent in premaskComponents:
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    
    # -------Start Routine "premask"-------
    while continueRoutine:
        # get current time
        t = premaskClock.getTime()
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *fixation* updates
        if t >= 0 and fixation.status == NOT_STARTED:
            # keep track of start time/frame for later
            fixation.tStart = t
            fixation.frameNStart = frameN  # exact frame index
            fixation.setAutoDraw(True)
        frameRemains = 0 + 0.5- win.monitorFramePeriod * 0.75  # most of one frame period left
        if fixation.status == STARTED and t >= frameRemains:
            fixation.setAutoDraw(False)
        
        # *blank* updates
        if (fixation.status==FINISHED) and blank.status == NOT_STARTED:
            # keep track of start time/frame for later
            blank.tStart = t
            blank.frameNStart = frameN  # exact frame index
            blank.setAutoDraw(True)
        if blank.status == STARTED and t >= (blank.tStart + 0.5):
            blank.setAutoDraw(False)
        
        # *premask_1* updates
        if (blank.status == FINISHED) and premask_1.status == NOT_STARTED:
            # keep track of start time/frame for later
            premask_1.tStart = t
            premask_1.frameNStart = frameN  # exact frame index
            premask_1.setAutoDraw(True)
        if premask_1.status == STARTED and frameN >= (premask_1.frameNStart + premask_dur):
            premask_1.setAutoDraw(False)
        if premask_1.status == STARTED:  # only update if drawing
            premask_1.setTex(np.random.rand(n_square,n_square) * 2 -1, log=False)
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in premaskComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # check for quit (the Esc key)
        if endExpNow or event.getKeys(keyList=["escape"]):
            core.quit()
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "premask"-------
    for thisComponent in premaskComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # the Routine "premask" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # ------Prepare to start Routine "probe_routine"-------
    t = 0
    probe_routineClock.reset()  # clock
    frameN = -1
    continueRoutine = True
    # update component parameters for each repeat
    trials.addData("image_onset_time", globalClock.getTime() - startTime)
    
    probe.setImage(probe_path)
    # keep track of which components have finished
    probe_routineComponents = [probe]
    for thisComponent in probe_routineComponents:
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    
    # -------Start Routine "probe_routine"-------
    while continueRoutine:
        # get current time
        t = probe_routineClock.getTime()
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        
        # *probe* updates
        if frameN >= 0.0 and probe.status == NOT_STARTED:
            # keep track of start time/frame for later
            probe.tStart = t
            probe.frameNStart = frameN  # exact frame index
            probe.setAutoDraw(True)
        if probe.status == STARTED and frameN >= (probe.frameNStart + curr):
            probe.setAutoDraw(False)
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in probe_routineComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # check for quit (the Esc key)
        if endExpNow or event.getKeys(keyList=["escape"]):
            core.quit()
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "probe_routine"-------
    for thisComponent in probe_routineComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    
    # the Routine "probe_routine" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # ------Prepare to start Routine "postmask"-------
    t = 0
    postmaskClock.reset()  # clock
    frameN = -1
    continueRoutine = True
    # update component parameters for each repeat
    postmask_1.setPhase(np.random.uniform(0,1,2).round(1))
    # keep track of which components have finished
    postmaskComponents = [postmask_1]
    for thisComponent in postmaskComponents:
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    
    # -------Start Routine "postmask"-------
    while continueRoutine:
        # get current time
        t = postmaskClock.getTime()
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *postmask_1* updates
        if frameN >= 0 and postmask_1.status == NOT_STARTED:
            # keep track of start time/frame for later
            postmask_1.tStart = t
            postmask_1.frameNStart = frameN  # exact frame index
            postmask_1.setAutoDraw(True)
        if postmask_1.status == STARTED and frameN >= (postmask_1.frameNStart + postmask_dur):
            postmask_1.setAutoDraw(False)
        if postmask_1.status == STARTED:  # only update if drawing
            postmask_1.setTex(np.random.rand(n_square,n_square) * 2 -1, log=False)
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in postmaskComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # check for quit (the Esc key)
        if endExpNow or event.getKeys(keyList=["escape"]):
            core.quit()
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "postmask"-------
    for thisComponent in postmaskComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # the Routine "postmask" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # ------Prepare to start Routine "jitter_delay"-------
    t = 0
    jitter_delayClock.reset()  # clock
    frameN = -1
    continueRoutine = True
    routineTimer.add(0.500000)
    # update component parameters for each repeat
    jitter_dur_options = [1.5,2.0,2.5,3.0,3.5]
    pick_chance = [0.5,0.25,0.1,0.1,0.05]
    jitter_delay_dur = np.random.choice(jitter_dur_options,size=1,p=pick_chance)[0]
    
    trials.addData("jitter1", jitter_delay_dur)
    # keep track of which components have finished
    jitter_delayComponents = [delay_post_mask]
    for thisComponent in jitter_delayComponents:
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    
    # -------Start Routine "jitter_delay"-------
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = jitter_delayClock.getTime()
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        
        # *delay_post_mask* updates
        if t >= 0.0 and delay_post_mask.status == NOT_STARTED:
            # keep track of start time/frame for later
            delay_post_mask.tStart = t
            delay_post_mask.frameNStart = frameN  # exact frame index
            delay_post_mask.setAutoDraw(True)
        frameRemains = 0.0 + 0.5- win.monitorFramePeriod * 0.75  # most of one frame period left
        if delay_post_mask.status == STARTED and t >= frameRemains:
            delay_post_mask.setAutoDraw(False)
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in jitter_delayComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # check for quit (the Esc key)
        if endExpNow or event.getKeys(keyList=["escape"]):
            core.quit()
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "jitter_delay"-------
    for thisComponent in jitter_delayComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    
    
    # ------Prepare to start Routine "response_routine"-------
    t = 0
    response_routineClock.reset()  # clock
    frameN = -1
    continueRoutine = True
    routineTimer.add(1.500000)
    # update component parameters for each repeat
    trials.addData("discrim_resptime", globalClock.getTime() - startTime)
    
    resp_options = [['nV_V',['Nonliving_Things','Living_Things']],
                    ['V_nV',['Living_Things','Nonliving_Things']]]
    
    idx = np.random.choice([0,1])
    msg = '{}'.format(resp_options[idx][0])
    
    trials.addData("response_window", resp_options[idx][0])
    response = event.BuilderKeyResponse()
    tell_response.setText(msg

)
    # keep track of which components have finished
    response_routineComponents = [response, tell_response]
    for thisComponent in response_routineComponents:
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    
    # -------Start Routine "response_routine"-------
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = response_routineClock.getTime()
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        
        # *response* updates
        if t >= 0.0 and response.status == NOT_STARTED:
            # keep track of start time/frame for later
            response.tStart = t
            response.frameNStart = frameN  # exact frame index
            response.status = STARTED
            # keyboard checking is just starting
            win.callOnFlip(response.clock.reset)  # t=0 on next screen flip
            event.clearEvents(eventType='keyboard')
        frameRemains = 0.0 + 1.5- win.monitorFramePeriod * 0.75  # most of one frame period left
        if response.status == STARTED and t >= frameRemains:
            response.status = STOPPED
        if response.status == STARTED:
            theseKeys = event.getKeys(keyList=['1', '2'])
            
            # check for quit:
            if "escape" in theseKeys:
                endExpNow = True
            if len(theseKeys) > 0:  # at least one key was pressed
                response.keys = theseKeys[-1]  # just the last key pressed
                response.rt = response.clock.getTime()
        
        # *tell_response* updates
        if frameN >= 0.0 and tell_response.status == NOT_STARTED:
            # keep track of start time/frame for later
            tell_response.tStart = t
            tell_response.frameNStart = frameN  # exact frame index
            tell_response.setAutoDraw(True)
        if tell_response.status == STARTED and t >= (tell_response.tStart + 1.5):
            tell_response.setAutoDraw(False)
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in response_routineComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # check for quit (the Esc key)
        if endExpNow or event.getKeys(keyList=["escape"]):
            core.quit()
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "response_routine"-------
    for thisComponent in response_routineComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    temp_correctAns = np.where(np.array(resp_options[idx][1]) == category)[0][0]+1
    
    trials.addData('correctAns',temp_correctAns)
    
    # objective accuracy
    
    if (response.keys == str(temp_correctAns)) or (response.keys == temp_correctAns):
       temp_corr = 1
    else:
        temp_corr = 0
    
    trials.addData('response.corr' , temp_corr)
    
    # check responses
    if response.keys in ['', [], None]:  # No response was made
        response.keys=None
    trials.addData('response.keys',response.keys)
    if response.keys != None:  # we had a response
        trials.addData('response.rt', response.rt)
    
    # ------Prepare to start Routine "visibility"-------
    t = 0
    visibilityClock.reset()  # clock
    frameN = -1
    continueRoutine = True
    routineTimer.add(1.500000)
    # update component parameters for each repeat
    visible = event.BuilderKeyResponse()
    trials.addData("visibil_resptime", globalClock.getTime() - startTime)
    
    
    # keep track of which components have finished
    visibilityComponents = [visible, tell_visible]
    for thisComponent in visibilityComponents:
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    
    # -------Start Routine "visibility"-------
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = visibilityClock.getTime()
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *visible* updates
        if t >= 0.0 and visible.status == NOT_STARTED:
            # keep track of start time/frame for later
            visible.tStart = t
            visible.frameNStart = frameN  # exact frame index
            visible.status = STARTED
            # keyboard checking is just starting
            win.callOnFlip(visible.clock.reset)  # t=0 on next screen flip
            event.clearEvents(eventType='keyboard')
        frameRemains = 0.0 + 1.5- win.monitorFramePeriod * 0.75  # most of one frame period left
        if visible.status == STARTED and t >= frameRemains:
            visible.status = STOPPED
        if visible.status == STARTED:
            theseKeys = event.getKeys(keyList=['1', '2', '3'])
            
            # check for quit:
            if "escape" in theseKeys:
                endExpNow = True
            if len(theseKeys) > 0:  # at least one key was pressed
                visible.keys = theseKeys[-1]  # just the last key pressed
                visible.rt = visible.clock.getTime()
                # was this 'correct'?
                if (visible.keys == str('1')) or (visible.keys == '1'):
                    visible.corr = 1
                else:
                    visible.corr = 0
        
        # *tell_visible* updates
        if t >= 0.0 and tell_visible.status == NOT_STARTED:
            # keep track of start time/frame for later
            tell_visible.tStart = t
            tell_visible.frameNStart = frameN  # exact frame index
            tell_visible.setAutoDraw(True)
        frameRemains = 0.0 + 1.5- win.monitorFramePeriod * 0.75  # most of one frame period left
        if tell_visible.status == STARTED and t >= frameRemains:
            tell_visible.setAutoDraw(False)
        
        
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in visibilityComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # check for quit (the Esc key)
        if endExpNow or event.getKeys(keyList=["escape"]):
            core.quit()
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "visibility"-------
    for thisComponent in visibilityComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # check responses
    if visible.keys in ['', [], None]:  # No response was made
        visible.keys=None
        # was no response the correct answer?!
        if str('1').lower() == 'none':
           visible.corr = 1  # correct non-response
        else:
           visible.corr = 0  # failed to respond (incorrectly)
    # store data for trials (TrialHandler)
    trials.addData('visible.keys',visible.keys)
    trials.addData('visible.corr', visible.corr)
    if visible.keys != None:  # we had a response
        trials.addData('visible.rt', visible.rt)
    
    
    trials.addData('probeFrames',curr)
    
    count += 1
    if visible.keys == '1':# invisible
            curr += np.random.choice([1,2,3],size=1)[0]
            if curr < 1:  curr = 1
            response_no_experience.append(temp_corr)
    elif visible.keys == '2':# partially aware
            curr -= 1
            if curr < 1:  curr = 1 
    elif visible.keys == '3': # visible
            curr -= np.random.choice([2,3],size=1,p=[0.5,0.5])[0]
            if curr < 1: curr = 1
        
    #elif visible.keys == '4': # fully visible
    #        curr -= 3
    #        if curr < 1: curr = 1
    
    
    # ------Prepare to start Routine "post_trial_jitter"-------
    t = 0
    post_trial_jitterClock.reset()  # clock
    frameN = -1
    continueRoutine = True
    routineTimer.add(1.000000)
    # update component parameters for each repeat
    jitter_dur_options = [6.0,6.5,7.0,7.5,8.0]
    pick_chance = [0.5,0.25,0.1,0.1,0.05]
    post_trial_jitter_duration = np.random.choice(jitter_dur_options,size=1,p=pick_chance)[0]
    
    
    trials.addData("jitter2", post_trial_jitter_duration)
    # keep track of which components have finished
    post_trial_jitterComponents = [post_fixation]
    for thisComponent in post_trial_jitterComponents:
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    
    # -------Start Routine "post_trial_jitter"-------
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = post_trial_jitterClock.getTime()
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *post_fixation* updates
        if t >= 0.0 and post_fixation.status == NOT_STARTED:
            # keep track of start time/frame for later
            post_fixation.tStart = t
            post_fixation.frameNStart = frameN  # exact frame index
            post_fixation.setAutoDraw(True)
        frameRemains = 0.0 + 1- win.monitorFramePeriod * 0.75  # most of one frame period left
        if post_fixation.status == STARTED and t >= frameRemains:
            post_fixation.setAutoDraw(False)
        
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in post_trial_jitterComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # check for quit (the Esc key)
        if endExpNow or event.getKeys(keyList=["escape"]):
            core.quit()
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "post_trial_jitter"-------
    for thisComponent in post_trial_jitterComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    
    
    # ------Prepare to start Routine "show_message"-------
    t = 0
    show_messageClock.reset()  # clock
    frameN = -1
    continueRoutine = True
    routineTimer.add(2.000000)
    # update component parameters for each repeat
    
    meanacc = trials.data['response.corr'].mean()
    meanvis = trials.data['visible.corr'].mean()
    print_msg="{:.2f}, {}".format(
    meanvis,curr,)
    print("mean vis = {:.2f}, frames = {}, p(correct) = {:.2f}".format(
        meanvis,curr,meanacc,))
    #msg = msg + '\nkey={},cor={}'.format(response.keys,str(temp_correctAns))
    #msg WOULD BE TO DISPLAY IN A TXT OBJECT - THAT HAS BEEN DELETED
    text_3.setText(print_msg)
    # keep track of which components have finished
    show_messageComponents = [text_3]
    for thisComponent in show_messageComponents:
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    
    # -------Start Routine "show_message"-------
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = show_messageClock.getTime()
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        
        # *text_3* updates
        if t >= 0.0 and text_3.status == NOT_STARTED:
            # keep track of start time/frame for later
            text_3.tStart = t
            text_3.frameNStart = frameN  # exact frame index
            text_3.setAutoDraw(True)
        frameRemains = 0.0 + 2- win.monitorFramePeriod * 0.75  # most of one frame period left
        if text_3.status == STARTED and t >= frameRemains:
            text_3.setAutoDraw(False)
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in show_messageComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # check for quit (the Esc key)
        if endExpNow or event.getKeys(keyList=["escape"]):
            core.quit()
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "show_message"-------
    for thisComponent in show_messageComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    
    thisExp.nextEntry()
    
# completed 1 repeats of 'trials'

# get names of stimulus parameters
if trials.trialList in ([], [None], None):
    params = []
else:
    params = trials.trialList[0].keys()
# save data for this loop
trials.saveAsText(filename + 'trials.csv', delim=',',
    stimOut=params,
    dataOut=['n','all_mean','all_std', 'all_raw'])

# ------Prepare to start Routine "End_experiment"-------
t = 0
End_experimentClock.reset()  # clock
frameN = -1
continueRoutine = True
routineTimer.add(3.000000)
# update component parameters for each repeat

# keep track of which components have finished
End_experimentComponents = [The_End]
for thisComponent in End_experimentComponents:
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED

# -------Start Routine "End_experiment"-------
while continueRoutine and routineTimer.getTime() > 0:
    # get current time
    t = End_experimentClock.getTime()
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *The_End* updates
    if t >= 0.0 and The_End.status == NOT_STARTED:
        # keep track of start time/frame for later
        The_End.tStart = t
        The_End.frameNStart = frameN  # exact frame index
        The_End.setAutoDraw(True)
    frameRemains = 0.0 + 3- win.monitorFramePeriod * 0.75  # most of one frame period left
    if The_End.status == STARTED and t >= frameRemains:
        The_End.setAutoDraw(False)
    
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in End_experimentComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # check for quit (the Esc key)
    if endExpNow or event.getKeys(keyList=["escape"]):
        core.quit()
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "End_experiment"-------
for thisComponent in End_experimentComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)









print(globalClock.getTime() - startTime)
print("mean vis = {:.2f}, frames = {}, p(correct) = {:.2f}".format(
    meanvis,curr,meanacc,))
print('performance vis = 1 is {:.2f}'.format(np.mean(response_no_experience)))
# these shouldn't be strictly necessary (should auto-save)
thisExp.saveAsWideText(filename+'.csv')
thisExp.saveAsPickle(filename)
logging.flush()
# make sure everything is closed down
thisExp.abort()  # or data files will save again on exit
win.close()
core.quit()
