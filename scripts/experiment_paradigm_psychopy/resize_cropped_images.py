#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 15:47:38 2019

@author: nmei
"""

import os
from glob import glob
from PIL import Image
import numpy as np

working_dir = '../experiment_cropped_2'
saving_dir = '../experiment_stimuli/experiment_images_resized_2'
if not os.path.exists(saving_dir):
    os.mkdir(saving_dir)


image_dirs = glob(os.path.join(working_dir,'*','*','*.jpg'))


temp = []
for image_dir in image_dirs:
    im = Image.open(image_dir)
    temp.append(im.size)


temp = np.array(temp)

size = 300

for image_dir in image_dirs:
    original = Image.open(image_dir)
    
    resized = original.resize((size,size),Image.ANTIALIAS)
    
    temp = image_dir.split('/')
    temp[1] = 'experiment_stimuli/experiment_images_resized_2'
    
    if not os.path.exists(os.path.join(*temp[:-1])):
        os.makedirs(os.path.join(*temp[:-1]))
    resized.save(os.path.join(*temp))