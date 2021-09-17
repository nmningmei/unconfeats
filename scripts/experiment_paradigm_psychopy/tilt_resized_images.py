#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 16:01:19 2019

@author: nmei
"""

import os
from glob import glob
from PIL import Image
from tqdm import tqdm
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img, img_to_array, load_img


figure_dir = '../experiment_stimuli/experiment_images_resized_2'
processed_dir = '../experiment_stimuli/experiment_images_tilted_2'
if not os.path.exists(processed_dir):
    os.mkdir(processed_dir)
saving_dir = ''
categories = {'Living_Things':1,'Nonliving_Things':2}

count = 100 + 1
for category,corrAn in categories.items():
    for subcategory in glob(os.path.join(figure_dir,category,'*')):
        images = glob(subcategory+'/*.jpg')
        np.random.seed(12345)
        sampled = np.random.choice(images,size=len(images),replace=False)
        
        for image_path in sampled:
            temp_dir = image_path.split('/')
            temp_dir[2] = processed_dir.split('/')[-1]
            image_name = temp_dir[-1]
            
            processed = os.path.join(*temp_dir[:-1])
            
            if not os.path.exists(processed):
                os.makedirs(processed)
            
            gen = ImageDataGenerator(
                                    featurewise_center=False,#True, 
                                    samplewise_center=False, 
                                    featurewise_std_normalization=False, 
                                    samplewise_std_normalization=False, 
                                    zca_whitening=False, 
                                    zca_epsilon=1e-06, 
                                    rotation_range=45, 
                                    width_shift_range=0,#0.1, 
                                    height_shift_range=0,#0.1, 
                                    brightness_range=None, 
                                    shear_range=0.01, 
                                    zoom_range=-0.02, 
                                    channel_shift_range=0.0, 
                                    fill_mode='nearest', 
                                    cval=0.0, 
                                    horizontal_flip=True, 
                                    vertical_flip=False, 
#                                    rescale=(200,200,3), 
                                    preprocessing_function=None, 
                                    data_format=None, 
                                    validation_split=0.0
                                )
            image = img_to_array(load_img(image_path))
            # reshape to array rank 4
            image = image.reshape((1,) + image.shape)
            gen.fit(image)
            
            # let's create infinite flow of images
            images_flow = gen.flow(image, batch_size=1)
            for i, new_images in tqdm(enumerate(images_flow),desc='{}'.format(image_name)):
                # we access only first image because of batch_size=1
                new_image = array_to_img(new_images[0], scale=True)
                new_image.save(os.path.join(processed,'{}_{}.{}'.format(image_name.split('.')[0],
                                                                        i,
                                                                        image_name.split('.')[1])))
                if i >= count:
                    break