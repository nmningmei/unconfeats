#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 17:18:33 2019

@author: nmei
"""

import os
from glob import glob
from PIL import Image,ImageChops
import numpy as np
from utils import scramble
from keras.preprocessing.image import (ImageDataGenerator,
                                       array_to_img,
                                       img_to_array,
                                       load_img)
from scipy import ndimage


working_dir = '../temp/test_image'
saving_dir = '../temp/generated'
if not os.path.exists(working_dir):
    os.makedirs(working_dir)
if not os.path.exists(saving_dir):
    os.makedirs(saving_dir)

size = 300
image_dir = glob(os.path.join(working_dir,'*.*'))[0]
image_name = image_dir.split('/')[-1].split('.')[0]
im = Image.open(image_dir)
im = im.resize((size,size),Image.ANTIALIAS)
im_data = np.array(im)

gen = ImageDataGenerator(
                    featurewise_center=False,#True, 
                    samplewise_center=False, 
                    featurewise_std_normalization=False, 
                    samplewise_std_normalization=False, 
                    zca_whitening=False, 
                    zca_epsilon=1e-06, 
                    rotation_range=30, 
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
                    preprocessing_function=None, 
                    data_format=None, 
                    validation_split=0.0
                )
im_data = img_to_array(im)
# reshape to array rank 4
im_data = im_data.reshape((1,) + im_data.shape)
gen.fit(im_data)
count = 5
images_flow = gen.flow(im_data, batch_size=1)
for ii, new_images in enumerate(images_flow):
    # we access only first image because of batch_size=1
    new_image = array_to_img(new_images[0], scale=True)
    for jj in range(5):
        img_scram = Image.fromarray(scramble(new_image)).convert('L')
        img_scram = Image.fromarray(ndimage.gaussian_filter(np.array(img_scram), 
                                                            sigma=1))
        a= image_name
        a = a + '_{}_{}'.format(ii+1,jj + 1)
        image_name_ = a + '.' + 'jpg'
        
        img_scram.save(os.path.join(saving_dir,image_name_))
    
    img_scram_background = np.array(Image.fromarray(scramble(new_image)).convert('L'))
    new_image = new_image.convert('L')
    idx_object = np.where(np.array(new_image) < 250) # need to play around this number
    img_scram_background[idx_object] = np.array(new_image)[idx_object]
    img_scram_back_blur = ndimage.gaussian_filter(img_scram_background, sigma=1);
    img_scram_back_blur = Image.fromarray(img_scram_back_blur).convert('L')
    img_scram_back_blur.save(os.path.join(saving_dir,'{}_{}.png'.format(image_name,ii+1)))
    print('done')
    if ii >= count:
        break























