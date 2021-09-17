#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 16:14:24 2019

@author: nmei
"""

import os
from glob import glob
import pandas as pd
import numpy as np
from PIL import Image, ImageChops
from tqdm import tqdm
from scipy import ndimage


def trans(img,threshold = 200,value = 0):
    img = img.convert("RGBA")
    datas = img.getdata()
    
    newData = []
    for item in datas:
        if value == 'random':
            pix = np.array(img).flatten()
            pix = pix[pix <= threshold]
            value = np.random.choice(pix,size = 1,)[0]
#            print(value)
        elif value == 'average':
            pix = np.array(img).flatten()
            pix = pix[pix <= threshold]
#            pix = pix[10 <= pix]
            value = int(np.mean(pix))
#            print(value)
        else:
            value = value
        if item[0] > threshold and item[1] > threshold and item[2] > threshold:
            newData.append((255, 255, 255, value))
        else:
            newData.append(item)
    
    img.putdata(newData)
    return img
def white_to_transparency(img):
    x = np.asarray(img.convert('RGBA')).copy()

    x[:, :, 3] = (255 * (x[:, :, :3] != 255).any(axis=2)).astype(np.uint8)

    return Image.fromarray(x)
def scramble(image):
    # convert to numpy array
    image_array = np.asarray(image)
    ImSize = image_array.shape
    # the same random phase for all layers
    RandomPhase = np.angle(np.fft.fft2(np.random.normal(size=(ImSize[0],ImSize[1]))))
    # preallocation
    ImFourier = np.zeros((ImSize[0],ImSize[1],ImSize[2]),dtype=np.complex128)
    Amp = np.zeros((ImSize[0],ImSize[1],ImSize[2]))
    Phase = np.zeros((ImSize[0],ImSize[1],ImSize[2]))
    ImScrambled = np.zeros((ImSize[0],ImSize[1],ImSize[2]),dtype=np.complex128)
    
    # for each color channel/layer
    for layer in range(image_array.shape[-1]):
        # get the fourier space of the layer
        ImFourier[:,:,layer] = np.fft.fft2(image_array[:,:,layer])
        # get the magnitude
        Amp[:,:,layer] = np.abs(ImFourier[:,:,layer])
        # get the phase
        Phase[:,:,layer] = np.angle(ImFourier[:,:,layer])
        # add gaussian noise to the phase
        Phase[:,:,layer] = Phase[:,:,layer] + RandomPhase
        # covert fourier space back to image space
        ImScrambled[:,:,layer] = np.fft.ifft2(Amp[:,:,layer] * np.exp(1j*(Phase[:,:,layer])))
    # take only the real part
    ImScrambled = np.real(ImScrambled)
    return ImScrambled.astype('uint8') # make sure to be in RGB format

figure_dir = '../experiment_stimuli/experiment_images_tilted_2'
processed_dir = '../experiment_stimuli/experiment_images_tilted_scramble_greyscale_2'
if not os.path.exists(processed_dir):
    os.mkdir(processed_dir)
saving_dir = '../experiment_stimuli/bw_bc_bl'
if not os.path.exists(saving_dir):
    os.mkdir(saving_dir)
background_dir = '../experiment_stimuli/experiment_background'
if not os.path.exists(background_dir):
    os.mkdir(background_dir)
categories = {'Living_Things':1,'Nonliving_Things':2}

for category,corrAn in categories.items():
    for subcategory in glob(os.path.join(figure_dir,category,'*')):
        images = glob(subcategory+'/*.jpg')
        np.random.seed(12345)
        sampled = np.random.choice(images,size=len(images),replace=False)
        
        for image_path in tqdm(sampled,desc = f'{subcategory.split("/")[-1]}'):
            temp_dir = image_path.split('/')
            temp_dir[2] = processed_dir.split('/')[-1]
            image_name = temp_dir[-1]
            
            processed = os.path.join(*temp_dir[:-1])
            if not os.path.exists(processed):
                os.makedirs(processed)
            img = Image.open(os.path.join(image_path))
            
            im_tran = trans(img,230,'average');im_tran # take out white background
            img_cropped = white_to_transparency(im_tran)
            
            img_cropped_bw = img_cropped.convert('L')
            temp = []
            for ii in range(5):
                img_scram = Image.fromarray(scramble(img_cropped)).convert('L');img_scram
                temp.append(np.array(img_scram))
                
                a,b = image_name.split('.')
                a = a + '_{}'.format(ii + 1)
                image_name_ = a + '.' + b
                
                img_scram.save(os.path.join(processed,image_name_))
            
            img_cropped_bw_data = np.array(img_cropped_bw)
            img_scram_background = np.array(Image.fromarray(scramble(img_cropped)).convert('L'))
            
            idx_object = np.where(img_cropped_bw_data < 250) # need to play around this number
            img_scram_background[idx_object] = img_cropped_bw_data[idx_object]
            img_scram_back_blur = ndimage.gaussian_filter(img_scram_background, sigma=1);
            img_scram_back_blur = Image.fromarray(img_scram_back_blur)
            
            image_saving_dir = image_path.split('/')
            image_saving_dir[2] = saving_dir.split('/')[-1]
            if not os.path.exists(os.path.join(*image_saving_dir[:-1])):
                os.makedirs(os.path.join(*image_saving_dir[:-1]))
            img_scram_back_blur.save(os.path.join(*image_saving_dir))
            
            background_saving_dir = image_path.split('/')
            background_saving_dir[2] = background_dir.split('/')[-1]
            if not os.path.exists(os.path.join(*background_saving_dir[:-1])):
                os.makedirs(os.path.join(*background_saving_dir[:-1]))
            Image.fromarray(scramble(img_cropped)).convert('L').save(os.path.join(*background_saving_dir))
