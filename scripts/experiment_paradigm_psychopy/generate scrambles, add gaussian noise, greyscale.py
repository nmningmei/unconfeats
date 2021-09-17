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

figure_dir = '../experiment_stimuli/experiment_images_tilted_2'
processed_dir = '../experiment_stimuli/experiment_images_tilted_scramble_greyscale_2'
if not os.path.exists(processed_dir):
    os.mkdir(processed_dir)
saving_dir = '../experiment_stimuli/bw_gau_bl'
if not os.path.exists(saving_dir):
    os.mkdir(saving_dir)
categories = {'Living_Things':1,'Nonliving_Things':2}

for category,corrAn in categories.items():
    for subcategory in glob(os.path.join(figure_dir,category,'*')):
        images = glob(subcategory+'/*.jpg')
        np.random.seed(12345)
        sampled = np.random.choice(images,size=len(images),replace=False)
        
        for image_path in tqdm(sampled):
            temp_dir = image_path.split('/')
            temp_dir[2] = processed_dir.split('/')[-1]
            image_name = temp_dir[-1]
            
            processed = os.path.join(*temp_dir[:-1])
            if not os.path.exists(processed):
                os.makedirs(processed)
            img = Image.open(os.path.join(image_path))
            
            im_tran = trans(img,245,'average');im_tran
            img_cropped = white_to_transparency(im_tran)
                
            img_cropped_bw = img_cropped.convert('L')
            
            img_cropped_bw_data = np.array(img_cropped_bw)
            img_cropped_bw_data[img_cropped_bw_data <= 25] = 25
            np.random.seed(12345)
            img_scram_background = np.random.randint(0,255,(300,300))
            
            idx_object = np.where(img_cropped_bw_data < 210) # need to play around this number
            img_scram_background[idx_object] = img_cropped_bw_data[idx_object]
            img_scram_back_blur = ndimage.gaussian_filter(img_scram_background, sigma=1)
            img_scram_back_blur = Image.fromarray(img_scram_back_blur.astype('uint8'))
            
            image_saving_dir = image_path.split('/')
            image_saving_dir[2] = saving_dir.split('/')[-1]
            if not os.path.exists(os.path.join(*image_saving_dir[:-1])):
                os.makedirs(os.path.join(*image_saving_dir[:-1]))
            img_scram_back_blur.save(os.path.join(*image_saving_dir))
