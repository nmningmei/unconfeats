#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 17:24:36 2019

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
            print(value)
        elif value == 'average':
            pix = np.array(img).flatten()
            pix = pix[pix <= threshold]
#            pix = pix[10 <= pix]
            value = int(np.mean(pix))
            print(value)
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