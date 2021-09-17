#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 16:41:58 2019

@author: nmei
"""

import os
print("resize_cropped_images.py")
os.system('python "resize_cropped_images.py"')
print("tilt_resized_images.py")
os.system('python "tilt_resized_images.py"')
print("generate scrambles, add background, greyscale.py")
os.system('python "generate scrambles, add background, greyscale.py"')
print("generate scrambles, add background, colored.py")
os.system('python "generate scrambles, add background, colored.py"')