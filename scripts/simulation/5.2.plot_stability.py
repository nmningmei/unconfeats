#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 08:37:19 2020

@author: nmei
"""

import os
import gc

from glob import glob
from tqdm import tqdm

import numpy as np
import pandas as pd
import seaborn as sns

working_dir = '../stability'
working_data = glob(os.path.join(working_dir,'*','stability*.npy'))

