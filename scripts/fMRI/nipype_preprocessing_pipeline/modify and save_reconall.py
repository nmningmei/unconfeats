#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 11:19:07 2019

@author: nmei
"""

import os

freesurfer_reconall = '2.freesurfer reconall.py'
if not os.path.exists('reconall'):
    os.mkdir('reconall')
else:
    [os.remove('reconall/'+f) for f in os.listdir('reconall')]
content = f"""
#!/bin/bash

# This is a script to send "{freesurfer_reconall}" as a batch job.
# it works on anatomical dataset

#$ -cwd
#$ -o reconall/out.txt
#$ -e reconall/err.txt
#$ -m be
#$ -q fsl.q
#$ -M nmei@bcbl.eu
#$ -N "reconall"
#$ -S /bin/bash

module load rocks-python-3.6 rocks-fsl-5.0.10 rocks-freesurfer-6.0.0
python "{freesurfer_reconall}"
"""
print(content)
with open('qsub_reconall','w') as f:
    f.write(content)
    f.close()