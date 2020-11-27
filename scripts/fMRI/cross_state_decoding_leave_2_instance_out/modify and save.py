#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 11:41:01 2019

@author: nmei
"""

template = 'post stats.py'
sub = 'sub-03'

with open(f'post stats ({sub}).py','w') as new_file:
    with open(template, 'r') as old_file:
        for line in old_file:
            if "sub = 'sub-0" in line:
                line = f"sub = '{sub}'\n"
            new_file.write(line)
        old_file.close()
    new_file.close()

with open(f'qsub_{sub}','w') as f:
    content = f"""
#!/bin/bash
#PBS -q bcbl
#PBS -l nodes=1:ppn=16
#PBS -l mem=4gb
#PBS -l cput=240:00:00
#PBS -N {sub}
#PBS -o out_{sub[-1]}.txt
#PBS -e err_{sub[-1]}.txt
cd $PBS_O_WORKDIR
export PATH="/scratch/ningmei/anaconda3/bin:$PATH"
pwd

python "post stats ({sub}).py"
    """
    f.write(content)