#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 07:03:49 2020

@author: nmei
"""


import os
import numpy as np
from glob import glob
template = 'post stats.py'
output_dir = '../LOO_cross_state_lr_stats_bash'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

content = '''
import os
import time
'''
with open(f'{output_dir}/qsub_jobs_decode_LOO.py','w') as f:
    f.write(content)
    f.close()

if not os.path.exists(f'{output_dir}/bash'):
    os.mkdir(f'{output_dir}/bash')

for k in [1,2,3,4,5,6,7]:
    sub                 = f'sub-0{k}'
    nodes               = 1
    cores               = 16
    mem                 = 3 * cores * nodes
    time_               = 24 * cores * nodes
    created_file_name = 'post_stat_{}.py'.format(f'sub{k}')
    with open(os.path.join(output_dir,created_file_name),'w') as new_file:
        with open(template,'r') as old_file:
            for line in old_file:
                if "sub                 = 'sub-" in line:
                    line = f"sub                 = '{sub}'\n"
                new_file.write(line)
            old_file.close()
        new_file.close()
    content = f"""
#!/bin/bash
#PBS -q bcbl
#PBS -l nodes={nodes}:ppn={cores}
#PBS -l mem={mem}gb
#PBS -l cput={time_}:00:00
#PBS -N PS{sub[-1]}
#PBS -o bash/out_sub{sub[-1]}.txt
#PBS -e bash/err_sub{sub[-1]}.txt
cd $PBS_O_WORKDIR
export PATH="/scratch/ningmei/anaconda3/bin:/scratch/ningmei/anaconda3/condabin:$PATH"
source activate keras-2.1.6_tensorflow-2.0.0
pwd
echo "{sub}"

python "{created_file_name}"
"""
    print(content)
    with open(f'{output_dir}/decode_{k}_q','w') as f:
        f.write(content)
        f.close()
    with open(f'{output_dir}/qsub_jobs_decode_LOO.py','a') as f:
        line = f'\nos.system("qsub decode_{k}_q")\n'
        f.write(line)
        f.close()




















































