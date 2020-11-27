#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 10:47:50 2019

@author: nmei
"""

import os
import numpy as np
from glob import glob
template = 'LOO.py'
output_dir = '../decode_bash'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

content = '''import os
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
    cores               = 18
    mem                 = 4 * cores * nodes
    time_               = 48 * cores * nodes
    stacked_data_dir    = '../../../../data/BOLD_average_BOLD_average_lr/{}/'.format(sub)
    BOLD_data           = np.sort(glob(os.path.join(stacked_data_dir,'*BOLD.npy')))
    
    for ii,BOLD_data_file in enumerate(BOLD_data):
        for conscious_state_source in ['unconscious','conscious']:
            for conscious_state_target in ['unconscious','conscious']:
                mask_name = BOLD_data_file.split('/')[-1].replace('_BOLD.npy','').replace('ctx-','')
                created_file_name = 'D_{}_{}_{}_{}.py'.format(
                                                    mask_name,
                                                    conscious_state_source,
                                                    conscious_state_target,
                                                    f'sub{k}')
                with open(os.path.join(output_dir,created_file_name),'w') as new_file:
                    with open(template,'r') as old_file:
                        for line in old_file:
                            if "sub                     = 'sub-" in line:
                                line = f"sub                     = '{sub}'\n"
                            elif "idx                     = 0" in line:
                                line = line.replace("idx                     = 0",
                                                f"idx                     = {ii}")
                            elif 'conscious_state_source  = ' in line:
                                line = f"conscious_state_source  = '{conscious_state_source}'\n"
                            elif 'conscious_state_target  = ' in line:
                                line = f"conscious_state_target  = '{conscious_state_target}'\n"
                            new_file.write(line)
                        old_file.close()
                    new_file.close()
                content = f"""
#!/bin/bash
#PBS -q bcbl
#PBS -l nodes={nodes}:ppn={cores}
#PBS -l mem={mem}gb
#PBS -l cput={time_}:00:00
#PBS -N S{sub[-1]}_{conscious_state_source[0]}_{conscious_state_target[0]}_{mask_name}
#PBS -o bash/out_sub{sub[-1]}_{mask_name}_{conscious_state_source}_{conscious_state_target}.txt
#PBS -e bash/err_sub{sub[-1]}_{mask_name}_{conscious_state_source}_{conscious_state_target}.txt
cd $PBS_O_WORKDIR
export PATH="/scratch/ningmei/anaconda3/bin:/scratch/ningmei/anaconda3/condabin:$PATH"
source activate keras-2.1.6_tensorflow-2.0.0
pwd
echo "{sub} {mask_name} {conscious_state_source} --> {conscious_state_target}"

python "{created_file_name}"
"""
                print(content)
                with open(f'{output_dir}/decode_{k}_{mask_name}_{conscious_state_source}_{conscious_state_target}_q','w') as f:
                    f.write(content)
                    f.close()
                with open(f'{output_dir}/qsub_jobs_decode_LOO.py','a') as f:
                    line = f'\nos.system("qsub decode_{k}_{mask_name}_{conscious_state_source}_{conscious_state_target}_q")'
                    f.write(line)
                    f.close()








