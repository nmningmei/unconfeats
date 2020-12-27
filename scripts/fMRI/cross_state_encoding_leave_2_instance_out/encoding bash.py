#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 16:44:04 2020

@author: nmei
"""

import os
import numpy as np
from glob import glob
template = 'encoding model.py'
output_dir = '../encoding_model_bash'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

content = '''
import os
import time
'''
with open(f'{output_dir}/qsub_jobs_encode_LOO.py','w') as f:
    f.write(content)
    f.close()

if not os.path.exists(f'{output_dir}/bash'):
    os.mkdir(f'{output_dir}/bash')

for k in [1,2,3,4,5,6,7]:
    sub                 = f'sub-0{k}'
    nodes               = 1
    cores               = 16
    mem                 = int(6 * cores * nodes)
    time_               = 30 * cores * nodes
    
    for conscious_state_source in ['unconscious','conscious']:
        for conscious_state_target in ['unconscious','conscious']:
            created_file_name = 'LOO_{}_{}_{}.py'.format(
                                                conscious_state_source,
                                                conscious_state_target,
                                                f'sub{k}')
            with open(os.path.join(output_dir,created_file_name),'w') as new_file:
                with open(template,'r') as old_file:
                    for line in old_file:
                        if "sub                     = 'sub-" in line:
                            line = f"sub                     = '{sub}'\n"
                        elif 'conscious_state_source  = ' in line:
                            line = f"conscious_state_source  = '{conscious_state_source}'\n"
                        elif 'conscious_state_target  = ' in line:
                            line = f"conscious_state_target  = '{conscious_state_target}'\n"
                        elif "n_jobs                  = " in line:
                            line = line.replace('-1',f'{cores}')
                        new_file.write(line)
                    old_file.close()
                new_file.close()


            content = f"""
#!/bin/bash
#PBS -q bcbl
#PBS -l nodes={nodes}:ppn={cores}
#PBS -l mem={mem}gb
#PBS -l cput={time_}:00:00
#PBS -N S{sub[-1]}_{conscious_state_source[0]}_{conscious_state_target[0]}
#PBS -o bash/out_sub{sub[-1]}_{conscious_state_source}_{conscious_state_target}.txt
#PBS -e bash/err_sub{sub[-1]}_{conscious_state_source}_{conscious_state_target}.txt
cd $PBS_O_WORKDIR
export PATH="/scratch/ningmei/anaconda3/bin:/scratch/ningmei/anaconda3/condabin:$PATH"
source activate keras-2.1.6_tensorflow-2.0.0
pwd
echo "{sub} {conscious_state_source} --> {conscious_state_target}"

python "{created_file_name}"
"""
            print(content)
            with open(f'{output_dir}/encode_{k}_{conscious_state_source}_{conscious_state_target}_q','w') as f:
                f.write(content)
                f.close()
            with open(f'{output_dir}/qsub_jobs_encode_LOO.py','a') as f:
                line = f'\nos.system("qsub encode_{k}_{conscious_state_source}_{conscious_state_target}_q")\n'
                f.write(line)
                f.close()