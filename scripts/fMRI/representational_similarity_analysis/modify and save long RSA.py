#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 17:29:59 2020
@author: nmei
"""
import os
import itertools
import numpy as np
import pandas as pd
from shutil import rmtree,copyfile

verbose = 1
node = 1
core = 12
mem = 5
cput = 24

model_names = os.listdir('../../../../data/computer_vision_features_no_background_caltech')

#############
template = 'brain to brain cross validation RSA.py'
scripts_folder = '../BB_CV_RSA'
if not os.path.exists(scripts_folder):
    os.mkdir(scripts_folder)
else:
    rmtree(scripts_folder)
    os.mkdir(scripts_folder)
os.mkdir(f'{scripts_folder}/outputs')
copyfile('utils.py',f'{scripts_folder}/utils.py')

collections = []
count = 1
for sub in [1,2,3,4,5,6,7]:
    for model_name in model_names:
        for conscious_source,conscious_target in zip(['unconscious','conscious','conscious'],
                                                     ['unconscious','unconscious','conscious']):
            for is_chance in ["False","True"]:
                src = '_sub-0{}_{}_{}_{}_{}'.format(sub,model_name,conscious_source,conscious_target,is_chance[0])
                
                new_scripts_name = os.path.join(scripts_folder,template.replace('.py',f'{src}.py'))
                with open(new_scripts_name,'w') as new_file:
                    with open(template,'r') as old_file:
                        for line in old_file:
                            if "# change subject" in line:
                                line = line.replace("sub-01",f"sub-0{sub}")
                            elif "# change source" in line:
                                line = f"    conscious_state_source  = '{conscious_source}'\n"
                            elif "# change target" in line:
                                line = f"    conscious_state_target  = '{conscious_target}'\n"
                            elif "# change model" in line:
                                line = f"    model_name              = '{model_name}'\n"
                            elif "# change chance" in line:
                                line = line.replace("False",is_chance)
                            new_file.write(line)
                        old_file.close()
                    new_file.close()
                new_batch_script_name = os.path.join(scripts_folder,f'RSA{count}')
                
#                content = f"""
##!/bin/bash
#
##$ -cwd
##$ -o outputs/out_{count}.txt
##$ -e outputs/err_{count}.txt
##$ -m be
##$ -M nmei@bcbl.eu
##$ -N RSA{count}
##$ -q long.q
##$ -S /bin/bash
#
#module load python/python3.6 fsl/6.0.0
#
#python "{new_scripts_name.split('/')[-1]}"
#"""
                content = f"""#!/bin/bash
#SBATCH --partition=regular
#SBATCH --job-name=RSA{count}
#SBATCH --cpus-per-task={core}
#SBATCH --nodes={node}
#SBATCH --ntasks-per-node=1
#SBATCH --time={cput}:00:00
#SBATCH --mem-per-cpu={mem}G
#SBATCH --output=outputs/out_{count}.txt
#SBATCH --error=outputs/err_{count}.txt
#SBATCH --mail-user=nmei@bcbl.eu

source /scratch/ningmei/.bashrc
conda activate bcbl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scratch/ningmei/anaconda3/lib
module load FSL/6.0.0-foss-2018b
cd $SLURM_SUBMIT_DIR

pwd
echo {new_scripts_name.split('/')[-1]}
python3 "{new_scripts_name.split('/')[-1]}"
"""
                with open(new_batch_script_name,'w') as f:
                    f.write(content)
                    f.close()
                collections.append(f"qsub RSA{count}")
                count += 1

with open(f'{scripts_folder}/qsub_jobs.py','w') as f:
    f.write("""import os\nimport time""")

with open(f'{scripts_folder}/qsub_jobs.py','a') as f:
    for ii,line in enumerate(collections):
        if ii == 0:
            f.write(f'\nos.system("{line}")\n')
        else:
            f.write(f'time.sleep(3)\nos.system("{line}")\n')
    f.close()
