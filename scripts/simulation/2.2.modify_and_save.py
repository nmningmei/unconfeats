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
from shutil import rmtree




verbose = 1
batch_size = 16
node = 1
core = 16
mem = 2 * node * core
cput = 12 * node * core
units = [2,5,10,20,50,100,300] # one unit hidden layer cannot learn
dropouts = [0,0.25,0.5,0.75]
activations = ['elu',
               'relu',
               'selu',
               'sigmoid',
               'tanh',
               'linear',
               ]
models = ['alexnet','vgg19','densenet',
#          'inception',
          'mobilenet','resnet']
output_activations = ['softmax','sigmoid',]

temp = np.array(list(itertools.product(*[units,dropouts,models,activations,output_activations])))
df = pd.DataFrame(temp,columns = ['hidden_units','dropouts','model_names','hidden_activations','output_activations'])
df['hidden_units'] = df['hidden_units'].astype(int)
df['dropouts'] = df['dropouts'].astype(float)

#############
train_template = '1.1.train_many_models.py'
decode_template = '2.1.add_noise_to_images_decode_noise_hidden_reprs.py'
scripts_folder = 'batch'
if not os.path.exists(scripts_folder):
    os.mkdir(scripts_folder)
else:
    rmtree(scripts_folder)
    os.mkdir(scripts_folder)
os.mkdir(f'{scripts_folder}/outputs')

from shutil import copyfile
copyfile('utils_deep.py',f'{scripts_folder}/utils_deep.py')

collections = []
first_GPU,second_GPU = [],[]
replace = False # change to second GPU
for ii,row in df.iterrows():

    src = '_{}_{}_{}_{}_{}'.format(*list(row.to_dict().values()))
    scripts = []
    for template in [train_template,decode_template]:
        new_scripts_name = os.path.join(scripts_folder,template.replace('.py',f'{src}.py').replace('1.1.train_many_models','1.1.train'))
        scripts.append(new_scripts_name)
        if ii > df.shape[0]/2 :
            replace = True
            second_GPU.append(new_scripts_name)
        else:
            first_GPU.append(new_scripts_name)
        with open(new_scripts_name,'w') as new_file:
            with open(template,'r') as old_file:
                for line in old_file:
                    if "../" in line:
                        line = line.replace("../","../../")

                    elif "print_train             = True" in line:
                        line = line.replace('True','False')

                    elif "pretrain_model_name     = " in line:
                        line = f"pretrain_model_name     = '{row['model_names']}'\n"

                    elif "hidden_units            = " in line:
                        line = f"hidden_units            = {row['hidden_units']}\n"

                    elif "hidden_func_name        = " in line:
                        line = f"hidden_func_name        = '{row['hidden_activations']}'\n"

                    elif "hidden_dropout          = " in line:
                        line = f"hidden_dropout          = {float(row['dropouts'])}\n"

                    elif "output_activation       = " in line:
                        line = f"output_activation       = '{row['output_activations']}'\n"

                    elif "train_folder            = " in line:
                        line = "train_folder            = 'grayscaled'\n"

                    elif "#plt.switch_backend('agg')" in line:
                        line = "plt.switch_backend('agg')\n"

                    elif "True #" in line:
                        line = line.replace("True","False")
                    elif "idx_GPU = 0" in line:
                        if replace:
                            line = line.replace('0','-1')
                    new_file.write(line)
            old_file.close()
        new_file.close()
    new_batch_script_name = os.path.join(scripts_folder,f'SIM{ii+1}')
    content = f"""#!/bin/bash
#PBS -q bcbl
#PBS -l nodes={node}:ppn={core}
#PBS -l mem={mem}gb
#PBS -l cput={cput}:00:00
#PBS -N SIM{ii+1}
#PBS -o outputs/out_{ii+1}.txt
#PBS -e outputs/err_{ii+1}.txt
cd $PBS_O_WORKDIR
export PATH="/scratch/ningmei/anaconda3/bin:/scratch/ningmei/anaconda3/condabin:$PATH"
source activate keras-2.1.6_tensorflow-2.0.0
pwd
echo {scripts[0].split('/')[-1]}
echo {scripts[1].split('/')[-1]}
python "{scripts[0].split('/')[-1]}"
python "{scripts[1].split('/')[-1]}"
    """
    with open(new_batch_script_name,'w') as f:
        f.write(content)
        f.close()
    collections.append(f"qsub SIM{ii+1}")

with open(f'{scripts_folder}/qsub_jobs.py','w') as f:
    f.write("""import os\nimport time""")

with open(f'{scripts_folder}/qsub_jobs.py','a') as f:
    for ii,line in enumerate(collections):
        if ii == 0:
            f.write(f'\nos.system("{line}")\n')
        else:
            f.write(f'time.sleep(3)\nos.system("{line}")\n')
    f.close()

from glob import glob
all_scripts = glob(os.path.join(scripts_folder,'simulation*.py'))

with open(os.path.join(scripts_folder,'run_all.py'),'w') as f:
    f.write('import os\n')
    for files in all_scripts:
        file_name = files.split('bash/')[-1]
        f.write(f'os.system("python {file_name}")\n')
    f.close()
