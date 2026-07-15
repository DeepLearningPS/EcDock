import os
import sys
import subprocess
import warnings
warnings.simplefilter("ignore")  # 


import os
import argparse
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=str, default="0", help="CUDA_VISIBLE_DEVICES")

args = parser.parse_args()


base_dir = os.getcwd()
dataset_name = 'tmpresault'




cmd = f'CUDA_VISIBLE_DEVICES={args.gpu} python ECDockScore/rtmscore_general.py --refine_flag norefine --data_dir {base_dir}/{dataset_name} \
    --data_name {dataset_name} -gen_pocket -c 10.0 -m {base_dir}/ECDockScore/trained_models/rtmscore_model1.pth --cover 0'

        
os.system(cmd)