import argparse
import os
import shutil
import time
import sys
sys.path.append(os.path.abspath('./'))\



# EcDock
import numpy as np 
from rdkit import Chem

import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.transforms import Compose
from torch_scatter import scatter_sum, scatter_mean
from tqdm.auto import tqdm


import copy
from rdkit import Chem
from rdkit.Chem import AllChem
import copy
from tqdm import tqdm
from rdkit.Geometry.rdGeometry import Point3D
from collections import Counter
import matplotlib.pyplot as plt
import random 
import dill
import json

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import defaultdict

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def compare_rmsd(base_path, file1, file2, file_tail_name):
    #rmsd，，，
    
    data_dict1 = {}
    data_dict2 = {}

    with open(os.path.join(base_path, file1 + file_tail_name), 'rb') as f:
        data_dict1 = dill.load(f)

    with open(os.path.join(base_path, file2 + file_tail_name), 'rb') as f:
        data_dict2 = dill.load(f)

    #
    common_keys = list(set(data_dict1.keys()) & set(data_dict2.keys()))

    rmsd_greater_than_1 = []
    rmsd_greater_than_2 = []
    rmsd_greater_than_3 = []

    #rmsd2，rmsd1,2,3
    for key in common_keys:
        if data_dict1[key] > 2:
            if np.fabs(data_dict1[key] - data_dict2[key]) > 1:
                rmsd_greater_than_1.append(key)
            if np.fabs(data_dict1[key] - data_dict2[key]) > 2:
                rmsd_greater_than_2.append(key)
            if np.fabs(data_dict1[key] - data_dict2[key]) > 3:
                rmsd_greater_than_3.append(key)
    
    with open(os.path.join(base_path, 'rmsd_greater_than_1.txt'), 'w') as f:
        for item in rmsd_greater_than_1:
            f.write(item + '\n')


    with open(os.path.join(base_path, 'rmsd_greater_than_2.txt'), 'w') as f:
        for item in rmsd_greater_than_2:
            f.write(item + '\n')


    with open(os.path.join(base_path, 'rmsd_greater_than_3.txt'), 'w') as f:
        for item in rmsd_greater_than_3:
            f.write(item + '\n')

    print('---------------------------------------------------------')
    print('rmsd_greater_than_1 num:', len(rmsd_greater_than_1))
    print('rmsd_greater_than_1:', rmsd_greater_than_1)
    print('---------------------------------------------------------')
    print('rmsd_greater_than_2 num:', len(rmsd_greater_than_2))
    print('rmsd_greater_than_2:', rmsd_greater_than_2)
    print('---------------------------------------------------------')
    print('rmsd_greater_than_3 num:', len(rmsd_greater_than_3))
    print('rmsd_greater_than_3:', rmsd_greater_than_3)


def file_copy(path, name_list, model):
    #
    data_name = []
    if not name_list:
        for i in os.listdir(path):
            path_ = os.path.join(path, i)
            if os.path.exists(path_) and os.path.isdir(path_) and os.listdir(path_): #
                data_name.append(i)
    else:
        for i in name_list:
            path_ = os.path.join(path, i)
            if os.path.exists(path_) and os.path.isdir(path_) and os.listdir(path_): #
                data_name.append(i)


    t_base_path = os.path.join(os.path.dirname(path), 'tmp')

    # 
    if os.path.exists(t_base_path):
        # 
        shutil.rmtree(t_base_path)
        print(f"Folder '{t_base_path}' has been deleted.")
    
    for name in data_name:

        if model == 'ecdock':
            s_path1 = os.path.join(path, name, 'step24/')
            s_path2 = os.path.join(path, name, f'{name}_protein.pdb')
        elif model == 'unimol':
            s_path1 = os.path.join(path, name)
            s_path2 = os.path.join(path, name, f'{name}_protein.pdb')

        t_path = os.path.join(t_base_path, name)  # 
        os.makedirs(t_path, exist_ok=True)

        # ，
        if os.path.exists(t_path):
            shutil.rmtree(t_path)
            #print('')

        os.makedirs(t_path, exist_ok = True)
        #print('s_path:', s_path)
        #print('t_path:', t_path)
        shutil.copytree(s_path1, t_path, dirs_exist_ok=True)
        shutil.copy(s_path2, t_path)



if __name__ == '__main__':
    #
    name_list =  ['7SGV', '8D5D', '7SZA', '7C0U', '8EX2', '7RSV', '7CL8', '7OEO', '7AKL', '7M6K', '7NGW', '7SSM', '6YMS', '6ZAE', '7JGW', '7MS7', '6W59', '7P1M', '8BRO', '8F4J', '8EAB', '7F8T', '6YRV']
    
    #ecdock，
    #ecdock_path = '/mnt/home/fanzhiguang/47/new_KGDiff-EcDock/posebusters_ecdock_cm_equiformer_step25_interaction_limit4.5ai_retain_fine_No_'
    #ecdock_path = '/mnt/home/fanzhiguang/47/new_KGDiff-EcDock/posebusters_ecdock_cm_equiformer_step25_interaction_limit4.5ai_retrain'
    #ecdock_path = '/mnt/home/fanzhiguang/47/new_KGDiff-EcDock/posebusters_ecdock_cm_equiformer_step25_interaction_limit4.5ai_retrain_fine'
    #ecdock_path = '/mnt/home/fanzhiguang/47/unimol_docking_v2/interface/posebusters_predict_sdf_boxsize10'
    #model = 'unimol'
    ecdock_path = '/mnt/home/fanzhiguang/47/new_KGDiff-EcDock/posebusters_ecdock_cm_equiformer_step10'
    model = 'ecdock'
    name_list = []
    file_copy(ecdock_path, name_list, model)