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


# SDF
def read_molecule_from_sdf(sdf_file):
    try:
        suppl = Chem.SDMolSupplier(sdf_file)
    except Exception as e:  #7nr8_ligand.mol2
        suppl = [Chem.MolFromMol2File(os.path.join(os.path.dirname(sdf_file), sdf_file.split('/')[-2] + '_ligand.mol2'), sanitize=False)]
    
    molecules = [Chem.RemoveHs(mol) for mol in suppl if mol is not None]
    return molecules

# 
def compare_molecules(mol1, mol2):
    # 
    if mol1.GetNumAtoms() != mol2.GetNumAtoms():
        return False
    
    # 
    atoms1 = [atom.GetSymbol() for atom in mol1.GetAtoms()]
    atoms2 = [atom.GetSymbol() for atom in mol2.GetAtoms()]
    
    return sorted(atoms1) == sorted(atoms2)

#，
def judge_mol(sdf_file1, sdf_file2):
    # 
    molecules1 = read_molecule_from_sdf(sdf_file1)
    molecules2 = read_molecule_from_sdf(sdf_file2)

    # SDF，
    mol1 = molecules1[0]
    mol2 = molecules2[0]
    compare_molecules(mol1, mol2)
    """
    # 
    if compare_molecules(mol1, mol2):
        ##print("")
        pass
    else:
        #print('error:', sdf_file1, sdf_file2)
        #raise Exception("")
        w_file.write(f'sdf_file1: {sdf_file1}\n')
        w_file.write(f'sdf_file2: {sdf_file2}\n')
        w_file.write(f'\n')
    """
        


if __name__ == '__main__':
    #
    w_file    = open('error_not_equal.txt', 'w')
    #base_path = '/mnt_191/fanzhiguang/47/mnt/unimol_docking_v2/interface/pdb2020_predict_sdf_boxsize10'
    base_path = '/data/fan_zg/MDocking/Docking_baseline/unimol_docking_v2/interface/BindingNetv2_High_predict_sdf_boxsize10'
    name_list = []

    for bs in os.listdir(base_path):
        if os.path.isdir(os.path.join(base_path, bs)):
            name_list.append(bs)

    reading_error = 0
    error_dict = {}
    for name in tqdm(name_list):
        try:
            sdf_file1 = os.path.join(base_path, name, f'{name}_ligand.sdf')
            sdf_file2 = os.path.join(base_path, name, f'gen_{name}.sdf')
            sdf_file3 = os.path.join(base_path, name, f'org_{name}.sdf')
            judge_mol(sdf_file1,sdf_file2)
            judge_mol(sdf_file1,sdf_file3)
        except Exception as e:
            print(f'reading error: {e}, name: {name}')
            #，，
            #shutil.rmtree(os.path.join(base_path, name))
            reading_error += 1
            w_file.write(name + '\n')
            
            error_dict[name] = e
        
    #print('reading_error_num:', reading_error)

    if reading_error > 0:
        print('， ，')
    else:
        print('， ，')
    w_file.close()
    print('error_dict:', error_dict)