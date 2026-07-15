import csv
import pandas as pd
import os
from collections import defaultdict
from ordered_set import OrderedSet
import shutil
import torch
import pickle
import dill


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



import numpy as np
import shutil
import os

import oddt
from oddt.toolkits.extras import rdkit as o_rdkit
from rdkit import Chem
from oddt.docking.AutodockVina import write_vina_pdbqt
import subprocess
import time


import glob
from multiprocessing import Pool
from functools import partial

import gzip
import shutil

from Bio.PDB import PDBParser, PDBIO, Select
from Bio import PDB

import pdbreader
from rdkit.Chem import PDBWriter

from scipy.spatial import distance_matrix





def rmsd_func(predict_coords, holo_coords, cutoff):
    cutoff_flag = predict_coords < cutoff #，，,，np.sqrt(np.sum((predict_coords[cutoff_flag] - holo_coords[cutoff_flag])**2) / sz[0])

    sz = holo_coords.shape
    rmsd = np.sqrt(np.sum((predict_coords[cutoff_flag] - holo_coords[cutoff_flag])**2) / sz[0]) #np.sum，, 
    #rmsd = np.sqrt(np.mean(np.sum((predict_coords[cutoff_flag] - holo_coords[cutoff_flag])**2, axis = -1)))# 
    
    
    return rmsd


def rmsd_func_sym(holo_coords: np.ndarray, predict_coords: np.ndarray, mol = None) -> float:
    '''
    RMSD（Symmetric RMSD）。，，RMSD
    '''
    """ Symmetric RMSD for molecules. """
    sz = holo_coords.shape
    if mol is not None:
        # get stereochem-unaware permutations: (P, N)
        base_perms = np.array(mol.GetSubstructMatches(mol, uniquify=False))
        # filter for valid stereochem only
        chem_order = np.array(list(Chem.rdmolfiles.CanonicalRankAtoms(mol, breakTies=False)))
        perms_mask = (chem_order[base_perms] == chem_order[None]).sum(-1) == mol.GetNumAtoms()
        base_perms = base_perms[perms_mask]
        noh_mask = np.array([a.GetAtomicNum() != 1 for a in mol.GetAtoms()])
        # (N, 3), (N, 3) -> (P, N, 3), ((), N, 3) -> (P,) -> min((P,))
        best_rmsd = np.inf
        for perm in base_perms:
            rmsd = np.sqrt(np.sum((predict_coords[perm[noh_mask]] - holo_coords) ** 2) / sz[0])
            if rmsd < best_rmsd:
                best_rmsd = rmsd

        rmsd = best_rmsd
    else:
        rmsd = np.sqrt(np.sum((predict_coords - holo_coords) ** 2) / sz[0]) #
        #rmsd = np.sqrt(np.mean(np.sum((predict_coords - holo_coords)**2))) #
    return rmsd






def cross_distance(mol_pos, pocket_pos):
    #
    dist = distance_matrix(mol_pos, pocket_pos).astype(np.float32)
    return dist



def compute_cross_distance_rmsd(file_path, num = 10000):
    name_list = []
    data_holo_cross_distance_list    = []
    data_predict_cross_distance_list = []
    data_target_cross_distance_list  = []

    '''
        data_dict['holo_coords_list'] = holo_coords_list
        data_dict['coords_predict_list'] = coords_predict_list
        data_dict['pocket_coords_list'] = pocket_coords_list
        data_dict['cross_distance_list'] = cross_distance_list
    '''
    for i in os.listdir(file_path)[:num]:
        path = os.path.join(file_path, i)
        if os.path.exists(path) and os.path.isdir(path) and os.listdir(path): #
            name_list.append(i)
            with open(f'{path}/interaction_{i}.pkl', 'rb') as f:
                data_dict = dill.load(f)
                holo_coords_list    = data_dict['holo_coords_list']
                coords_predict_list = data_dict['coords_predict_list']
                pocket_coords_list  = data_dict['pocket_coords_list']
                target_cross_distance_list = data_dict['cross_distance_list'] #

                holo_cross_distance_list    = []
                predict_cross_distance_list = []

                for h, pr, po in zip(holo_coords_list, coords_predict_list, pocket_coords_list):
                    holo_cross_distance     = cross_distance(h, po)      #
                    predict_cross_distance  = cross_distance(pr, po)     #
                    holo_cross_distance_list.append(holo_cross_distance) 
                    predict_cross_distance_list.append(predict_cross_distance)
                
                data_holo_cross_distance_list.append(holo_cross_distance_list)
                data_predict_cross_distance_list.append(predict_cross_distance_list)
                data_target_cross_distance_list.append(target_cross_distance_list) 
        
    
    #rmsd
    predict_ligand_rmsd_cutoff_dict     = defaultdict(list) #rmsd
    predict_distance_rmsd_cutoff_dict   = defaultdict(list) #rmsd
    predict_ligand_distance_rmsd_cutoff_dict = defaultdict(list) #rmsd
    predict_distance_ligand_rmsd_cutoff_dict = defaultdict(list) #rmsd
    for cutoff in [0,1,2,3,4,5,6,7,8,9,10,11,12,10000]: # 10000
        
        predict_ligand_rmsd    = []
        predict_distance_rmsd  = []
        predict_ligand_distance_rmsd = []
        predict_distance_ligand_rmsd = []
        #
        for dt_h_list, dt_pr_list, dt_ta_list in zip(data_holo_cross_distance_list, data_predict_cross_distance_list, data_target_cross_distance_list):
            sub_predict_ligand_rmsd    = []
            sub_predict_distance_rmsd  = []
            sub_predict_ligand_distance_rmsd = []
            sub_predict_distance_ligand_rmsd = []
            #40，rmsd
            for h, pr, ta in zip(dt_h_list, dt_pr_list, dt_ta_list):
                #，dt_h_list
                predict_ligand_r    = rmsd_func(pr, h, cutoff)
                predict_distance_r  = rmsd_func(ta, h, cutoff)
                predict_ligand_distance_r  = rmsd_func(pr, ta, cutoff)
                predict_distance_ligand_r  = rmsd_func(ta, pr, cutoff)

                sub_predict_ligand_rmsd.append(predict_ligand_r)
                sub_predict_distance_rmsd.append(predict_distance_r)
                sub_predict_ligand_distance_rmsd.append(predict_ligand_distance_r)
                sub_predict_distance_ligand_rmsd.append(predict_distance_ligand_r)
        
            predict_ligand_rmsd.append(np.mean(sub_predict_ligand_rmsd))
            predict_distance_rmsd.append(np.mean(sub_predict_distance_rmsd))
            predict_ligand_distance_rmsd.append(np.mean(sub_predict_ligand_distance_rmsd))
            predict_distance_ligand_rmsd.append(np.mean(sub_predict_distance_ligand_rmsd))
        
        predict_ligand_rmsd_cutoff_dict[f'cutoff{cutoff}'].append(round(np.mean(predict_ligand_rmsd), 4))
        predict_distance_rmsd_cutoff_dict[f'cutoff{cutoff}'].append(round(np.mean(predict_distance_rmsd), 4))
        predict_ligand_distance_rmsd_cutoff_dict[f'cutoff{cutoff}'].append(round(np.mean(predict_ligand_distance_rmsd), 4))
        predict_distance_ligand_rmsd_cutoff_dict[f'cutoff{cutoff}'].append(round(np.mean(predict_distance_ligand_rmsd), 4))

    resault_dict = {}
    resault_dict['rmsd']  = predict_ligand_rmsd_cutoff_dict
    resault_dict['rmsd']  = predict_distance_rmsd_cutoff_dict
    resault_dict['rmsd']  = predict_ligand_distance_rmsd_cutoff_dict
    resault_dict['rmsd']  = predict_distance_ligand_rmsd_cutoff_dict

    #json_str = json.dumps(resault_dict, indent=4)
    ##print(json_str)
    #print(resault_dict)
    
    #JSON
    file_name = f'cross_distance_rmsd.json'
    with open(file_name, 'w', encoding = 'utf-8') as file:
        json.dump(resault_dict, file, ensure_ascii=False, indent=4) #ensure_ascii=False





if __name__ == '__main__':
    compute_cross_distance_rmsd('posebusters_predict_sdf_boxsize10')