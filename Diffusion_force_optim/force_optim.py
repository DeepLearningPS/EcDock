import math
from typing import Any, Callable, Iterable, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from tqdm.auto import tqdm



from typing import Iterator
from torch import Tensor, nn

import random
import numpy as np
from torch_scatter import scatter_sum, scatter_mean
import torch.nn.functional as F
import copy
import random

import networkx as nx
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Rotation as R
from rdkit.Chem import AllChem, rdMolTransforms
from rdkit import Geometry
from rdkit.Chem import AllChem, GetPeriodicTable, RemoveHs
#from models.molopt_score_model import center_pos, index_to_log_onehot, q_v_sample

from rdkit.Geometry.rdGeometry import Point3D
import time
import os
import copy
from rdkit import Chem
from rdkit.Chem import AllChem
import copy
import subprocess
import time
import multiprocessing

from collections import defaultdict
from ordered_set import OrderedSet
from biopandas.pdb import PandasPdb
from Bio import PDB

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import defaultdict

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm
import math
import dill
import shutil

import glob
from multiprocessing import Pool
from functools import partial

import gzip
import shutil

from torch.utils.data import Dataset, DataLoader

from comparm import GP

from TreeInvent2.model.consistency.consistency_models import opt_coords_moves,opt_complex_coords_moves
import copy

import os
import argparse
from pathlib import Path


np.random.seed(2023)
torch.manual_seed(2023)
random.seed(2023)
torch.cuda.manual_seed_all(2023)


def Change_Mol_D3coord(inputmol,coords):
    '''
        ，inputmol，coords：
    '''
    molobj=copy.deepcopy(inputmol)
    conformer=molobj.GetConformer()
    id=conformer.GetId()
    for cid,xyz in enumerate(coords):
        ##print(xyz[0],xyz[1],xyz[2],type(xyz))
        conformer.SetAtomPosition(cid,Point3D(float(xyz[0]),float(xyz[1]),float(xyz[2]))) #
    conf_id=molobj.AddConformer(conformer)
    molobj.RemoveConformer(id)
    #molobj = Chem.RemoveHs(molobj)
    return molobj








def karmadock_mmff(pocket_mol_list, ligand_mol_list, n_iters=200, ff_type='mmff'):
    mmf_mol_list = []
    for pocket_mol, ligand_mol in zip(pocket_mol_list, ligand_mol_list):
        try:
            # form complex
            ligand_mol = AllChem.AddHs(ligand_mol, addCoords=True)
            complex_mol = Chem.CombineMols(pocket_mol, ligand_mol)
            try:
                Chem.SanitizeMol(complex_mol)
            except Chem.AtomValenceException:
                print('Invalid valence')
            except (Chem.AtomKekulizeException, Chem.KekulizeException):
                print('Failed to kekulize')
            try:
                if ff_type == 'mmff':
                    ff = AllChem.MMFFGetMoleculeForceField(complex_mol, AllChem.MMFFGetMoleculeProperties(complex_mol), confId=0, ignoreInterfragInteractions=False)
                else:
                    ff = AllChem.UFFGetMoleculeForceField(complex_mol, confId=0, ignoreInterfragInteractions=False)
                ff.Initialize()
                # fix pocket points
                [ff.AddFixedPoint(i) for i in range(pocket_mol.GetNumAtoms())]
                # minimize
                ff.Minimize(maxIts=n_iters)
                print(f"Performed {ff_type} with binding site...")
            except:
                print(f'Skip {ff_type}_{n_iters} ...')
            coords = complex_mol.GetConformer().GetPositions()
            rd_conf = ligand_mol.GetConformer()
            [rd_conf.SetAtomPosition(i, xyz) for i, xyz in enumerate(coords[-ligand_mol.GetNumAtoms():])] 
        except Exception as e:
            print('mmf fail in:', e)
            continue
        
        mmf_mol_list.append(ligand_mol)
        
    return mmf_mol_list
    




def ecdock_mmff_adaptive(pocket_mol_list, ligand_mol_list, n_iters=200, ff_type='mmff', energy_tol=1e-4, stagnation_window=5):
    '''
    energy_tol=1e-4,      # (kcal/mol)
    stagnation_window=5   # 
    '''
    mmf_mol_list = []
    for pocket_mol, ligand_mol in zip(pocket_mol_list, ligand_mol_list):
        try:
            # form complex
            ligand_mol = AllChem.AddHs(ligand_mol, addCoords=True)
            complex_mol = Chem.CombineMols(pocket_mol, ligand_mol)
            try:
                Chem.SanitizeMol(complex_mol)
            except Chem.AtomValenceException:
                print('Invalid valence')
            except (Chem.AtomKekulizeException, Chem.KekulizeException):
                print('Failed to kekulize')
            try:
                if ff_type == 'mmff':
                    ff = AllChem.MMFFGetMoleculeForceField(complex_mol, AllChem.MMFFGetMoleculeProperties(complex_mol), confId=0, ignoreInterfragInteractions=False)
                else:
                    ff = AllChem.UFFGetMoleculeForceField(complex_mol, confId=0, ignoreInterfragInteractions=False)
                ff.Initialize()
                # fix pocket points
                [ff.AddFixedPoint(i) for i in range(pocket_mol.GetNumAtoms())]
                # minimize
                #ff.Minimize(maxIts=n_iters)
                
                
                
                
                # 
                energy_history = []
                stagnation_count = 0
                converged = False
                for step in range(n_iters):
                    # 
                    ff.Minimize(maxIts=1)
                    
                    # 
                    current_energy = ff.CalcEnergy()
                    energy_history.append(current_energy)
                    
                    # 
                    if len(energy_history) > 1:
                        delta = abs(energy_history[-1] - energy_history[-2])
                        
                        if delta < energy_tol:
                            stagnation_count += 1
                            # 
                            if stagnation_count >= stagnation_window:
                                converged = True
                                break
                        else:
                            stagnation_count = 0  # 
                
                
                
                
                print(f"Performed {ff_type} with binding site...")
            except:
                print(f'Skip {ff_type}_{n_iters} ...')
                
            coords = complex_mol.GetConformer().GetPositions()
            rd_conf = ligand_mol.GetConformer()
            [rd_conf.SetAtomPosition(i, xyz) for i, xyz in enumerate(coords[-ligand_mol.GetNumAtoms():])] 
        except Exception as e:
            print('mmf fail in:', e)
            continue
        
        mmf_mol_list.append(ligand_mol)
        
    return mmf_mol_list



def adaptive_forcefield_optimization(
    complex_mol, 
    pocket_mol,
    max_iters=500,
    energy_tol=1e-4,      # (kcal/mol)
    stagnation_window=5  # 
):
    # 
    ff = AllChem.MMFFGetMoleculeForceField(
        complex_mol,
        AllChem.MMFFGetMoleculeProperties(complex_mol),
        confId=0,
        ignoreInterfragInteractions=False
    )
    ff.Initialize()
    
    # （）
    for i in range(pocket_mol.GetNumAtoms()):
        ff.AddFixedPoint(i)
    
    # 
    energy_history = []
    stagnation_count = 0
    converged = False
    
    # 
    for step in range(max_iters):
        # 
        ff.Minimize(maxIts=1)
        
        # 
        current_energy = ff.CalcEnergy()
        energy_history.append(current_energy)
        
        # 
        if len(energy_history) > 1:
            delta = abs(energy_history[-1] - energy_history[-2])
            
            if delta < energy_tol:
                stagnation_count += 1
                # 
                if stagnation_count >= stagnation_window:
                    converged = True
                    break
            else:
                stagnation_count = 0  # 
                
    # 
    return {
        "optimized_mol": complex_mol,
        "total_iters": step + 1,
        "final_energy": current_energy,
        "converged": converged
    }

#  --------------------------------------------------
# 
# complex_mol = Chem.MolFromPDBFile("complex.pdb")
# pocket_mol = Chem.MolFromPDBFile("pocket.pdb")

# 
# result = adaptive_forcefield_optimization(complex_mol, pocket_mol)
# print(f": {result['total_iters']}, : {result['final_energy']:.2f} kcal/mol")




def MMFF(ligand_pos, protein_pos, complex_pos, graph_id_batch_vector, mask_vector, complex_mol, cross_distance, loop = 10):
    #：
    '''
    x, #
    rdkit_mols, #ridkt mol,, 
    gmasks.bool(), #
    loop=guide_loops, #，1，
    show_state=show_state,
    min_type=min_type,  #，LBFGS
    fix_masks=fix_mask, #，
    pocket_masks=pocket_labels.bool(), #
    ligand_masks=ligand_labels.bool()  #
    '''
    with torch.enable_grad():
        #_,  cross_loss = self.Distance_Opt(ligand_pos.clone().detach(), protein_pos.clone().detach(), cross_distance[0].clone().detach(), min_type = GP.min_type)
        if GP.guide_type=='synchronous': #，x，，，
            if GP.opt_types=="complex":
                pass
                #x_moves,energy_min=opt_complex_coords_moves(x_bp,rdkit_mols,gmasks.bool(),loop=1,show_state=Ture,min_type=LBFGS,fix_masks=fix_mask,pocket_masks=pocket_labels.bool(),ligand_masks=ligand_labels.bool())
            else:
                pass
                #x_moves,energy_min=opt_coords_moves(x_bp,rdkit_mols,gmasks.bool(),loop=guide_loops,show_state=show_state,min_type=min_type,fix_masks=fix_mask)
        
        #，，。+complex, asynchronous ,
        else:
            try:
                if GP.opt_types=="complex":
                    x_moves,energy_min=opt_complex_coords_moves(complex_pos.clone().detach(), copy.deepcopy(complex_mol), graph_id_batch_vector, loop=loop,\
                        show_state=True, min_type=GP.min_type, mask_ligand = mask_vector, cross_loss = None, \
                        ligand_pos = ligand_pos.clone().detach(), protein_pos = protein_pos.clone().detach(), cross_distance = copy.deepcopy(cross_distance))
                    
                    '''
                    x_moves,energy_min=opt_complex_coords_moves(complex_pos.clone().detach(), copy.deepcopy(complex_mol), preds['batch_all'].clone().detach(), loop=GP.loop,\
                        show_state=True, min_type=GP.min_type, mask_ligand = preds['mask_ligand'].clone().detach(), cross_loss = None, \
                        ligand_pos = ligand_pos.clone().detach(), protein_pos = protein_pos.clone().detach(), cross_distance = copy.deepcopy(cross_distance))
                    '''
                else:
                    x_moves,energy_min=opt_coords_moves(complex_pos.clone().detach(), copy.deepcopy(complex_mol), graph_id_batch_vector, loop=1, \
                        show_state=True, min_type=GP.min_type, mask_ligand = mask_vector, cross_loss = None, \
                            ligand_pos = ligand_pos.clone().detach(), protein_pos = protein_pos.clone().detach(), cross_distance = copy.deepcopy(cross_distance))
            except Exception as e:
                print('error:', e)
                x_moves = 0

    print('ligand_pos:', ligand_pos.shape)
    #print('x_moves:', x_moves.shape)
    ligand_pos = ligand_pos+x_moves #x_moves，，+# * 0.1, 5step
    
    return ligand_pos


def _read_pdb_ONRing_biopandas(pdb_file):
    #pdbON
    #  PDB 
    pdb = PandasPdb().read_pdb(pdb_file)

    #  ATOM 
    atom_df = pdb.df['ATOM']

    # 
    # ，
    atom_df['coords'] = atom_df[['x_coord', 'y_coord', 'z_coord']].apply(tuple, axis=1)

    # ，
    unique_atom_df = atom_df.drop_duplicates(subset='coords')

    # 
    unique_atom_df = unique_atom_df.drop(columns=['coords'])

    # 
    unique_atom_df = unique_atom_df.reset_index(drop=True)

    #  'line_idx' ， 0 
    unique_atom_df['line_idx'] = range(len(unique_atom_df))

    # 
    #print(unique_atom_df.head())


    #  PDB 
    #pdb.df['ATOM'] = unique_atom_df
    #pdb.to_pdb(path='output_protein.pdb', records=None)

    atom_df = unique_atom_df


    # 
    oxygen_atoms = []
    nitrogen_atoms = []
    ring_atoms = []
    xyzs = []
    atom_index = []
    atom_isspecial = []

    # 
    for i, row in atom_df.iterrows():
        atom_name    = row['atom_name'].strip()
        element_symb = row['element_symbol'].strip()
        residue_name = row['residue_name'].strip()

        #
        atom_index.append(row['line_idx'])
        
        #
        x, y, z = row['x_coord'], row['y_coord'], row['z_coord']
        xyzs.append([x, y, z])

        # 
        oxygen_atoms.append(element_symb == 'O')
        o_flag = element_symb == 'O'

        # 
        nitrogen_atoms.append(element_symb == 'N')
        n_flag = element_symb == 'N'


        # 
        # 
        ring_atom_dict = {
            'PHE': ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
            'TYR': ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
            'TRP': ['CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
            'HIS': ['CG', 'ND1', 'CD2', 'CE1', 'NE2'],
            'PRO': ['CG', 'CD'],
            # 
        }

        # 
        if residue_name in ring_atom_dict and atom_name in ring_atom_dict[residue_name]:
            ring_atoms.append(True)
            r_flag = True
        else:
            ring_atoms.append(False)
            r_flag = False

        atom_isspecial.append(o_flag or n_flag or r_flag)

    # 
    assert len(oxygen_atoms) == len(nitrogen_atoms) and len(ring_atoms) == len(atom_isspecial) and len(oxygen_atoms) == len(atom_index)
    print(f'{max(atom_index)} <= {len(atom_index) - 1}')
    assert max(atom_index) <= len(atom_index) - 1

    if not np.array(xyzs).shape == np.unique(np.array(xyzs), axis = 0).shape: #，，
        raise Exception(f'{np.array(xyzs).shape} == {np.unique(np.array(xyzs), axis = 0).shape}')


    return np.array(oxygen_atoms), np.array(nitrogen_atoms), np.array(ring_atoms), np.array(atom_isspecial), np.array(atom_index), np.array(xyzs) 


class BuiltDataset(Dataset):
    """
    Training Dataset class.

    Parameters
    ----------
    triples:	The triples used for training the model
    params:		Parameters for the experiments

    Returns
    -------
    
    A training Dataset class instance used by DataLoader
    """

    def __init__(self, name_list, base_path):
        self.name_list = name_list
        self.base_path = base_path

    def __len__(self):
        #sdf，1
        return len(self.name_list)

    def __getitem__(self, idx):
        
        '''
        ligand_pos_matrix
        protein_pos_matrix
        complex_pos_matrix
        complex_mol_list
        graph_id_batch_vector
        mask_vector
        cross_distance_list
        '''                                       
                                    
        name    = self.name_list[idx]
        tg_path = os.path.join(base_path, name)
        
        #ligand_file         = os.path.join(tg_path, f'gen_ligand_{name}.sdf')
        ligand_file         = os.path.join(tg_path, f'gen_{name}_ligand.sdf')
        origin_ligand_file  = os.path.join(tg_path, f'origin_{name}_ligand.sdf')
        pocket_protein_file = os.path.join(tg_path, f'{name}_protein_400.pdb')
        protein_file        = os.path.join(tg_path, f'{name}_protein.pdb')
        complex_file        = os.path.join(tg_path, f'{name}_protein_400_complex.pdb')
        cross_distance_file = os.path.join(tg_path, f'interaction_{name}_v2.pkl')
        
        
        #
        print('ligand_file:', ligand_file)
        l_mol_list      = list(Chem.SDMolSupplier(ligand_file, removeHs=False, sanitize=True)) #mmfsanitize=True
        l_mol_list      = [AllChem.AddHs(ml, addCoords=True) for ml in l_mol_list]
        ligand_pos_list = []
        for l_mol in l_mol_list:
            l_pos = np.array(l_mol.GetConformers()[0].GetPositions())
            ligand_pos_list.append(l_pos)
        ligand_pos_matrix = np.array(ligand_pos_list)
        
        
        #
        #atom_isO, atom_isN, atom_isring, atom_isspecial, atom_id, p_pos = _read_pdb_ONRing_biopandas(pocket_protein_file)
        if os.path.exists(pocket_protein_file):
            p_mol = Chem.MolFromPDBFile(pocket_protein_file,removeHs=False,sanitize=False)
        else:
            p_mol = Chem.MolFromPDBFile(protein_file,removeHs=False,sanitize=False)
        p_pos = np.array(p_mol.GetConformers()[0].GetPositions())
        protein_pos_list = [p_pos] * len(l_mol_list)
        
        #
        #complex_mol            = Chem.MolFromPDBFile(complex_file, removeHs=False, sanitize=True) #，
        complex_mol_list        = []
        p_mol_list              = [p_mol] * len(l_mol_list)
        

        for l_mol in l_mol_list:
            complex_mol = Chem.CombineMols(p_mol, l_mol)
            complex_mol_list.append(complex_mol)

        #
        complex_pos_list    = []
        mask_list           = []
        graph_id_batch_list = [] #id
        for ids, (l_pos, p_pos) in enumerate(zip(ligand_pos_list, protein_pos_list)):
            c_pos = np.concatenate((p_pos, l_pos), axis=0)
            complex_pos_list.append(c_pos)
            
            l_mask = np.ones([l_pos.shape[0]])  #1
            p_mask = np.zeros([p_pos.shape[0]]) #0
            #print('l_mask, p_mask:', l_mask.shape, p_mask.shape) #l_mask, p_mask: (43,) (15109,)
            mask   = np.concatenate((p_mask, l_mask), axis=0)
            #print('mask.shape:', mask.shape) #(15152,)
            mask_list.append(mask) 
            
            graph_id_batch_list.append(np.array([ids] * c_pos.shape[0], dtype = np.int64))
        
        
        
        complex_pos_matrix      = np.concatenate(complex_pos_list, axis=0)
        mask_vector             = np.concatenate(mask_list, axis=0)
        graph_id_batch_vector   = np.concatenate(graph_id_batch_list, axis=0)
        protein_pos_matrix      = np.concatenate(protein_pos_list, axis=0)
        
        #print('complex_pos_matrix:', complex_pos_matrix.shape)
        #print('mask_vector:', mask_vector.shape)
        #print('graph_id_batch_vector:', graph_id_batch_vector.shape)
        
        
        '''
        with open(cross_distance_file, 'rb') as file:
            interaction_data = dill.load(file)
            
        holo_coords_list = interaction_data['holo_coords_list']
        coords_predict_list = interaction_data['coords_predict_list']
        pocket_coords_list = interaction_data['pocket_coords_list']
        cross_dt_list = interaction_data['cross_distance_list']
        
        
        #rmsd，，
        rmsd_list = []
        for pre_pos, holo_pos in zip(coords_predict_list, holo_coords_list):
            assert pre_pos.shape == holo_pos.shape   #"Coordinate matrices must have the same shape"
            rmsd = np.sqrt(np.mean(np.sum((pre_pos - holo_pos) ** 2, axis=1)))
            rmsd_list.append(rmsd)
        
        sorted_indices = np.argsort(rmsd_list)
        best_index = sorted_indices[0]
        
        cross_distance_list = [torch.FloatTensor(cross_dt_list[best_index])] * len(ligand_mols)
        ''' 
        cross_distance_list = [[]] * len(l_mol_list)
    

        return torch.FloatTensor(ligand_pos_matrix), torch.FloatTensor(protein_pos_matrix), torch.FloatTensor(complex_pos_matrix), \
            torch.LongTensor(graph_id_batch_vector), torch.LongTensor(mask_vector), complex_mol_list, cross_distance_list, ligand_file, len(l_mol_list), name, p_mol_list, l_mol_list

    @staticmethod
    def collate_fn(batch):
        #
        ligand_pos_matrix       = torch.stack([_[0] for _ in batch], dim = 0)
        protein_pos_matrix      = torch.stack([_[1] for _ in batch], dim = 0)
        complex_pos_matrix      = torch.stack([_[2] for _ in batch], dim = 0)
        graph_id_batch_vector   = torch.stack([_[3] for _ in batch], dim = 0)
        mask_vector             = torch.stack([_[4] for _ in batch], dim = 0)
        
        complex_mol_list        = [_[5] for _ in batch] 
        cross_distance_list     = [_[6] for _ in batch] 
        ligand_file             = [_[7] for _ in batch]
        conf_num                = [_[8] for _ in batch]
        name                    = [_[9] for _ in batch]
        
        p_mol_list              = [_[10] for _ in batch]
        l_mol_list              = [_[11] for _ in batch]
        
        return ligand_pos_matrix, protein_pos_matrix, complex_pos_matrix, graph_id_batch_vector, mask_vector, complex_mol_list, cross_distance_list, ligand_file, conf_num, name, p_mol_list, l_mol_list

    def get_neg_ent(self, triple, label):
        def get(triple, label):
            pos_obj = label
            mask = np.ones([self.p.num_ent], dtype=np.bool)
            mask[label] = 0
            neg_ent = np.int32(
                np.random.choice(self.entities[mask], self.p.neg_num - len(label), replace=False)).reshape([-1])
            neg_ent = np.concatenate((pos_obj.reshape([-1]), neg_ent))

            return neg_ent

        neg_ent = get(triple, label)
        return neg_ent

    def get_label(self, label):
        y = np.zeros([self.p.num_ent], dtype=np.float32)
        for e2 in label: y[e2] = 1.0
        return torch.FloatTensor(y)





def read_batch(batch):
    """
    Function to read a batch of data and move the tensors in batch to CPU/GPU

    Parameters
    ----------
    batch: 		the batch to process
    split: (string) If split == 'train', 'valid' or 'test' split


    Returns
    -------
    Head, Relation, Tails, labels
    """

    ligand_pos_matrix, protein_pos_matrix, complex_pos_matrix, graph_id_batch_vector, mask_vector, complex_mol_list, cross_distance_list, ligand_file, conf_num, name, p_mol_list, l_mol_list = [_[0] for _ in batch]
    return ligand_pos_matrix.view(-1, 3), protein_pos_matrix.view(-1, 3), complex_pos_matrix.view(-1, 3), graph_id_batch_vector, mask_vector, complex_mol_list, cross_distance_list, ligand_file, conf_num, name, p_mol_list, l_mol_list



def data_move(spath = None, tpath = None):
    '''#'''
    
    data_name_list = []
    with open('/data/fan_zg/MDocking/new_VSDS/data_name.txt') as f:
        for n in f:
            data_name_list.append(n.strip())
    
    for data_name_ in tqdm(data_name_list):
        data_name2 = copy.deepcopy(data_name_)
        data_name = f'{data_name_}_ecdock_cm_equiformer_step5_interaction_limit4.5ai_3Dmultidistance'
        name_list = os.listdir(f'{spath}/{data_name}')
        
        
        
        '''
        with open(f'/data/fan_zg/MDocking/new_VSDS/glide_refine_fail_name/{data_name}_data_name.txt') as f:
            for n in f:
                name_list.append(n.strip())
        '''
                
        
        for nm in name_list:
            s_path = f'{spath}/{data_name}/{nm}/step4'
            
            if not os.path.isdir(s_path):
                continue
            
            t_path = f'{tpath}/{data_name2}/{nm}'
            os.makedirs(t_path, exist_ok=True)
            
            
            s_file = f'{s_path}/{nm}_protein.pdb'
            t_file = f'{t_path}/{nm}_protein.pdb'
            #shutil.copy(s_file, t_file)
            #print('s_file:', s_file)
            #print('t_file:', t_file)
            
            
            s_file = f'{s_path}/gen_ligand_{nm}.sdf'
            t_file = f'{t_path}/gen_{nm}_ligand.sdf'
            shutil.copy(s_file, t_file)
            continue
            
            s_path = f'/data/fan_zg/MDocking/VSDS_DTEBV-D/data/{data_name2}/{data_name2}/{nm}'
            
            
            s_file = f'{s_path}/interaction_{nm}_v2.pkl'
            t_file = f'{t_path}/interaction_{nm}_v2.pkl'
            shutil.copy(s_file, t_file)
            
            s_file = f'{s_path}/{nm}_ligand_docking_grid_boxsize10.json'
            t_file = f'{t_path}/{nm}_ligand_docking_grid_boxsize10.json'
            shutil.copy(s_file, t_file)
            
            
            #s_file = f'{s_path}/{nm}_protein_400_complex.pdb'
            #t_file = f'{t_path}/{nm}_protein_400_complex.pdb'
            #shutil.copy(s_file, t_file)
            
            
            s_file = f'{s_path}/{nm}_protein_400.pdb'
            t_file = f'{t_path}/{nm}_protein_400.pdb'
            shutil.copy(s_file, t_file)
            
            #s_file = f'{s_path}/{nm}_ligand.sdf'
            #t_file = f'{t_path}/origin_{nm}_ligand.sdf'
            shutil.copy(s_file, t_file)
    

def run(batch):
    ligand_pos_matrix, protein_pos_matrix, complex_pos_matrix, graph_id_batch_vector, mask_vector, complex_mol_list, cross_distance_list, ligand_file, conf_num, name, p_mol_list, l_mol_list = read_batch(batch)
    if GP.mmf_method_mode == 'xu':
        loop = GP.loop
        mmf_ligand_pos = MMFF(ligand_pos_matrix, protein_pos_matrix, complex_pos_matrix, graph_id_batch_vector, mask_vector, complex_mol_list, cross_distance_list, loop)

        
        #ref_mol         = list(Chem.SDMolSupplier(ligand_file, removeHs=False, sanitize=False))[0]
        ref_mol         = l_mol_list[0] #
        
        #mmfsdf
        #mmf_ligand_file = os.path.join(os.path.dirname(ligand_file), 'mmff_' + os.path.basename(ligand_file))
        mmf_ligand_file = ligand_file
        supp=Chem.SDWriter(mmf_ligand_file)
        for mmf_pos in mmf_ligand_pos.reshape([conf_num, -1, 3]).detach().numpy():
            try:
                new_mol = Change_Mol_D3coord(ref_mol, copy.deepcopy(mmf_pos))
                #mol2 = Chem.RemoveHs(new_mol)
                supp.write(new_mol)
            except Exception as e:
                print(e)
            
        supp.close()    #
    
    elif GP.mmf_method_mode == 'ecdock':
        print('GP.rdkit_force_mode:', GP.rdkit_force_mode)
        #mmf_mol_list = karmadock_mmff(p_mol_list, l_mol_list, n_iters=10, ff_type='mmff')
        mmf_mol_list = ecdock_mmff_adaptive(p_mol_list, l_mol_list, n_iters=100, ff_type=GP.rdkit_force_mode, energy_tol=1e-4, stagnation_window=5)
        
        #mmfsdf
        #mmf_ligand_file = os.path.join(os.path.dirname(ligand_file), 'mmff_' + os.path.basename(ligand_file))
        mmf_ligand_file = ligand_file
        supp=Chem.SDWriter(mmf_ligand_file)
        for mol in mmf_mol_list:
            try:
                supp.write(mol)
            except Exception as e:
                print(e)
        supp.close()    #


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='tmpresault')
    args = parser.parse_args()

    base            = '/data/fan_zg/MDocking/EcDock_Evaluate/evaluate_model'
    #base_path_list  = ['ecdock_mmf_step1', 'ecdock_mmf_step2', 'ecdock_mmf_step3', 'ecdock_mmf_step4', 'ecdock_mmf_step10', 'ecdock_mmf_step15', 'ecdock_mmf_step5_conf1000']
    base_path_list = [args.input_dir][:]
    for base_name in base_path_list:                  
        name_list = []
        base_path = base_name
        name_list = os.listdir(base_path)     
        name_list = list(sorted(name_list))
        name_list = [name for name in name_list if os.path.isdir(os.path.join(base_path, name))]
        #print('len(name_list):', len(name_list))
        dataset     = BuiltDataset(name_list[:], base_path)
        data_loader = DataLoader(dataset, 1, shuffle=False, num_workers=20, collate_fn=dataset.collate_fn) #sdf，1
        data_iter   = iter(data_loader)

        batch_list = [batch for batch in data_iter]
        #print('GP.mmf_method_mode:', GP.mmf_method_mode)

        parameters = batch_list
        #  Pool ，num，.100*40=4000，，50
        with Pool(processes=50) as pool:
            pool.map(run, parameters)  #, batch，
            #pool.starmap(run, parameters) #，
                

