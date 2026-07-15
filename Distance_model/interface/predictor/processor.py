# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
import lmdb
import pickle
import copy
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from multiprocessing import Pool
from typing import List
from sklearn.cluster import KMeans
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolAlign import AlignMolConformers
from biopandas.pdb import PandasPdb
import dill




import time
from functools import wraps

class FunctionTimeoutError(Exception):
    """，"""
    #raise Exception('time out')
    pass

def measure_time(threshold):
    """，
    
    Args:
        threshold (float): （）
    
    Returns:
        function: 
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)  # 
            elapsed_time = time.time() - start_time
            
            if elapsed_time > threshold:
                raise FunctionTimeoutError(
                    f"Function '{func.__name__}' exceeded time threshold. "
                    f"Elapsed time: {elapsed_time:.2f}s, Threshold: {threshold}s"
                )
            return result
        return wrapper
    return decorator

# 
@measure_time(threshold=1.5)  # 1.5
def my_function():
    """"""
    time.sleep(2)  # 2
    return "Done"



class Processor:
    def __init__(self, 
        mode:str='single', 
        nthreads:int=20, 
        conf_size:int=10, 
        cluster:bool=False, 
        main_atoms:List[str]=["N", "CA", "C", "O", "H"], 
        allow_pocket_atoms:List[str]=[['C', 'H', 'N', 'O', 'S']],
        use_current_ligand_conf:bool=False
    ):
        self.mode = mode
        self.nthreads = nthreads
        self.conf_size = conf_size
        self.cluster = cluster
        self.main_atoms = main_atoms
        self.allow_pocket_atoms = allow_pocket_atoms
        if self.mode in ['batch_one2one', 'batch_one2many']:
            self.lmdb_name = 'batch_data'
        self.use_current_ligand_conf = use_current_ligand_conf

    def preprocess(self, input_protein:str, input_ligand, input_docking_grid:str, output_ligand_name:str, out_lmdb_dir:str):
        seed = 42 
        if self.mode=='single':
            supp = Chem.SDMolSupplier(input_ligand)
            mol = [mol for mol in supp if mol][0]
            ori_smiles = Chem.MolToSmiles(mol)
            smiles_list = [ori_smiles]
            input_protein = [input_protein]
            input_ligand = [input_ligand]
            input_docking_grid = [input_docking_grid]
        elif self.mode in ['batch_one2one', 'batch_one2many']:
            if self.mode == 'batch_one2many':
                input_protein = [input_protein] * len(input_ligand)
            smiles_list = []
            error_count = 0
            error_name_list = []
            error_index_list = []
            for i in range(len(input_ligand)):
                try:
                    supp = Chem.SDMolSupplier(input_ligand[i])
                    mol = [mol for mol in supp if mol][0]
                    ori_smiles = Chem.MolToSmiles(mol)
                    mol  = Chem.RemoveHs(mol)
                    ligand_pos = np.array(mol.GetConformer(0).GetPositions())
                    smiles_list.append(ori_smiles)
                except Exception as e:
                    try:
                        #
                        supp = Chem.SDMolSupplier(input_ligand[i], sanitize=False)
                        mol = [mol for mol in supp if mol][0]
                        ori_smiles = Chem.MolToSmiles(mol)
                        
                        #mol  = Chem.RemoveHs(mol)
                        #try:
                            #mol  = Chem.RemoveHs(mol)
                        #except Exception as e:
                            #print('not RemoveHs')
                            
                        #mol_h = Chem.AddHs(mol, addCoords=True)
                        # （“”）
                        mol = Chem.RemoveHs(mol, sanitize=False)  #，sanitize = False
                            
                        ligand_pos = np.array(mol.GetConformer(0).GetPositions())
                        smiles_list.append(ori_smiles)
                        input_ligand[i] = os.path.join(os.path.dirname(input_ligand[i]), 'origin_' + input_ligand[i].split('/')[-2] + '_ligand.sdf')
                    except Exception as e:
                        try:
                            mol = Chem.MolFromMol2File(os.path.join(os.path.dirname(input_ligand[i]), input_ligand[i].split('/')[-2] + '_ligand.mol2'), sanitize=False)
                            ori_smiles = Chem.MolToSmiles(mol)
                            mol  = Chem.RemoveHs(mol)
                            ligand_pos = np.array(mol.GetConformer(0).GetPositions())
                            smiles_list.append(ori_smiles)
                            input_ligand[i] = os.path.join(os.path.dirname(input_ligand[i]), input_ligand[i].split('/')[-2] + '_ligand.mol2')
                        except Exception as e:
                            error_count += 1
                            error_name_list.append(input_ligand[i].split('/')[-2])
                            #print(e)
                            #print('error ligand:', input_ligand[i])
                            error_index_list.append(i)


            with open('error_ligand.txt', 'a') as f:
                for name in error_name_list:
                    f.write(name + '\n')

                
            #print('lack error_count:', error_count)
            #exit()
        #smiles，
        print('error_index_list:', error_index_list)
        print('output_ligand_name:', output_ligand_name)
        for index_i in error_index_list:
            print('index_i:', index_i)
            del output_ligand_name[index_i]
            del input_protein[index_i]
            del input_ligand[index_i]
            del input_docking_grid[index_i]
            #del out_lmdb_dir[index_i]
            
        lmdb_name = self.write_lmdb(output_ligand_name, smiles_list, input_protein, input_ligand, input_docking_grid, seed=seed, result_dir=out_lmdb_dir)
        return lmdb_name

    def single_conf_gen(self, tgt_mol, num_confs=1000, seed=42, removeHs=True):
        mol = copy.deepcopy(tgt_mol)
        mol = Chem.AddHs(mol)
        allconformers = AllChem.EmbedMultipleConfs(
            mol, numConfs=num_confs, randomSeed=seed, clearConfs=True
        )
        sz = len(allconformers)
        for i in range(sz):
            try:
                AllChem.MMFFOptimizeMolecule(mol, confId=i)
            except:
                continue
        if removeHs:
            mol = Chem.RemoveHs(mol)
        return mol

    def single_conf_gen_no_MMFF(self, tgt_mol, num_confs=1000, seed=42, removeHs=True):
        mol = copy.deepcopy(tgt_mol)
        mol = Chem.AddHs(mol)
        allconformers = AllChem.EmbedMultipleConfs(
            mol, numConfs=num_confs, randomSeed=seed, clearConfs=True
        )
        if removeHs:
            mol = Chem.RemoveHs(mol)
        return mol

    @measure_time(threshold=10)
    def clustering_coords_copy(self, mol, M=1000, N=100, seed=42, cluster=False, removeHs=True, gen_mode='mmff'):
        # N，MMrdkit，10，40，40*10rdkit，，，

        try:
            rdkit_coords_list = []
            if not cluster:
                M = N
            if gen_mode == 'mmff':
                rdkit_mol = self.single_conf_gen(mol, num_confs=M, seed=seed, removeHs=removeHs)
            elif gen_mode == 'no_mmff':
                rdkit_mol = self.single_conf_gen_no_MMFF(mol, num_confs=M, seed=seed, removeHs=removeHs)
            noHsIds = [
                rdkit_mol.GetAtoms()[i].GetIdx()
                for i in range(len(rdkit_mol.GetAtoms()))
                if rdkit_mol.GetAtoms()[i].GetAtomicNum() != 1
            ]
            ### exclude hydrogens for aligning
            AlignMolConformers(rdkit_mol, atomIds=noHsIds) #，
            sz = len(rdkit_mol.GetConformers()) #
            for i in range(sz):
                _coords = rdkit_mol.GetConformers()[i].GetPositions().astype(np.float32)
                rdkit_coords_list.append(_coords)

            ### exclude hydrogens for clustering, pick closest to centroid:
            
            if cluster:
                # (num_confs, num_atoms, 3)
                rdkit_coords = np.array(rdkit_coords_list)[:, noHsIds]
                # (num_confa, num_atoms, 3) -> (num_confs, num_atoms*3)
                rdkit_coords_flatten = rdkit_coords.reshape(sz, -1)
                kmeans = KMeans(n_clusters=N, random_state=seed).fit(rdkit_coords_flatten) #N，
                # (num_clusters, num_atoms, 3)
                center_coords = kmeans.cluster_centers_.reshape(N, -1, 3)
                # (num_cluster, num_confs)
                cdist = ((center_coords[:, None] - rdkit_coords[None, :])**2).sum(axis=(-1, -2))
                # (num_confs,)
                argmin = np.argmin(cdist, axis=-1)
                coords_list = [rdkit_coords_list[i] for i in argmin]
            else:
                coords_list = rdkit_coords_list
            

            #，
            #print('len(coords_list):', len(coords_list))
            if len(coords_list) != N:
                coords_list = coords_list + [coords_list[0]] * (N - len(coords_list)) 
        except Exception as e:
            #print(e)
            cluster = False
            rdkit_coords_list = []
            if not cluster:
                M = N
            if gen_mode == 'mmff':
                rdkit_mol = self.single_conf_gen(mol, num_confs=M, seed=seed, removeHs=removeHs)
            elif gen_mode == 'no_mmff':
                rdkit_mol = self.single_conf_gen_no_MMFF(mol, num_confs=M, seed=seed, removeHs=removeHs)
            noHsIds = [
                rdkit_mol.GetAtoms()[i].GetIdx()
                for i in range(len(rdkit_mol.GetAtoms()))
                if rdkit_mol.GetAtoms()[i].GetAtomicNum() != 1
            ]
            ### exclude hydrogens for aligning
            AlignMolConformers(rdkit_mol, atomIds=noHsIds) #，
            sz = len(rdkit_mol.GetConformers()) #
            for i in range(sz):
                _coords = rdkit_mol.GetConformers()[i].GetPositions().astype(np.float32)
                rdkit_coords_list.append(_coords)

            ### exclude hydrogens for clustering, pick closest to centroid:
            
            if cluster:
                # (num_confs, num_atoms, 3)
                rdkit_coords = np.array(rdkit_coords_list)[:, noHsIds]
                # (num_confa, num_atoms, 3) -> (num_confs, num_atoms*3)
                rdkit_coords_flatten = rdkit_coords.reshape(sz, -1)
                kmeans = KMeans(n_clusters=N, random_state=seed).fit(rdkit_coords_flatten) #N，
                # (num_clusters, num_atoms, 3)
                center_coords = kmeans.cluster_centers_.reshape(N, -1, 3)
                # (num_cluster, num_confs)
                cdist = ((center_coords[:, None] - rdkit_coords[None, :])**2).sum(axis=(-1, -2))
                # (num_confs,)
                argmin = np.argmin(cdist, axis=-1)
                coords_list = [rdkit_coords_list[i] for i in argmin]
            else:
                coords_list = rdkit_coords_list
            

            #，
            #print('len(coords_list):', len(coords_list))
            if len(coords_list) != N:
                coords_list = coords_list + [coords_list[0]] * (N - len(coords_list)) 

        return coords_list



    @measure_time(threshold=100)
    def clustering_coords(self, mol, M=1000, N=100, seed=42, cluster=False, removeHs=True, gen_mode='mmff'):
        # N，MMrdkit，10，40，40*10rdkit，，，
        rdkit_coords_list = []
        if not cluster:
            M = N
        if gen_mode == 'mmff':
            rdkit_mol = self.single_conf_gen(mol, num_confs=M, seed=seed, removeHs=removeHs)
        elif gen_mode == 'no_mmff':
            rdkit_mol = self.single_conf_gen_no_MMFF(mol, num_confs=M, seed=seed, removeHs=removeHs)
        noHsIds = [
            rdkit_mol.GetAtoms()[i].GetIdx()
            for i in range(len(rdkit_mol.GetAtoms()))
            if rdkit_mol.GetAtoms()[i].GetAtomicNum() != 1
        ]
        ### exclude hydrogens for aligning
        AlignMolConformers(rdkit_mol, atomIds=noHsIds) #，
        sz = len(rdkit_mol.GetConformers()) #
        for i in range(sz):
            _coords = rdkit_mol.GetConformers()[i].GetPositions().astype(np.float32)
            rdkit_coords_list.append(_coords)

        ### exclude hydrogens for clustering, pick closest to centroid:
        
        if cluster:
            # (num_confs, num_atoms, 3)
            rdkit_coords = np.array(rdkit_coords_list)[:, noHsIds]
            # (num_confa, num_atoms, 3) -> (num_confs, num_atoms*3)
            rdkit_coords_flatten = rdkit_coords.reshape(sz, -1)
            kmeans = KMeans(n_clusters=N, random_state=seed).fit(rdkit_coords_flatten) #N，
            # (num_clusters, num_atoms, 3)
            center_coords = kmeans.cluster_centers_.reshape(N, -1, 3)
            # (num_cluster, num_confs)
            cdist = ((center_coords[:, None] - rdkit_coords[None, :])**2).sum(axis=(-1, -2))
            # (num_confs,)
            argmin = np.argmin(cdist, axis=-1)
            coords_list = [rdkit_coords_list[i] for i in argmin]
        else:
            coords_list = rdkit_coords_list
        

        #，
        ##print('len(coords_list):', len(coords_list))
        if len(coords_list) != N:
            coords_list = coords_list + [coords_list[0]] * (N - len(coords_list)) 
        

        return coords_list

    def find_residues_in_pocket(self, pocket: dict, pdf):
        """
        Given a pocket config and a residue df, 
        return a list of residues that are in the pocket
        """
        def _get_vertex(pocket: dict, axis: str) -> tuple:
            """
            Return the minimum and maximum values of the given axis

            Args:
            pocket (dict): pocket config
            axis (str): ["x", "y", "z"]

            Returns:
            A tuple of floats.
            """
            return (
                pocket["center_{}".format(axis)] \
                    - pocket["size_{}".format(axis)] / 2,
                pocket["center_{}".format(axis)] \
                    + pocket["size_{}".format(axis)] / 2
                )
        min_x, max_x = _get_vertex(pocket, "x")
        min_y, max_y = _get_vertex(pocket, "y")
        min_z, max_z = _get_vertex(pocket, "z")
        min_array = np.array([min_x, min_y, min_z]).reshape(1,3)
        max_array = np.array([max_x, max_y, max_z]).reshape(1,3)
        patoms, pcoords, residues = [], np.empty((0,3)), []
        for i in range(len(pdf)):
            atom_info = pdf.iloc[i]
            _rescoor = np.array(atom_info[['x_coord','y_coord','z_coord']].values).reshape(-1,3)
            mapping = (_rescoor > min_array) & (_rescoor < max_array)
            if (mapping.sum(-1) == 3).sum() > 0:
                patoms += [atom_info['atom_name']]
                pcoords = np.concatenate((pcoords, _rescoor), axis=0)
                residues += [str(atom_info['chain_id'])+str(atom_info['residue_number'])]
        return patoms, pcoords, residues

    def extract_pocket(self, input_protein, input_docking_grid):
        pmol = PandasPdb().read_pdb(input_protein) #
        
        #
        ##print('pmol.df:', pmol.df)
        '''
        atom_df = pmol.df['ATOM']
        hetatm_df = pmol.df['HETATM']

        #  = ATOM + HETATM 
        total_atoms = len(atom_df) + len(hetatm_df)
        #print(f"1：{total_atoms}")
        '''
        
        
        
        
        df_no_h = pmol.df['ATOM'][pmol.df['ATOM']['element_symbol'] != 'H']
        pmol.df['ATOM'] = df_no_h
        if 'HETATM' in pmol.df:
            pmol.df['HETATM'] = pmol.df['HETATM'][pmol.df['HETATM']['element_symbol'] != 'H']

        '''
        atom_df = pmol.df['ATOM']
        hetatm_df = pmol.df['HETATM']

        #  = ATOM + HETATM 
        total_atoms = len(atom_df) + len(hetatm_df)
        #print(f"2：{total_atoms}")
        
        raise Exception('test')
        '''
        
        with open(input_docking_grid, "r") as file:
            box_dict = json.load(file)

        pdf = pmol.df['ATOM']
        patoms, pcoords, residues = self.find_residues_in_pocket(box_dict, pdf) #
        def _filter_pocketatoms(atom):
            if atom[:2] in ['Cd','Cs', 'Cn', 'Ce', 'Cm', 'Cf', 'Cl', 'Ca', 'Cr', 'Co', 'Cu', 'Nh', 'Nd', 'Np', 'No', 'Ne', 'Na', 'Ni', \
                'Nb', 'Os', 'Og', 'Hf', 'Hg', 'Hs', 'Ho', 'He', 'Sr', 'Sn', 'Sb', 'Sg', 'Sm', 'Si', 'Sc', 'Se']:
                return None
            if atom[0] >= '0' and atom[0] <= '9':
                return _filter_pocketatoms(atom[1:])
            if atom[0] in ['Z','M','P','D','F','K','I','B']:
                return None
            if atom[0] in self.allow_pocket_atoms:
                return atom
            return atom

        atoms, index, residues_tmp = [], [], []
        for i,a in enumerate(patoms):
            output = _filter_pocketatoms(a)
            if output is not None:
                index.append(True)
                atoms.append(output)
                residues_tmp.append(residues[i])
            else:
                index.append(False)
        coordinates = pcoords[index].astype(np.float32)
        residues = residues_tmp
        patoms = atoms
        pcoords = [coordinates]
        side = [0 if a in self.main_atoms else 1 for a in patoms]
        return patoms, pcoords, residues, side, box_dict

    def parser(self, content):
        try:
            smiles, input_protein, input_ligand, input_docking_grid, seed = content
            name = input_protein.split('/')[-2]
            
            tg = os.path.basename(input_protein).split('_')
            if len(tg) == 2:
                complex_name = tg[0]
            elif len(tg) == 3:
                complex_name = tg[1]


            #，，40，？
            #，1403d，40？
            #，40，data_lmdb.pkl
            #
            patoms, pcoords, residues, side, config = self.extract_pocket(input_protein, input_docking_grid) #，，ai
            #print('len(patoms):', len(patoms))
            #print('pcoords num:', len(pcoords[0])) #1
            #print('pcoords[0].shape:', pcoords[0].shape)
            #print('pcoords[0][:2]:', pcoords[0][:2])
            #print('pcoords[0]_centor:', np.mean(pcoords[0], axis = 0)) #，？

            #raise Exception('test')
            #pcoords[0]_centor: [-21.913183  12.446733  26.25213 ]
            #，，/mnt/home/fanzhiguang/47/unimol_docking_v2/unimol/data/normalize_dataset.py，
            #
            #raise Exception('stop')
            #pcoords num: 1
            #pcoords[0].shape: (128, 3) #，(136, 3), ，H，(0,0,0), ，
            #0
            #pcoords[0][:2]: [[-18.67   19.158  23.089]
            #[-17.539  18.199  22.696]]

            #，xyz + 10ai， 10ai，
            #10ai = （xyz + 10ai） = （10 + 10）/ 2 = 10

            
            #print('input_protein:', input_protein) #input_protein: /mnt/home/fanzhiguang/47/CrossDocked2020/data/pdbbind2020_r10/pdbbind_connect/4u5o/4u5o_protein.pdb
            #print('complex_name:', complex_name) #complex_name: 4u5o
            #raise Exception('stop')

            # get ground truth conformation and generate ligand conformation, sdf，，，mol2
            if 'origin' in input_ligand:
                supp = Chem.SDMolSupplier(input_ligand, sanitize=False)
                mol = [Chem.RemoveHs(mol) for mol in supp if mol][0]
            elif '.mol2' in input_ligand:
                mol = Chem.MolFromMol2File(input_ligand, sanitize=False)
            else:
                supp = Chem.SDMolSupplier(input_ligand)
                mol = [Chem.RemoveHs(mol) for mol in supp if mol][0]
            
            ##print('ligand coords.shape:', mol.GetConformer().GetPositions().astype(np.float32).shape) #ligand coords.shape: (13, 3)， H，
            #raise Exception('stop')
            self.use_current_ligand_conf = False #ecdock，True
            if self.use_current_ligand_conf: #false，，。，，
                #，rdkit, 1
                return pickle.dumps(
                    {
                        "atoms": [atom.GetSymbol() for atom in mol.GetAtoms()],
                        "coordinates": [mol.GetConformer().GetPositions().astype(np.float32)],
                        "mol_list": [mol],
                        "pocket_atoms": patoms,
                        "pocket_coordinates": pcoords,
                        "side": side,
                        "residue": residues,
                        "config": config,
                        "holo_coordinates": [mol.GetConformer().GetPositions().astype(np.float32)],
                        "holo_mol": mol,
                        "holo_pocket_coordinates": pcoords,
                        "smi": smiles,
                        "pocket": input_protein,
                        "flag": 'success',
                        "name": name
                    },
                    protocol=-1,
                    ), True, input_ligand, complex_name, pcoords
            
            
            #mol = Chem.AddHs(mol) #addCoords=True
            #if mol == None:
                #raise Exception("mol is None")
            smiles = Chem.MolToSmiles(mol)
            latoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
            holo_coordinates = [mol.GetConformer().GetPositions().astype(np.float32)]
            holo_mol = mol
            N = self.conf_size # 1，？
            M = self.conf_size * 10 #，rdkit10，，，
            mol_list = [mol] * N #holo，mol_listmol，，
            #40rdkit，，，，，
            coordinate_list = []
            try:
                coordinate_list = self.clustering_coords(mol, M=M, N=N, seed=seed, cluster = True, removeHs=True, gen_mode='mmff')
            except (Exception, FunctionTimeoutError) as e: 
                try:
                    coordinate_list = self.clustering_coords(mol, M=M, N=N, seed=seed, cluster = False, removeHs=True, gen_mode='mmff')
                except (Exception, FunctionTimeoutError) as e: 
                    try:
                        coordinate_list = self.clustering_coords(mol, M=1, N=1, seed=seed, cluster = False, removeHs=True, gen_mode='no_mmff')
                        coordinate_list = coordinate_list * N
                    except (Exception, FunctionTimeoutError) as e:
                            coordinate_list = holo_coordinates * N
                
            #：/data/fan_zg/MDocking/Docking_baseline/unimol_docking_v2/unimol/tasks/docking_pose_v2.py
            
            '''
            try:
                coordinate_list = self.clustering_coords(mol, M=M, N=N, seed=seed, cluster = True, removeHs=True, gen_mode='mmff')
            except (Exception, FunctionTimeoutError) as e:
                try:
                    coordinate_list = self.clustering_coords(mol, M=M, N=N, seed=seed, cluster = False, removeHs=True, gen_mode='no_mmff')
                except (Exception, FunctionTimeoutError) as e:
                    try:
                        coordinate_list = self.clustering_coords(mol, M=1, N=1, seed=seed, cluster = False, removeHs=True, gen_mode='no_mmff')
                        coordinate_list = coordinate_list * N
                    except (Exception, FunctionTimeoutError) as e:
                        #print(e)
                        coordinate_list = holo_coordinates * N
            '''    

            assert len(coordinate_list) == N
            return pickle.dumps(
                {
                    "atoms": latoms,
                    "coordinates": coordinate_list, #rdkit
                    "mol_list": mol_list,
                    "pocket_atoms": patoms,
                    "pocket_coordinates": pcoords,
                    "side": side,
                    "residue": residues,
                    "config": config,
                    "holo_coordinates": holo_coordinates, #ground truth
                    "holo_mol": holo_mol, #ground truth
                    "holo_pocket_coordinates": pcoords, #ground truth
                    "smi": smiles,
                    "pocket": input_protein,
                    "flag": 'success',
                    "name": name
                },
                protocol=-1,
                ), True, input_ligand, complex_name, pcoords
        except (Exception, FunctionTimeoutError) as e:
            #print(e)
            pcoords = 0
            return None, False,  input_ligand, complex_name, pcoords


                



    def write_lmdb(self, output_ligand_name, smiles_list, input_protein, input_ligand, input_docking_grid, seed=42, result_dir="./results"):
        #print('result_dir:', result_dir)
        os.makedirs(result_dir, exist_ok=True)
        if self.mode == 'single':
            outputfilename = os.path.join(result_dir, output_ligand_name + ".lmdb")
        elif self.mode in ['batch_one2one', 'batch_one2many']:
            outputfilename = os.path.join(result_dir, self.lmdb_name + ".lmdb")
            output_ligand_name = self.lmdb_name
        try:
            os.remove(outputfilename)
        except:
            pass
        env_new = lmdb.open(
            outputfilename,
            subdir=False,
            readonly=False,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1,
            map_size=int(100*(1024*1024*1024)), # 100GB
        )
        txn_write = env_new.begin(write=True)
        #print("Start preprocessing data...")
        #print(f'Number of ligands: {len(smiles_list)}')
        fail_file_list = []
        seed = [seed] * len(input_ligand)
        content_list = zip(smiles_list, input_protein, input_ligand, input_docking_grid, seed)

        #unimol
        
        #
        '''
        pcoords_dict = {}
        with Pool(self.nthreads) as pool:
            ii = 0
            failed_num = 0
            for inner_output, flag, file_ligand, complex_name, pcoords in tqdm(pool.imap(self.parser, content_list)): #content_list
                #print('flag:', flag)
                if flag is True:
                    txn_write.put(f"{ii}".encode("ascii"), inner_output)
                    ii+=1
                    pcoords_dict[complex_name] = pcoords
                elif flag is False: 
                    #
                    fail_file_list.append(file_ligand)
                    failed_num += 1
                    continue
                    #txn_write.put(f"{i}".encode("ascii"), inner_output) #，
                    #i+=1 #
                    #failed_num += 1
            txn_write.commit()
            env_new.close()
        '''
        
        #
        pcoords_dict = {}
        with Pool(100) as pool:
            ii = 0
            failed_num = 0
            for inner_output, flag, file_ligand, complex_name, pcoords in tqdm(pool.imap(self.parser, content_list),total=len(smiles_list)): #content_list
                ##print('flag:', flag)
                if flag is True:
                    txn_write.put(f"{ii}".encode("ascii"), inner_output)
                    ii+=1
                    pcoords_dict[complex_name] = pcoords
                elif flag is False: 
                    #
                    fail_file_list.append(file_ligand)
                    failed_num += 1
                    continue
                    #txn_write.put(f"{i}".encode("ascii"), inner_output) #，
                    #i+=1 #
                    #failed_num += 1
            txn_write.commit()
            env_new.close()
        
        
        #
        '''
        pcoords_dict = {}
        ii = 0
        failed_num = 0
        for smiles_i, input_protein_i, input_ligand_i, input_docking_grid_i, seed_i in tqdm(content_list, total=len(smiles_list)): #content_list
            #print('ii:', ii)
            try:
                inner_output, flag, file_ligand, complex_name, pcoords = self.parser((smiles_i, input_protein_i, input_ligand_i, input_docking_grid_i, seed_i))
                #print('flag:', flag)
            
                txn_write.put(f"{ii}".encode("ascii"), inner_output)
                ii+=1
                pcoords_dict[complex_name] = pcoords
            except Exception as e: 
                #print('data deal fail:', e)
                #
                fail_file_list.append(input_ligand_i)
                failed_num += 1
                continue
                #txn_write.put(f"{i}".encode("ascii"), inner_output) #，
                #i+=1 #
                #failed_num += 1
            
        txn_write.commit()
        env_new.close()
        '''

        return output_ligand_name #

    def load_lmdb_data(self, lmdb_path, key):
        env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
        )
        txn = env.begin()
        _keys = list(txn.cursor().iternext(values=False))
        collects = []
        for idx in range(len(_keys)):
            datapoint_pickled = txn.get(f"{idx}".encode("ascii"))
            data = pickle.loads(datapoint_pickled)
            collects.append(data[key])
        return collects

    def postprocess_data_pre_copy(self, predict_file, lmdb_file):
        old_mol_list = self.load_lmdb_data(lmdb_file, "mol_list")
        #mol_list = [Chem.RemoveHs(mol) for items in mol_list for mol in items]
        fail_index = []
        mol_list = []
        num = 0
        #print('mol_list1:', old_mol_list) #
        for items in old_mol_list:
            for mol in items:
                try:
                    Chem.RemoveHs(mol)
                    mol_list.append(mol)
                except Exception as e:
                    #print(f"Failed to remove Hs from mol: {mol}, Error: {e}")
                    fail_index.append(num)
                    mol_list.append(None)
                
                num += 1

        #print('mol_list2:', mol_list) #

        #print('predict_file:', predict_file)
        predict = pd.read_pickle(predict_file) #，，，，, pkl，None
        ##print('predict:', predict)
        #print('predict.keys():', predict[0].keys())
        '''
        predict.keys(): dict_keys(['loss', 'cross_distance_loss', 'distance_loss', 'coord_loss', 'prmsd_loss', 'prmsd_score', 'bsz', 'sample_size', 
        'coord_predict', 'coord_target', 'smi_name', 'pocket_name', 'atoms', 'pocket_atoms', 'coordinates', 'holo_coordinates', 'pocket_coordinates', 
        'holo_center_coordinates'])
        '''
        smi_list, pocket_list, coords_predict_list, holo_coords_list, holo_center_coords_list, prmsd_score_list = [],[],[],[],[],[]

        #，，，
        cross_distance_list = []
        pocket_coords_list = []
        holo_pocket_coords_list = []

        #print('predict num:', len(predict)) #4

        for batch in predict:

            if batch == None:
                print('batch == None, skip') #，，，，，1
            else:
                #print("batch['atoms']:", batch['atoms']) #
                #print("batch['atoms']:", len(batch['atoms'])) #1
                #print("batch['atoms'].shape:", batch['atoms'].shape) #1
                sz = batch['atoms'].size(0) #1
                #print('batch num:', len(predict))
                #print('sz:', sz)
                #raise Exception("Invalid batch") #

                '''
                batch['atoms']: 40
                batch['atoms'].shape: torch.Size([40, 16])
                batch num: 2
                sz: 40
                '''
                
                for i in range(sz):
                    try:
                        smi_list.append(batch['smi_name'][i])
                        pocket_list.append(batch['pocket_name'][i])
                        prmsd_score_list.append(batch['prmsd_score'][i].numpy().astype(np.float32))

                        cross_distance_list.append(batch['cross_distance'][i].numpy().astype(np.float32))
                        pocket_coords_list.append(batch['pocket_coordinates'][i].numpy().astype(np.float32))
                        holo_pocket_coords_list.append(batch['holo_pocket_coordinates'][i].numpy().astype(np.float32))
                        ##print('holo_pocket_coords_list:', holo_pocket_coords_list)
                        #raise Exception('stop')
                        
                        token_mask = batch['atoms'][i]>2 #3，3？？，，3

                        holo_coordinates = batch['holo_coordinates'][i]
                        holo_coordinates = holo_coordinates[token_mask,:]
                        holo_coordinates = holo_coordinates.numpy().astype(np.float32)

                        coord_predict = batch['coord_predict'][i]
                        coord_predict = coord_predict[token_mask,:]
                        coord_predict = coord_predict.numpy().astype(np.float32)

                        holo_center_coordinates = batch["holo_center_coordinates"][i][:3]
                        holo_center_coordinates.numpy().astype(np.float32)

                        holo_center_coords_list.append(holo_center_coordinates)        
                        coords_predict_list.append(coord_predict)
                        holo_coords_list.append(holo_coordinates)
                    except Exception as e:
                        #exit()
                        #print('error,skip:', e)
                        smi_list.append(None) 
                        pocket_list.append(None)
                        coords_predict_list.append(None)
                        holo_coords_list.append(None)
                        holo_center_coords_list.append(None) 
                        prmsd_score_list.append(None)


        return mol_list, smi_list, coords_predict_list, holo_coords_list, holo_center_coords_list, prmsd_score_list, fail_index, pocket_coords_list, cross_distance_list, holo_pocket_coords_list




    def postprocess_data_pre(self, predict_file, lmdb_file):
        #，0，，

        old_mol_list = self.load_lmdb_data(lmdb_file, "mol_list")
        #mol_list = [Chem.RemoveHs(mol) for items in mol_list for mol in items]
        fail_index = []
        mol_list = []
        num = 0
        ##print('mol_list1:', old_mol_list) #
        for items in old_mol_list:
            for mol in items:
                try:
                    new_mol = Chem.RemoveHs(mol)
                    mol_list.append(new_mol)
                except Exception as e:
                    #print(f"Failed to remove Hs from mol: {mol}, Error: {e}")
                    fail_index.append(num)
                    mol_list.append(None)
                
                num += 1

        ##print('mol_list2:', mol_list) #

        #print('predict_file:', predict_file)
        predict = pd.read_pickle(predict_file) #，，，，, pkl，None
        ##print('predict:', predict)

        #print(f'assert {len(old_mol_list)} == {len(predict)}') #2934 == 4797
        assert len(old_mol_list) == len(predict)

        #print('predict.keys():', predict[0].keys())
        '''
        predict.keys(): dict_keys(['loss', 'cross_distance_loss', 'distance_loss', 'coord_loss', 'prmsd_loss', 'prmsd_score', 'bsz', 'sample_size', 
        'coord_predict', 'coord_target', 'smi_name', 'pocket_name', 'atoms', 'pocket_atoms', 'coordinates', 'holo_coordinates', 'pocket_coordinates', 
        'holo_center_coordinates'])
        '''
        smi_list, pocket_list, coords_predict_list, holo_coords_list, holo_center_coords_list, prmsd_score_list = [],[],[],[],[],[]

        #，，，
        cross_distance_list = []
        pocket_coords_list = []
        holo_pocket_coords_list = []
        
        ligand_emb = []
        pocket_emb = []

        #print('predict num:', len(predict)) #4

        for batch in predict:

            if batch == None:
                #print('batch == None, skip') #，，，，，1
                raise Exception('error, None')
            else:
                #print("batch['atoms']:", batch['atoms']) #
                #print("batch['atoms']:", len(batch['atoms'])) #1
                #print("batch['atoms'].shape:", batch['atoms'].shape) #1
                sz = batch['atoms'].size(0) #1
                #print('batch num:', len(predict))
                #print('sz:', sz)
                #raise Exception("Invalid batch") #

                '''
                batch['atoms']: 40
                batch['atoms'].shape: torch.Size([40, 16])
                batch num: 2
                sz: 40
                '''
                #
                for i in range(sz):
                    try:
                        smi_list.append(batch['smi_name'][i])
                        pocket_list.append(batch['pocket_name'][i])
                        prmsd_score_list.append(batch['prmsd_score'][i].numpy().astype(np.float32))

                        cross_distance_list.append(batch['cross_distance'][i].numpy().astype(np.float32))
                        pocket_coords_list.append(batch['pocket_coordinates'][i].numpy().astype(np.float32))
                        holo_pocket_coords_list.append(batch['holo_pocket_coordinates'][i].numpy().astype(np.float32))
                        #ligand_emb.append(batch['ligand_emb'][i].numpy())
                        #pocket_emb.append(batch['pocket_emb'][i].numpy())
                        ##print('holo_pocket_coords_list:', holo_pocket_coords_list)
                        #raise Exception('stop')
                        
                        #，0，，
                        #，0
                        #token_mask = batch['atoms'][i]>2 #3，3？？，，3

                        holo_coordinates = batch['holo_coordinates'][i]
                        #holo_coordinates = holo_coordinates[token_mask,:]
                        holo_coordinates = holo_coordinates.numpy().astype(np.float32)

                        coord_predict = batch['coord_predict'][i]
                        #coord_predict = coord_predict[token_mask,:]
                        coord_predict = coord_predict.numpy().astype(np.float32)

                        holo_center_coordinates = batch["holo_center_coordinates"][i][:3]
                        holo_center_coordinates.numpy().astype(np.float32)

                        holo_center_coords_list.append(holo_center_coordinates)        
                        coords_predict_list.append(coord_predict)
                        holo_coords_list.append(holo_coordinates)
                    except Exception as e:
                        #print('error,skip:', e)
                        raise Exception('error, skip')
                        smi_list.append(None) 
                        pocket_list.append(None)
                        coords_predict_list.append(None)
                        holo_coords_list.append(None)
                        holo_center_coords_list.append(None) 
                        prmsd_score_list.append(None)

        #，pocket_coords_listholo_pocket_coords_list
        for i in range(len(pocket_coords_list)):
            assert np.allclose(np.array(pocket_coords_list[i]), np.array(holo_pocket_coords_list[i]), rtol=0.00, atol=0.00)

        '''
        np.set_#printoptions(precision=4, suppress=True) 
        #print('len(pocket_coords_list):', len(pocket_coords_list)) #list
        #print('len(holo_pocket_coords_list):', len(holo_pocket_coords_list)) #list
        
        #print(np.array(pocket_coords_list[0]).shape, np.array(pocket_coords_list[2]).shape)
        #print(np.array(holo_pocket_coords_list[0]).shape, np.array(holo_pocket_coords_list[2]).shape)

        A = np.array(pocket_coords_list[0])
        sorted_indices1 = np.lexsort((A[:, 2], A[:, 1], A[:, 0]))
        # 
        sorted_A = A[sorted_indices1]

        B = np.array(pocket_coords_list[2])
        sorted_indices2 = np.lexsort((B[:, 2], B[:, 1], B[:, 0]))
        # 
        sorted_B = B[sorted_indices2]

        # ，
        coords1_tuples = {tuple(row) for row in A}
        coords2_tuples = {tuple(row) for row in B}

        # 
        intersection = np.array(list(coords1_tuples & coords2_tuples))

        #print("：")
        #print(intersection) #[] ，？40，，，，

        
        #print('pocket_coords_list[0]\n:', sorted_A[:5])
        #print('pocket_coords_list[2]\n:', sorted_B[:5])

        #，pocket_coords_listholo_pocket_coords_list
        for i in range(len(pocket_coords_list)):
            assert np.allclose(np.array(pocket_coords_list[i]), np.array(holo_pocket_coords_list[i]), rtol=0.00, atol=0.00)



        #print(np.sum(np.array(pocket_coords_list[0])), np.sum(np.array(pocket_coords_list[2]))) #，，summean0,
        assert np.allclose(sorted_A, sorted_B, rtol=0.01, atol=0.02)

        #print(np.sum(np.array(holo_pocket_coords_list[0])), np.sum(np.array(holo_pocket_coords_list[2])))
        assert np.allclose(np.array(holo_pocket_coords_list[0]), np.array(holo_pocket_coords_list[2]), rtol=0.01, atol=0.02)

        exit()
        '''
        return mol_list, smi_list, coords_predict_list, holo_coords_list, holo_center_coords_list, prmsd_score_list, fail_index, pocket_coords_list, cross_distance_list, holo_pocket_coords_list, ligand_emb, pocket_emb




    def set_coord(self, mol, coords):
        for i in range(coords.shape[0]):
            mol.GetConformer(0).SetAtomPosition(i, coords[i].tolist())
        return mol

    def add_coord(self, mol, xyz):
        x, y, z = xyz
        conf = mol.GetConformer(0)
        pos = conf.GetPositions()
        pos[:, 0] += x
        pos[:, 1] += y
        pos[:, 2] += z
        for i in range(pos.shape[0]):
            conf.SetAtomPosition(
                i, Chem.rdGeometry.Point3D(pos[i][0], pos[i][1], pos[i][2])
            )
        return mol
    

    def subtract_coord(self, mol, xyz):
        x, y, z = xyz
        conf = mol.GetConformer(0)
        pos = conf.GetPositions()
        pos[:, 0] -= x
        pos[:, 1] -= y
        pos[:, 2] -= z
        for i in range(pos.shape[0]):
            conf.SetAtomPosition(
                i, Chem.rdGeometry.Point3D(pos[i][0], pos[i][1], pos[i][2])
            )
        return mol
    
    def get_sdf(self, mol_list, smi_list, coords_predict_list, holo_center_coords_list, prmsd_score_list, output_ligand_name, output_ligand_dir, \
                output_ligand_dir2, holo_coords_list, pocket_coords_list, holo_pocket_coords_list, cross_distance_list, ligand_emb_list, pocket_emb_list, tta_times=10):
        #print("Start converting model predictions into sdf files...")
        output_ligand_list = []
        if self.mode == 'single':
            output_ligand_name = [output_ligand_name]
        #print('tta_times:', tta_times) #40
        #tta_times = 1
        ##print('mol_list:', mol_list)
        ##print('smi_list:', smi_list)
        #print('output_ligand_name:', output_ligand_name) #l
        #print('output_ligand_dir2:', output_ligand_dir2) #['predict_sdf_boxsize20/6erv']
        #outputfilename = os.path.join(output_ligand_dir, str(output_ligand_name[i]) + '.sdf')
        #try:
            #os.remove(outputfilename)
        #except:
            #pass
        new_holo_coords_lists    = []
        new_coords_predict_lists = []
        new_pocket_coords_lists  = []
        new_cross_distance_lists = []
        new_ligand_emb_lists     = []
        new_pocket_emb_lists     = []

        for i in tqdm(range(len(smi_list)//tta_times)): #，
            #print('===============================================================')
            #print('i:', i)
            coords_predict_tta = coords_predict_list[i*tta_times:(i+1)*tta_times]
            prmsd_score_tta = prmsd_score_list[i*tta_times:(i+1)*tta_times]
            mol_list_tta = mol_list[i*tta_times:(i+1)*tta_times]
            holo_center_coords_tta = holo_center_coords_list[i*tta_times:(i+1)*tta_times]

            holo_coords_tta   = holo_coords_list[i*tta_times:(i+1)*tta_times]
            pocket_coords_tta = pocket_coords_list[i*tta_times:(i+1)*tta_times]
            holo_pocket_coords_tta = holo_pocket_coords_list[i*tta_times:(i+1)*tta_times]
            cross_distance_tta = cross_distance_list[i*tta_times:(i+1)*tta_times]
            
            #ligand_emb_tta = ligand_emb_list[i*tta_times:(i+1)*tta_times]
            #pocket_emb_tta = pocket_emb_list[i*tta_times:(i+1)*tta_times]

            #
            #idx = np.argmin(prmsd_score_tta) #rmsd
            #bst_predict_coords = coords_predict_tta[idx]
            #mol = mol_list_tta[idx]
            #print('mol_list_tta:', mol_list_tta)
            new_mol_list = []
            new_org_mol_list = []

            new_holo_coords_list    = []
            new_coords_predict_list = []
            new_pocket_coords_list  = []
            new_cross_distance_list = []
            new_ligand_emb_list     = []
            new_pocket_emb_list     = []

            for org_mol, mol, coords, centor, holo_coords, pocket_coords, holo_pocket_coords, cross_distance in zip(copy.deepcopy(mol_list_tta), \
                copy.deepcopy(mol_list_tta), coords_predict_tta, holo_center_coords_tta, copy.deepcopy(holo_coords_tta), copy.deepcopy(pocket_coords_tta), \
                    copy.deepcopy(holo_pocket_coords_tta), copy.deepcopy(cross_distance_tta)):
                #holo_coords, pocket_coords，holo_center_coords？，mol_list
                #？？，0，(0,0,0), ，0？
                #pcoords[0].shape: (128, 3) #，(136, 3), ，H，(0,0,0), ，
                #0
                #H，
                orgin_pos = org_mol.GetConformer(0).GetPositions().astype(np.float32)
                holo_center_coords = centor

                holo_coords_c, pocket_coords_c, holo_pocket_coords_c = np.mean(holo_coords , axis = 0), np.mean(pocket_coords, axis = 0), np.mean(holo_pocket_coords, axis = 0)
                
                #，，，
                #print('，，，')
                #print('')
                #print('holo_coords_c1:', holo_coords_c)
                #print('pocket_coords_c1:', pocket_coords_c) 
                #print('holo_pocket_coords_c1:', holo_pocket_coords_c)

                #print('holo_coords.shape1:', holo_coords.shape) #(0,0,0),holo_center_coords，holo_coords.shape1: (13, 3), 0
                #print('orgin_pos.shape1:', orgin_pos.shape) #(0,0,0), orgin_pos.shape1: (22, 3), ， #(22, 3)
                #print('predict_coords1:', coords.shape) #(13, 3)， ：orgin_poscoords，coordsmol？，
                #print('pocket_coords.shape, holo_pocket_coords.shape1:', pocket_coords.shape, holo_pocket_coords.shape) #(136, 3) (136, 3)
                #print('cross_distance.shape:', cross_distance.shape) #cross_distance.shape: (16, 136), ，16，13，22

                #print('holo_coords[:2]1:', holo_coords[:2]) #(0,0,0),holo_center_coords，，0，
                #print('orgin_pos[:2]1:', orgin_pos[:2]) # ，

                #print('pocket_coords[:2]1:', pocket_coords[:2])
                #print('holo_pocket_coords[:2]1:', holo_pocket_coords[:2])


                #print('\n')

                #print('--------------------------------------------------------')
                #print('，，，')
                #print('')
                #0，，
                holo_coords, pocket_coords, holo_pocket_coords = holo_coords + np.array(holo_center_coords), pocket_coords + np.array(holo_center_coords), holo_pocket_coords + np.array(holo_center_coords)
                #holo_coords = holo_coords + np.array(holo_center_coords)

                #print('type(holo_center_coords):', type(holo_center_coords)) #<class 'torch.Tensor'>
                #print('holo_center_coords:', holo_center_coords)

                holo_coords_c, pocket_coords_c, holo_pocket_coords_c = np.mean(holo_coords , axis = 0), np.mean(pocket_coords, axis = 0), np.mean(holo_pocket_coords, axis = 0)
                #print('holo_coords_c2:', holo_coords_c)
                #print('pocket_coords_c2:', pocket_coords_c) #0,0,0，holo_center_coords？
                #print('holo_pocket_coords_c2:', holo_pocket_coords_c) #

                
                orgin_pos = org_mol.GetConformer(0).GetPositions().astype(np.float32)
                #？。 , ，sdf4，
                #print('holo_coords.shape2:', holo_coords.shape) #(0,0,0),holo_center_coords，
                #print('orgin_pos.shape2:', orgin_pos.shape)

                #print('holo_coords[:]2:', holo_coords[:2]) #(0,0,0),holo_center_coords，
                #print('orgin_pos[:]2:', orgin_pos[:2])

                #？
                #print('pocket_coords[:2]2:', pocket_coords[:2])
                #print('holo_pocket_coords[:2]2:', holo_pocket_coords[:2])

                #print('\n')
                #print('*****************************************************')
                #

                #print('，，，')
                #print('mol，sdf')
                #print('holo_center_coords:', holo_center_coords)
                #print('holo_coords_c:', holo_coords_c)
                
                new_org_mol_list.append(org_mol)
                #：orgin_poscoords，coordsmol？，，，
                #mol。，
                new_mol = self.set_coord(copy.deepcopy(mol), coords)
                #print('new_mol1:', new_mol)
                #print('holo_center_coords.numpy():', holo_center_coords.numpy())
                try:
                    new_mol = self.add_coord(new_mol, holo_center_coords.numpy()) 
                    #print('new_mol2:', new_mol)
                except Exception as e:
                    print('e:', e)
                    with open('get_sdf_error.txt', 'a') as f:
                        input_ligand = os.path.join(output_ligand_dir2[i], 'org_' + str(output_ligand_name[i]) + '.sdf')
                        #print('input_ligand num:', len(input_ligand))
                        f.write(input_ligand + '\n')
                        continue

                new_pos = new_mol.GetConformer(0).GetPositions()
                new_centor = np.mean(new_pos, axis = 0)
                #print('new_mol centor:', new_centor)


                #
                #print('coords，，，')
                #print('coords_centor:', np.mean(coords, axis = 0))
                #print('coords')
                new_coords = coords + np.array(holo_center_coords)
                #print('coords_centor:', np.mean(new_coords, axis = 0))


                #raise Exception('test')

                
                new_mol_list.append(new_mol)

                #print('tyep(holo_coords):', type(holo_coords)) #<class 'numpy.ndarray'>
                #print('type(coords):', type(new_coords))#<class 'numpy.ndarray'>
                #print('type(holo_pocket_coords):', type(holo_pocket_coords))#<class 'numpy.ndarray'>
                #print('type(cross_distance):', type(cross_distance))#<class 'numpy.ndarray'>

                #print('len(holo_coords):', holo_coords.shape)
                #print('len(coords):', new_coords.shape)
                #print('len(holo_pocket_coords):', holo_pocket_coords.shape)
                #print('len(cross_distance):', cross_distance.shape)

                new_holo_coords_list.append(holo_coords)
                new_coords_predict_list.append(new_coords)  #coords？
                new_pocket_coords_list.append(holo_pocket_coords)
                new_cross_distance_list.append(cross_distance)
                #new_ligand_emb_list.append(ligand_emb)
                #new_pocket_emb_list.append(pocket_emb)


                #raise Exception('test')
            
            new_holo_coords_lists.append(new_holo_coords_list)
            new_coords_predict_lists.append(new_coords_predict_list)
            new_pocket_coords_lists.append(new_pocket_coords_list)
            new_cross_distance_lists.append(new_cross_distance_list)
            #new_ligand_emb_lists.append(new_ligand_emb_list)
            #new_pocket_emb_lists.append(new_pocket_emb_list)

            data_dict = {}
            data_dict['holo_coords_list'] = new_holo_coords_list
            data_dict['coords_predict_list'] = new_coords_predict_list
            data_dict['pocket_coords_list'] = new_pocket_coords_list
            data_dict['cross_distance_list'] = new_cross_distance_list
            #data_dict['ligand_emb_list'] = new_ligand_emb_list
            #data_dict['pocket_emb_list'] = new_pocket_emb_list


            if len(new_holo_coords_list) > 2:
                #？
                for j in range(len(new_holo_coords_list))[1:]:
                    assert np.allclose(np.array(new_holo_coords_list[j-1]), np.array(new_holo_coords_list[j]), rtol=0.01, atol=0.02)


                np.set_printoptions(precision=4, suppress=True) 
                #print('len(pocket_coords_list):', len(new_pocket_coords_list)) #list
                
                #print(np.array(new_pocket_coords_list[0]).shape, np.array(new_pocket_coords_list[2]).shape)

                A = np.array(new_pocket_coords_list[0])
                sorted_indices1 = np.lexsort((A[:, 2], A[:, 1], A[:, 0]))
                # 
                sorted_A = A[sorted_indices1]

                B = np.array(new_pocket_coords_list[2])
                sorted_indices2 = np.lexsort((B[:, 2], B[:, 1], B[:, 0]))
                # 
                sorted_B = B[sorted_indices2]

                #print('pocket_coords_list[0]\n:', sorted_A[:5])
                #print('pocket_coords_list[2]\n:', sorted_B[:5])

                


                # ，
                coords1_tuples = {tuple(row) for row in A}
                coords2_tuples = {tuple(row) for row in B}

                # 
                intersection = np.array(list(coords1_tuples & coords2_tuples))
                #print('A.shape:', A.shape)
                #print('B.shape:', B.shape)
                #print(" num：", len(intersection)) #171, 40，？，？
            

            '''
            assert np.allclose(sorted_A, sorted_B, rtol=0.01, atol=0.02)
            #. ，，40，？
            #print(np.sum(np.array(new_pocket_coords_list[0])), np.sum(np.array(new_pocket_coords_list[-1]))) #-2764.335 -2859.1602
            assert np.allclose(np.sum(np.array(new_pocket_coords_list[0])), np.sum(np.array(new_pocket_coords_list[-1])), rtol=0.01, atol=0.02)
            assert np.allclose(np.array(new_pocket_coords_list[0]), np.array(new_pocket_coords_list[-1]), rtol=0.01, atol=0.02)
            '''
            
            #print('output_ligand_dir2[i]:', output_ligand_dir2[i]) #list index out of range
            os.makedirs(output_ligand_dir2[i], exist_ok=True)

            #
            outputfilename = os.path.join(output_ligand_dir2[i], 'interaction_' + str(output_ligand_name[i]) + '.pkl')
            with open(outputfilename, "wb") as f:
                dill.dump(data_dict, f)

            outputfilename = os.path.join(output_ligand_dir2[i], 'org_' + str(output_ligand_name[i]) + '.sdf')
            #sdf
            w = Chem.SDWriter(outputfilename)
            #Chem.MolToMolFile(mol, outputfilename)
            ##print('new_mol_list:', len(new_mol_list))
            for mol in new_org_mol_list[:1]:
                # 
                new_mol = Chem.RemoveHs(mol)
                #  SDF 
                w.write(new_mol)
            w.close()



            outputfilename = os.path.join(output_ligand_dir2[i], 'gen_' + str(output_ligand_name[i]) + '.sdf')
            try:
                os.remove(outputfilename)
            except:
                pass

            #sdf
            w = Chem.SDWriter(outputfilename)
            #Chem.MolToMolFile(mol, outputfilename)
            ##print('new_mol_list:', len(new_mol_list))
            for mol in new_mol_list:
                # 
                new_mol = Chem.RemoveHs(mol)
                #  SDF 
                w.write(new_mol)
            w.close()

            output_ligand_list.append(outputfilename)

        

        #print("Done!")
        ##print('output_ligand_list:', output_ligand_list)
        if self.mode == 'single':
            return output_ligand_list[0]
        elif self.mode in ['batch_one2one', 'batch_one2many']:
            return output_ligand_list
    
    def single_clash_fix(self, input_content):
        input_ligand, output_ligand, label_ligand, pocket_mol = input_content
        script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "ecmol", "scripts", "6tsr.py")
        cmd = "python {} --input-ligand {} --output-ligand {} --label-ligand {} --pocket-mol {} --num-6t-trials 5".format(
            script_path, input_ligand, output_ligand, label_ligand, pocket_mol
        )
        os.system(cmd)
        return True

    def clash_fix(self, predicted_ligand, input_protein, input_ligand):
        if self.mode=='batch_one2many':
            input_protein = [input_protein] * len(input_ligand)
        elif self.mode == 'single':
            input_ligand = [input_ligand]
            input_protein = [input_protein]
            predicted_ligand = [predicted_ligand]
        input_content = zip(predicted_ligand, predicted_ligand, input_ligand, input_protein)

        with Pool(self.nthreads) as pool:
            for inner_output in tqdm(
                pool.imap(self.single_clash_fix, input_content), total=len(input_ligand) if type(input_ligand) is list else 1
            ):
                if not inner_output:
                    print("fail to clash fix")
        return predicted_ligand

    @classmethod
    def build_processors(
        cls, 
        mode='single', 
        nthreads = 4, 
        conf_size = 1, 
        cluster=False,
        use_current_ligand_conf:bool=False
    ):
        return cls(
            mode, 
            nthreads, 
            conf_size=conf_size, 
            cluster=cluster, 
            use_current_ligand_conf=use_current_ligand_conf
        )