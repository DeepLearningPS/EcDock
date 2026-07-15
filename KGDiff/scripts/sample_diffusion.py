import argparse
import os
import shutil
import time
import sys
sys.path.append(os.path.abspath('./'))\



# EcDock
import numpy as np 
from rdkit import Chem
from EcDock.graphs import * 
from EcDock.utils import *
from EcDock.model import *  #，torchDataLoaderpygDataLoader，，pyg
from EcDock.comparm import *
#from EcDock.model import ConsistencySamplingAndEditing



import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.transforms import Compose
from torch_scatter import scatter_sum, scatter_mean
from tqdm.auto import tqdm

#from torch_geometric.data import DataLoader #
from torch_geometric.loader import DataLoader #torchDataLoader, ，, ，list

import KGDiff.utils.misc as misc
import KGDiff.utils.transforms as trans
from KGDiff.datasets import get_dataset
from KGDiff.datasets.pl_data import FOLLOW_BATCH
from models.molopt_score_model import ScorePosNet3D, log_sample_categorical
from KGDiff.utils.evaluation import atom_num
from KGDiff.utils.transforms import MAP_INDEX_TO_ATOM_TYPE_ONLY, MAP_INDEX_TO_ATOM_TYPE_AROMATIC, MAP_INDEX_TO_ATOM_TYPE_FULL


import copy
from rdkit import Chem
from rdkit.Chem import AllChem
import copy
from tqdm import tqdm
from rdkit.Geometry.rdGeometry import Point3D
from collections import Counter
import matplotlib.pyplot as plt
import time

from KGDiff.scripts.evaluate import read_file, rmsds, boxplot


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

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import torch.multiprocessing as mp

import torch.distributed as dist

import warnings
warnings.filterwarnings("ignore")

local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"]) #, GPU，CPU

collate_exclude_keys = ['ligand_nbh_list']

def setup_seed_for_local_rank(base_seed=42):
    """
     GPU（local_rank），。
    ，，，，，
    """
    # local_rank  torchrun 
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))


    #  rank 
    seed = base_seed + local_rank
    #print(f"[Rank {local_rank}] ：{seed}")

    # ， rank 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 
    os.environ["PYTHONHASHSEED"] = str(seed)

    return seed, local_rank, world_size

def rebulit_rdkitmol(file):
    #  PandasPdb  PDB 
    ppdb = PandasPdb().read_pdb(file)
    
    # 
    atom_df = ppdb.df['ATOM']
    
    # 
    selected_atoms = atom_df[['x_coord', 'y_coord', 'z_coord']]
    print('selected_atoms:', selected_atoms)

    #  PandasPdb  3D RDKit Mol 
    mol = Chem.RWMol()
    atom_idx_map = {}

    for _, atom_row in selected_atoms.iterrows():
        print('atom_row:', atom_row)
        atomic_num = Chem.GetPeriodicTable().GetAtomicNumber(atom_row['element_symbol'])
        atom = Chem.Atom(atomic_num)
        idx = mol.AddAtom(atom)
        atom_idx_map[atom_row['serial_number']] = idx
        
    conf = Chem.Conformer(mol.GetNumAtoms())
    for _, atom_row in selected_atoms.iterrows():
        idx = atom_idx_map[atom_row['serial_number']]
        coord = Point3D(atom_row['x_coord'], atom_row['y_coord'], atom_row['z_coord'])
        conf.SetAtomPosition(idx, coord)
    
    mol.AddConformer(conf)
    mol = mol.GetMol()

    # ，
    if mol is None:
        raise ValueError("RDKit  3D ，。")

    return mol



def biopandas_read_proetein(file):
    #O,N, ，rdkitpdb，，rdkit，
    #。，，rdkit，
    #biopands，rdkit，

    #  PDB 
    pdb = PandasPdb().read_pdb(file)
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

    #  PDB 
    #pdb.df['ATOM'] = unique_atom_df
    #pdb.to_pdb(path='output_protein.pdb', records=None)

    atom_df = unique_atom_df

    #  'ATOM'  x, y, z 
    coordinates_df = atom_df[['x_coord', 'y_coord', 'z_coord']]
    #  NumPy 
    coordinates_matrix = coordinates_df.to_numpy()  
    #print('atom_df:', atom_df)

    return coordinates_matrix






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
    molobj = Chem.RemoveHs(molobj)
    return molobj

def unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms):
    all_step_v = [[] for _ in range(n_data)]
    for v in ligand_v_traj:  # step_i
        v_array = v.cpu().numpy()
        for k in range(n_data):
            all_step_v[k].append(v_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
    all_step_v = [np.stack(step_v) for step_v in all_step_v]  # num_samples * [num_steps, num_atoms_i]
    return all_step_v


def sample_diffusion_ligand_single_gpu(model, data, num_samples, batch_size=1, device='cuda',
                            num_steps=None, center_pos_mode='protein',
                            sample_num_atoms='ref',guide_mode='joint',
                            value_model=None,
                            protein_atom_feature_dim = None,
                            ligand_atom_feature_dim  = None, 
                            protein_element = None,
                            ligand_element  = None,
                            ckpt = None,
                            data_list = None,
                            consistency_sampling_and_editing = None,
        type_grad_weight=1.,pos_grad_weight=1., args = None, config = None):
    
    all_pred_pos, all_pred_v, all_pred_exp = [], [], []
    all_coords_predict = []
    all_pred_pos_traj, all_pred_v_traj, all_pred_exp_traj, all_pred_exp_atom_traj = [], [], [], []
    all_pred_pos_traj_dict = []
    all_pred_v0_traj, all_pred_vt_traj = [], []
    time_list = []
    num_batch = int(np.ceil(num_samples / batch_size))
    current_i = 0
    
    #unmol40，，，4040，
    target_protein_cross_distance = data.protein_cross_distance #40

    for i in tqdm(range(num_batch)):
        sub_data_list = data_list[i * batch_size : (i + 1) * batch_size]
        #print('data:', data)
        # 
        #for attr, value in data._store._mapping.__dict__.items():
        #for attr, value in data._store._mapping.items():
            #print(f": {attr}")


        #print('data:', data)
        #print('ligand_bond_index:', data.ligand_bond_index.shape)
        #print('ligand_bond_type:', data.ligand_bond_type.shape)

        #print('protein_protein_link_e:', data.protein_link_e.shape)
        #print('protein_protein_link_t:', data.protein_link_t.shape)

        #print('protein_cross_bond_index_reverse:', data.protein_cross_bond_index_reverse.shape)
        #print('protein_cross_bond_type_reverse:', data.protein_cross_bond_type_reverse.shape)

        #print('protein_cross_bond_index:', data.protein_cross_bond_index[0][0].dtype)
        #print('protein_cross_bond_type:', data.protein_cross_bond_type[0].dtype)

        #print('protein_cross_bond_index_reverse:', data.protein_cross_bond_index_reverse)
        #print('protein_cross_bond_type_reverse:', data.protein_cross_bond_type_reverse)

        #raise Exception('test')
    
        #print('protein_cross_ligand:', data.clone().protein_cross_ligand)
        n_data = batch_size if i < num_batch - 1 else num_samples - batch_size * (num_batch - 1) #，
        #print('n_data:', n_data)
        #print('[data.clone() for _ in range(n_data)]:', [data.clone() for _ in range(n_data)])
        #batch = Batch.from_data_list([sub_data_list[0].clone() for _ in range(n_data)], follow_batch=FOLLOW_BATCH, exclude_keys = collate_exclude_keys).to(device)
        batch = Batch.from_data_list(sub_data_list[:n_data], follow_batch=FOLLOW_BATCH, exclude_keys = collate_exclude_keys).to(device)
        
        
        #print('batch:', batch)
        '''
        atch: ProteinLigandDataBatch(protein_element=[228], protein_element_batch=[228], protein_element_ptr=[2], protein_molecule_name=[1], protein_pos=[228, 3], protein_is_backbone=[228], 
        protein_atom_name=[1], protein_atom_to_aa_type=[228], protein_atom_isring=[228], protein_atom_isO=[228], protein_atom_isN=[228], protein_cross_lig_isring_flag=[38], protein_cross_lig_isO_flag=[38], 
        protein_cross_lig_isN_flag=[38], protein_cross_pro_isring_flag=[228], protein_cross_pro_isO_flag=[228], protein_cross_pro_isN_flag=[228], protein_cross_ligand=[38, 3], protein_cross_protein=[228, 3],
        protein_cross_distance=[1], protein_link_e=[351, 2], protein_link_t=[351], protein_link_t_batch=[351], protein_link_t_ptr=[2], protein_link_e_reverse=[351, 2], protein_link_t_reverse=[351], 
        protein_link_t_reverse_batch=[351], protein_link_t_reverse_ptr=[2], protein_coords_predict=[38, 3], ligand_smiles=[1], ligand_element=[38], ligand_element_batch=[38], ligand_element_ptr=[2], 
        ligand_pos=[38, 3], ligand_mol=[1], ligand_rd_mol=[1], ligand_rd_pos=[38, 3], ligand_bond_index=[2, 82], ligand_bond_type=[82], ligand_bond_type_batch=[82], ligand_bond_type_ptr=[2], 
        ligand_center_of_mass=[3], ligand_atom_feature=[38, 8], ligand_hybridization=[1], ligand_fill_coords=[250, 3], ligand_fill_zmats=[250, 4], ligand_fill_masks=[250], ligand_fill_atom_order=[38], 
        ligand_atom_isring=[38], ligand_atom_isO=[38], ligand_atom_isN=[38], complex_mol=[1], protein_filename=[1], ligand_filename=[1], affinity=[1], name=[1], id=[1], protein_atom_feature=[228, 27], 
        ligand_atom_feature_full=[38], ligand_bond_feature=[82, 5])

        protein_element
        protein_element_batch
        protein_link_t_batch
        protein_link_t_reverse_batch

        ligand_element
        ligand_element_batch
        ligand_bond_type

        '''
        #exit()

        #print('ligand_bond_index:', batch.ligand_bond_index.shape)
        #print('ligand_bond_type:', batch.ligand_bond_type.shape) #bond_index（pyg）list，link_e，？，，

        #print('protein_protein_link_e:', batch.protein_link_e.shape)
        #print('protein_protein_link_t:', batch.protein_link_t.shape)

        #print('data:', data)
        #raise Exception('test')
        b_protein_cross_distance = []
        #b_cross_bond_index, b_cross_bond_type, b_cross_bond_index_reverse, b_cross_bond_type_reverse = [], [], [], []
        if isinstance(batch.protein_cross_distance, list):
            #print('：', len(batch.protein_cross_distance)) #5
            #print('1：', batch.protein_cross_distance[0].shape) #： 5
            #assert np.allclose(batch.protein_cross_distance[0], batch.protein_cross_distance[1], atol=0.02)
            #exit()

            for i in batch.protein_cross_distance: #protein_cross_distanceslist
                tg = torch.from_numpy(i).cuda()
                #print('tg.shape:', tg.shape)
                #print('tg[:3]:', tg[:3])
                b_protein_cross_distance.append(torch.from_numpy(i).cuda())

                #ii = torch.stack(list(i), dim = 0) #list，，
                #print('set ii.shape:', ii.shape)
                #print('set ii[:3]:', ii[:3])
                #b_protein_cross_distance.append(ii.cuda())

        else:
            print('：', batch.protein_cross_distance.shape)
            #exit()
            b_protein_cross_distance.append(batch.protein_cross_distance.cuda())
            #b_cross_bond_index.append(batch.cross_bond_index.cuda())
            #b_cross_bond_type.append(batch.cross_bond_type.cuda())
            #b_cross_bond_index_reverse.append(batch.cross_bond_index_reverse.cuda()) 
            #b_cross_bond_type_reverse.append(batch.cross_bond_type_reverse.cuda())
        
        #b_link_e = batch.protein_link_e.cuda()
        #b_link_e_reverse = batch.protein_link_e_reverse.cuda()
        
        
        #numpy 
        try:
            b_link_e = batch.protein_link_e.cuda(local_rank, non_blocking=True)
            b_link_e_reverse = batch.protein_link_e_reverse.cuda(local_rank, non_blocking=True)
        except Exception as e:
            b_link_e = []
            b_link_e_reverse = []

            if isinstance(batch.protein_link_e, list):
                for i in batch.protein_link_e:
                    b_link_e.append(torch.from_numpy(i).cuda())
            
            if isinstance(batch.protein_link_e_reverse, list):
                for i in batch.protein_link_e_reverse:
                    b_link_e_reverse.append(torch.from_numpy(i).cuda())
            
            b_link_e = torch.cat(b_link_e, dim = 0).cuda(local_rank, non_blocking=True)
            b_link_e_reverse = torch.cat(b_link_e_reverse, dim = 0).cuda(local_rank, non_blocking=True)
        
            



        t1 = time.time()
        with torch.no_grad():
            model.eval()
            #。，。mod == ref,
            batch_protein = batch.protein_element_batch
            if sample_num_atoms == 'prior':
                pocket_size = atom_num.get_space_size(batch.protein_pos.detach().cpu().numpy())
                ligand_num_atoms = [atom_num.sample_atom_num(pocket_size).astype(int) for _ in range(n_data)]
                batch_ligand = torch.repeat_interleave(torch.arange(n_data), torch.tensor(ligand_num_atoms)).to(device)
            elif sample_num_atoms == 'range':
                ligand_num_atoms = list(range(current_i + 1, current_i + n_data + 1))
                batch_ligand = torch.repeat_interleave(torch.arange(n_data), torch.tensor(ligand_num_atoms)).to(device)
            elif sample_num_atoms == 'ref':
                batch_ligand = batch.ligand_element_batch
                ligand_num_atoms = scatter_sum(torch.ones_like(batch_ligand), batch_ligand, dim=0).tolist()
            else:
                raise ValueError
            
            #，KNN，KNN，，。
            #，，KNN，？，
            #KNN，KNN？KNN，，
            #，，，

            if args.data_name == 'Glide' or args.data_name == 'Vina':
                org_ligand_pos = copy.deepcopy(batch.ligand_pos) #ligand_posglide, ligand_rd_pos，
            else:
                #org_ligand_pos = copy.deepcopy(batch.ligand_pos)
                org_ligand_pos = copy.deepcopy(batch.ligand_rd_pos.float()) #ground truth

            # init ligand pos
            '''
            center_pos = scatter_mean(batch.protein_pos, batch_protein, dim=0)
            batch_center_pos = center_pos[batch_ligand]
            init_ligand_pos = batch_center_pos + torch.randn_like(batch_center_pos)
            '''

            #，。。
            center_pos = scatter_mean(batch.protein_pos, batch_protein, dim=0)
            batch_center_pos = center_pos[batch_ligand]
            init_ligand_pos = batch_center_pos + torch.randn_like(batch_center_pos)  # + 
            #init_ligand_pos = batch.ligand_pos + torch.randn_like(batch_center_pos) # + ，，，
            #init_ligand_pos = batch.ligand_pos #
            #init_ligand_pos = torch.randn_like(batch_center_pos) #，

            #org_ligand_pos = copy.deepcopy(init_ligand_pos)

            '''
            batch: ProteinLigandDataBatch(protein_element=[3106], protein_element_batch=[3106], protein_element_ptr=[9], 
            protein_molecule_name=[8], protein_pos=[3106, 3], protein_is_backbone=[3106], protein_atom_name=[8], 
            protein_atom_to_aa_type=[3106], ligand_smiles=[8], ligand_element=[282], ligand_element_batch=[282],
            ligand_element_ptr=[9], ligand_pos=[282, 3], ligand_bond_index=[2, 582], ligand_bond_type=[582], 
            ligand_bond_type_batch=[582], ligand_bond_type_ptr=[9], ligand_center_of_mass=[24], ligand_atom_feature=[282, 8], 
            ligand_hybridization=[8], protein_filename=[8], ligand_filename=[8], affinity=[8], id=[8], protein_atom_feature=[3106, 27], 
            ligand_atom_feature_full=[282], ligand_bond_feature=[582, 5])
            '''

            # init ligand v
            '''
            uniform_logits = torch.zeros(len(batch_ligand), model.num_classes).to(device)
            init_ligand_v_prob = log_sample_categorical(uniform_logits)
            init_ligand_v = init_ligand_v_prob.argmax(dim=-1)
            '''
            #，，
            init_ligand_v = batch.ligand_atom_feature_full #，813

            '''
            
            # 8
            str2id_atom_encoder = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'P': 15, 'S': 16, 'Cl': 17}
            id2str_atom_decoder = {v: k for k, v in atom_encoder.items()}
            id2str_atom_decoder = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl'}
            
            self.atom_types=[1,6,7,8,9,15,16,17]


            #print('ligand_element:', batch.ligand_element) #
            
            ligand_element: tensor([6, 8, 6, 8, 6, 6, 6, 6, 6, 6, 6, 8, 6, 6, 6, 6, 6, 6, 6, 8, 6, 6, 8, 6,
            8, 6, 6, 6, 6, 6, 6, 6, 8, 6, 6, 6, 6, 6, 6, 6, 8, 6, 6, 8, 6, 8, 6, 6,
            6, 6, 6, 6, 6, 8, 6, 6, 6, 6, 6, 6, 6, 8, 6], device='cuda:0')

            
            ##print('ligand_atom_feature_full:', batch.ligand_atom_feature_full) #8, 
            #，813，1~12


            # self.atomic_numbers = torch.LongTensor([1, 6, 7, 8, 9, 15, 16, 17])  # H, C, N, O, F, P, S, Cl
            MAP_ATOM_TYPE_AROMATIC_TO_INDEX = {
            (1, False): 0,
            (6, False): 1,
            (6, True): 2,
            (7, False): 3,
            (7, True): 4,
            (8, False): 5,
            (8, True): 6,
            (9, False): 7,
            (15, False): 8,
            (15, True): 9,
            (16, False): 10,
            (16, True): 11,
            (17, False): 12
            }
        
            

            tensor([1, 5, 1, 5, 2, 2, 2, 2, 2, 2, 2, 6, 2, 2, 2, 2, 2, 2, 2, 5, 2, 1, 5, 1,
            5, 2, 2, 2, 2, 2, 2, 2, 6, 2, 2, 2, 2, 2, 2, 2, 5, 2, 1, 5, 1, 5, 2, 2,
            2, 2, 2, 2, 2, 6, 2, 2, 2, 2, 2, 2, 2, 5, 2], device='cuda:0')
            '''
            

            '''
            r = {
            'pos': ligand_pos,
            'v': 0,
            'exp': exp_traj[-1] if len(exp_traj) else [],
            'pos_traj': pos_traj,
            'v_traj': 0,
            'exp_traj': exp_traj,
            'exp_atom_traj': exp_atom_traj,
            'v0_traj': 0,
            'vt_traj': 0,
            }

            '''

            #？
            #protein_noise = torch.randn_like(batch.protein_pos) * config.train.pos_noise_std
            #gt_protein_pos = batch.protein_pos + protein_noise
            gt_protein_pos = batch.protein_pos

            #GP.final_timesteps = 50
            if ckpt.model.diffusion_mode == 'CM':
                r = consistency_sampling_and_editing(
                    sigma_min=GP.sigma_min,
                    sigma_max=GP.sigma_max,
                    rho=GP.rho,
                    sigma_data=GP.sigma_data,
                    initial_timesteps=GP.initial_timesteps,
                    final_timesteps=GP.final_timesteps,
                    total_training_steps=GP.consistency_training_steps,


                    config = ckpt.model,

                    model = model, 
                    protein_atom_feature_dim=protein_atom_feature_dim,  #
                    ligand_atom_feature_dim =ligand_atom_feature_dim,   #


                    #ground truth
                    #protein_pos=gt_protein_pos,
                    #protein_v=batch.protein_atom_feature.float(),
                    affinity=None, #
                    #batch_protein=batch.protein_element_batch,

                    ligand_pos=None, 
                    ligand_v=None,
                    org_ligand_pos= org_ligand_pos,
                    #ligand_v=batch.ligand_atom_feature_full, #
                    #batch_ligand=batch.ligand_element_batch



                    #sample params
                    guide_mode=guide_mode,
                    value_model=value_model,
                    type_grad_weight=type_grad_weight,
                    pos_grad_weight=pos_grad_weight,

                    protein_pos=batch.protein_pos,
                    protein_v=batch.protein_atom_feature.float(),
                    batch_protein=batch_protein,

                    init_ligand_pos=init_ligand_pos,
                    init_ligand_v=init_ligand_v,
                    batch_ligand=batch_ligand,

                    num_steps=num_steps,
                    center_pos_mode=center_pos_mode,

                    ligand_bond_index = batch.ligand_bond_index, #[2, 582]
                    ligand_bond_type  = batch.ligand_bond_type,
                    ligand_bond_type_batch = batch.ligand_bond_type_batch,

                    batch_center_pos = batch_center_pos,

                    y = init_ligand_pos,

                    protein_element = batch.protein_element,
                    ligand_element  = batch.ligand_element,

                    ligand_atom_isring  = batch.ligand_atom_isring,
                    ligand_atom_isO     = batch.ligand_atom_isO,
                    ligand_atom_isN     = batch.ligand_atom_isN,

                    protein_atom_isring = batch.protein_atom_isring,
                    protein_atom_isO    = batch.protein_atom_isO,
                    protein_atom_isN    = batch.protein_atom_isN,

                        
                    cross_lig_isring_flag = batch.protein_cross_lig_isring_flag,
                    cross_lig_isO_flag = batch.protein_cross_lig_isO_flag,
                    cross_lig_isN_flag = batch.protein_cross_lig_isN_flag,

                    cross_pro_isring_flag = batch.protein_cross_pro_isring_flag,
                    cross_pro_isO_flag = batch.protein_cross_pro_isO_flag,
                    cross_pro_isN_flag = batch.protein_cross_pro_isN_flag,

                    cross_ligand    = batch.protein_cross_ligand,
                    cross_protein   = batch.protein_cross_protein,
                    cross_distance  = b_protein_cross_distance,

                    cross_bond_index = b_link_e.T,
                    cross_bond_type = batch.protein_link_t, 
                    cross_bond_index_reverse = b_link_e_reverse.T, 
                    cross_bond_type_reverse = batch.protein_link_t_reverse,

                    protein_coords_predict = batch.protein_coords_predict,

                    complex_mol = batch.complex_mol,

                    #protein_element = batch.protein_element,
                    protein_element_batch = batch.protein_element_batch,
                    protein_link_t_batch = batch.protein_link_t_batch,
                    protein_link_t_reverse_batch = batch.protein_link_t_reverse_batch,
                    
                    #ligand_element = batch.ligand_element,
                    ligand_element_batch = batch.ligand_element_batch,

                    rd_pos = batch.ligand_rd_pos.float(),
                    
                    #ligand_emb = batch.protein_ligand_emb,
                    #pocket_emb = batch.protein_pocket_emb,
                    
                    

                )
                

            elif ckpt.model.diffusion_mode == 'DDPM':
                r = model.sample_diffusion(
                    config = ckpt.model,

                    model = model, 
                    protein_atom_feature_dim=protein_atom_feature_dim,  #
                    ligand_atom_feature_dim =ligand_atom_feature_dim,   #


                    #ground truth
                    #protein_pos=gt_protein_pos,
                    #protein_v=batch.protein_atom_feature.float(),
                    affinity=None, #
                    #batch_protein=batch.protein_element_batch,

                    ligand_pos=None, 
                    ligand_v=None,
                    org_ligand_pos= org_ligand_pos,
                    #ligand_v=batch.ligand_atom_feature_full, #
                    #batch_ligand=batch.ligand_element_batch



                    #sample params
                    guide_mode=guide_mode,
                    value_model=value_model,
                    type_grad_weight=type_grad_weight,
                    pos_grad_weight=pos_grad_weight,

                    protein_pos=batch.protein_pos,
                    protein_v=batch.protein_atom_feature.float(),
                    batch_protein=batch_protein,

                    init_ligand_pos=init_ligand_pos,
                    init_ligand_v=init_ligand_v,
                    batch_ligand=batch_ligand,

                    num_steps=num_steps,
                    center_pos_mode=center_pos_mode,

                    ligand_bond_index = batch.ligand_bond_index, #[2, 582]
                    ligand_bond_type  = batch.ligand_bond_type,
                    ligand_bond_type_batch = batch.ligand_bond_type_batch,

                    batch_center_pos = batch_center_pos,

                    y = init_ligand_pos,

                    protein_element = batch.protein_element,
                    ligand_element  = batch.ligand_element,

                    ligand_atom_isring  = batch.ligand_atom_isring,
                    ligand_atom_isO     = batch.ligand_atom_isO,
                    ligand_atom_isN     = batch.ligand_atom_isN,

                    protein_atom_isring = batch.protein_atom_isring,
                    protein_atom_isO    = batch.protein_atom_isO,
                    protein_atom_isN    = batch.protein_atom_isN,

                    
                    cross_lig_isring_flag = batch.protein_cross_lig_isring_flag,
                    cross_lig_isO_flag = batch.protein_cross_lig_isO_flag,
                    cross_lig_isN_flag = batch.protein_cross_lig_isN_flag,

                    cross_pro_isring_flag = batch.protein_cross_pro_isring_flag,
                    cross_pro_isO_flag = batch.protein_cross_pro_isO_flag,
                    cross_pro_isN_flag = batch.protein_cross_pro_isN_flag,

                    cross_ligand    = batch.protein_cross_ligand,
                    cross_protein   = batch.protein_cross_protein,
                    cross_distance  = b_protein_cross_distance,

                    cross_bond_index = b_link_e.T,
                    cross_bond_type = batch.protein_link_t, 
                    cross_bond_index_reverse = b_link_e_reverse.T, 
                    cross_bond_type_reverse = batch.protein_link_t_reverse,

                    protein_coords_predict = batch.protein_coords_predict,
                    complex_mol = batch.complex_mol,

                    #protein_element = batch.protein_element,
                    protein_element_batch = batch.protein_element_batch,
                    protein_link_t_batch = batch.protein_link_t_batch,
                    protein_link_t_reverse_batch = batch.protein_link_t_reverse_batch,
                    
                    #ligand_element = batch.ligand_element,
                    ligand_element_batch = batch.ligand_element_batch,

                    rd_pos = batch.ligand_rd_pos.float(),
                    
                    #ligand_emb = batch.protein_ligand_emb,
                    #pocket_emb = batch.protein_pocket_emb,

                    )


            ligand_pos_list = copy.deepcopy(r['pos_traj'])


            #，，，？
            ligand_pos, ligand_v, ligand_pos_traj, ligand_v_traj = r['pos'], r['v'], r['pos_traj'], r['v_traj']
            ligand_v0_traj, ligand_vt_traj = r['v0_traj'], r['vt_traj']
            exp_traj = r['exp_traj'] #2
            exp_atom_traj = r['exp_atom_traj']

            coords_predict = r['coords_predict'].cpu().numpy().astype(np.float64)

            # unbatch exp
            if guide_mode == 'joint' or guide_mode == 'pdbbind_random' or guide_mode == 'valuenet' or guide_mode == 'wo':
                #all_pred_exp += exp_traj[-1]
                #all_pred_exp_traj += exp_traj
                pass
            
            # unbatch pos，ligand_pos2，，32。
            # 。，。mod == ref,
            #print('ligand_pos.shape:', ligand_pos.shape) #ligand_pos.shape: torch.Size([111, 3]).，. 37 * 3 == 111, ok
            ligand_cum_atoms = np.cumsum([0] + ligand_num_atoms) #，0
            #print('ligand_cum_atoms:', ligand_cum_atoms) #ligand_cum_atoms: [  0  37  74 111]

            ligand_pos_array = ligand_pos.cpu().numpy().astype(np.float64)
            all_pred_pos += [ligand_pos_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in
                                range(n_data)]  # num_samples * [num_atoms_i, 3]
            
            all_coords_predict += [coords_predict[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in
                                range(n_data)]  # num_samples * [num_atoms_i, 3]


            #all_pred_pos_traj
            all_step_pos = [[] for _ in range(n_data)]
            ##print('ligand_pos_traj.shape:', ligand_pos_traj) #list, 
            for p in ligand_pos_traj:  # step_i
                p_array = p.cpu().numpy().astype(np.float64)
                for k in range(n_data):
                    all_step_pos[k].append(p_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
            all_step_pos = [np.stack(step_pos) for step_pos in
                            all_step_pos]  # num_samples * [num_steps, num_atoms_i, 3]
            all_pred_pos_traj += [p for p in all_step_pos]

            
            # unbatch v
            ligand_v_array = ligand_v.cpu().numpy()
            all_pred_v += [ligand_v_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in range(n_data)]

            all_step_v = unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms)
            all_pred_v_traj += [v for v in all_step_v]
            '''
            all_step_v0 = unbatch_v_traj(ligand_v0_traj, n_data, ligand_cum_atoms)
            all_pred_v0_traj += [v for v in all_step_v0]
            all_step_vt = unbatch_v_traj(ligand_vt_traj, n_data, ligand_cum_atoms)
            all_pred_vt_traj += [v for v in all_step_vt]
            '''

            #all_step_exp_atom = unbatch_v_traj(exp_atom_traj, n_data, ligand_cum_atoms)
            #all_pred_exp_atom_traj += [v for v in all_step_exp_atom]
            
                
            t2 = time.time()
            time_list.append(t2 - t1)
            current_i += n_data
        
        
    #all_pred_exp = torch.stack(all_pred_exp,dim=0).numpy()
    #all_pred_exp_traj = torch.stack(all_pred_exp_traj,dim=0).numpy()
        
    return all_pred_pos, all_pred_v, all_pred_exp, all_pred_pos_traj, all_pred_v_traj, all_pred_exp_traj, all_pred_v0_traj, all_pred_vt_traj, all_pred_exp_atom_traj, time_list, ligand_pos_list, all_coords_predict





def sample_diffusion_ligand(model, data, num_samples, batch_size=1, device='cuda',
                            num_steps=None, center_pos_mode='protein',
                            sample_num_atoms='ref',guide_mode='joint',
                            value_model=None,
                            protein_atom_feature_dim = None,
                            ligand_atom_feature_dim  = None, 
                            protein_element = None,
                            ligand_element  = None,
                            ckpt = None,
                            data_list = None,
                            consistency_sampling_and_editing = None,
        type_grad_weight=1.,pos_grad_weight=1., args = None, config = None):
    
    all_pred_pos, all_pred_v, all_pred_exp = [], [], []
    all_coords_predict = []
    all_pred_pos_traj, all_pred_v_traj, all_pred_exp_traj, all_pred_exp_atom_traj = [], [], [], []
    all_pred_pos_traj_dict = []
    all_pred_v0_traj, all_pred_vt_traj = [], []
    time_list = []
    num_batch = int(np.ceil(num_samples / batch_size))
    current_i = 0
    
    #unmol40，，，4040，
    target_protein_cross_distance = data.protein_cross_distance #40


        
    test_sampler = torch.utils.data.distributed.DistributedSampler(data_list, shuffle = False, num_replicas=world_size, rank=local_rank) #sample

    test_loader =DataLoader(
        data_list,
        batch_size=batch_size,
        shuffle=False, #sampler，DataLoader，sampler。shuffleFalse
        #num_workers=config.train.num_workers,
        num_workers=10,
        follow_batch=FOLLOW_BATCH,  #FOLLOW_BATCH = ('protein_element', 'ligand_element', 'ligand_bond_type',) #id，_batch
        exclude_keys=collate_exclude_keys,  #，。，GNN？？
        sampler=test_sampler,
        pin_memory=True,
        prefetch_factor=10, #，
        #collate_fn=custom_collate, #pyg.Data，， <class 'KGDiff.datasets.pl_data.ProteinLigandData'>
        #exclude_keys = exclude_keys
    )

    test_loader = iter(test_loader)
    for batch in tqdm(test_loader):
        n_data = len(batch)
        batch = batch.to(device)
        b_protein_cross_distance = []
        if isinstance(batch.protein_cross_distance, list):
            for i in batch.protein_cross_distance: #protein_cross_distanceslist
                tg = torch.from_numpy(i).cuda()
                b_protein_cross_distance.append(torch.from_numpy(i).cuda())

        else:
            print('：', batch.protein_cross_distance.shape)
            b_protein_cross_distance.append(batch.protein_cross_distance.cuda())

        
        try:
            b_link_e = batch.protein_link_e.cuda(local_rank, non_blocking=True)
            b_link_e_reverse = batch.protein_link_e_reverse.cuda(local_rank, non_blocking=True)
        except (Exception, AttributeError) as e:
            b_link_e = []
            b_link_e_reverse = []
            #print('batch.protein_link_e:', batch.protein_link_e)
            if isinstance(batch.protein_link_e, list):
                for i in batch.protein_link_e:
                    b_link_e.append(torch.from_numpy(i).cuda())
            
            if isinstance(batch.protein_link_e_reverse, list):
                for i in batch.protein_link_e_reverse:
                    b_link_e_reverse.append(torch.from_numpy(i).cuda())
            
            b_link_e = torch.cat(b_link_e, dim = 0).cuda(local_rank, non_blocking=True)
            b_link_e_reverse = torch.cat(b_link_e_reverse, dim = 0).cuda(local_rank, non_blocking=True)



        t1 = time.time()
        with torch.no_grad():
            model.eval()
            #。，。mod == ref,
            batch_protein = batch.protein_element_batch
            if sample_num_atoms == 'prior':
                pocket_size = atom_num.get_space_size(batch.protein_pos.detach().cpu().numpy())
                ligand_num_atoms = [atom_num.sample_atom_num(pocket_size).astype(int) for _ in range(n_data)]
                batch_ligand = torch.repeat_interleave(torch.arange(n_data), torch.tensor(ligand_num_atoms)).to(device)
            elif sample_num_atoms == 'range':
                ligand_num_atoms = list(range(current_i + 1, current_i + n_data + 1))
                batch_ligand = torch.repeat_interleave(torch.arange(n_data), torch.tensor(ligand_num_atoms)).to(device)
            elif sample_num_atoms == 'ref':
                batch_ligand = batch.ligand_element_batch
                ligand_num_atoms = scatter_sum(torch.ones_like(batch_ligand), batch_ligand, dim=0).tolist()
            else:
                raise ValueError
            
            #，KNN，KNN，，。
            #，，KNN，？，
            #KNN，KNN？KNN，，
            #，，，

            if args.data_name == 'Glide' or args.data_name == 'Vina':
                org_ligand_pos = copy.deepcopy(batch.ligand_pos) #ligand_posglide, ligand_rd_pos，
            else:
                #org_ligand_pos = copy.deepcopy(batch.ligand_pos)
                org_ligand_pos = copy.deepcopy(batch.ligand_rd_pos.float()) #ground truth

            # init ligand pos
            '''
            center_pos = scatter_mean(batch.protein_pos, batch_protein, dim=0)
            batch_center_pos = center_pos[batch_ligand]
            init_ligand_pos = batch_center_pos + torch.randn_like(batch_center_pos)
            '''

            #，。。
            center_pos = scatter_mean(batch.protein_pos, batch_protein, dim=0)
            batch_center_pos = center_pos[batch_ligand]
            init_ligand_pos = batch_center_pos + torch.randn_like(batch_center_pos)  # + 
            #init_ligand_pos = batch.ligand_pos + torch.randn_like(batch_center_pos) # + ，，，
            #init_ligand_pos = batch.ligand_pos #
            #init_ligand_pos = torch.randn_like(batch_center_pos) #，

            #org_ligand_pos = copy.deepcopy(init_ligand_pos)

            '''
            batch: ProteinLigandDataBatch(protein_element=[3106], protein_element_batch=[3106], protein_element_ptr=[9], 
            protein_molecule_name=[8], protein_pos=[3106, 3], protein_is_backbone=[3106], protein_atom_name=[8], 
            protein_atom_to_aa_type=[3106], ligand_smiles=[8], ligand_element=[282], ligand_element_batch=[282],
            ligand_element_ptr=[9], ligand_pos=[282, 3], ligand_bond_index=[2, 582], ligand_bond_type=[582], 
            ligand_bond_type_batch=[582], ligand_bond_type_ptr=[9], ligand_center_of_mass=[24], ligand_atom_feature=[282, 8], 
            ligand_hybridization=[8], protein_filename=[8], ligand_filename=[8], affinity=[8], id=[8], protein_atom_feature=[3106, 27], 
            ligand_atom_feature_full=[282], ligand_bond_feature=[582, 5])
            '''

            # init ligand v
            '''
            uniform_logits = torch.zeros(len(batch_ligand), model.num_classes).to(device)
            init_ligand_v_prob = log_sample_categorical(uniform_logits)
            init_ligand_v = init_ligand_v_prob.argmax(dim=-1)
            '''
            #，，
            init_ligand_v = batch.ligand_atom_feature_full #，813

            '''
            
            # 8
            str2id_atom_encoder = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'P': 15, 'S': 16, 'Cl': 17}
            id2str_atom_decoder = {v: k for k, v in atom_encoder.items()}
            id2str_atom_decoder = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl'}
            
            self.atom_types=[1,6,7,8,9,15,16,17]


            #print('ligand_element:', batch.ligand_element) #
            
            ligand_element: tensor([6, 8, 6, 8, 6, 6, 6, 6, 6, 6, 6, 8, 6, 6, 6, 6, 6, 6, 6, 8, 6, 6, 8, 6,
            8, 6, 6, 6, 6, 6, 6, 6, 8, 6, 6, 6, 6, 6, 6, 6, 8, 6, 6, 8, 6, 8, 6, 6,
            6, 6, 6, 6, 6, 8, 6, 6, 6, 6, 6, 6, 6, 8, 6], device='cuda:0')

            
            ##print('ligand_atom_feature_full:', batch.ligand_atom_feature_full) #8, 
            #，813，1~12


            # self.atomic_numbers = torch.LongTensor([1, 6, 7, 8, 9, 15, 16, 17])  # H, C, N, O, F, P, S, Cl
            MAP_ATOM_TYPE_AROMATIC_TO_INDEX = {
            (1, False): 0,
            (6, False): 1,
            (6, True): 2,
            (7, False): 3,
            (7, True): 4,
            (8, False): 5,
            (8, True): 6,
            (9, False): 7,
            (15, False): 8,
            (15, True): 9,
            (16, False): 10,
            (16, True): 11,
            (17, False): 12
            }
        
            

            tensor([1, 5, 1, 5, 2, 2, 2, 2, 2, 2, 2, 6, 2, 2, 2, 2, 2, 2, 2, 5, 2, 1, 5, 1,
            5, 2, 2, 2, 2, 2, 2, 2, 6, 2, 2, 2, 2, 2, 2, 2, 5, 2, 1, 5, 1, 5, 2, 2,
            2, 2, 2, 2, 2, 6, 2, 2, 2, 2, 2, 2, 2, 5, 2], device='cuda:0')
            '''
            

            '''
            r = {
            'pos': ligand_pos,
            'v': 0,
            'exp': exp_traj[-1] if len(exp_traj) else [],
            'pos_traj': pos_traj,
            'v_traj': 0,
            'exp_traj': exp_traj,
            'exp_atom_traj': exp_atom_traj,
            'v0_traj': 0,
            'vt_traj': 0,
            }

            '''

            #？
            #protein_noise = torch.randn_like(batch.protein_pos) * config.train.pos_noise_std
            #gt_protein_pos = batch.protein_pos + protein_noise
            gt_protein_pos = batch.protein_pos

            #GP.final_timesteps = 50
            if ckpt.model.diffusion_mode == 'CM':
                r = consistency_sampling_and_editing(
                    sigma_min=GP.sigma_min,
                    sigma_max=GP.sigma_max,
                    rho=GP.rho,
                    sigma_data=GP.sigma_data,
                    initial_timesteps=GP.initial_timesteps,
                    final_timesteps=GP.final_timesteps,
                    total_training_steps=GP.consistency_training_steps,


                    config = ckpt.model,

                    model = model, 
                    protein_atom_feature_dim=protein_atom_feature_dim,  #
                    ligand_atom_feature_dim =ligand_atom_feature_dim,   #


                    #ground truth
                    #protein_pos=gt_protein_pos,
                    #protein_v=batch.protein_atom_feature.float(),
                    affinity=None, #
                    #batch_protein=batch.protein_element_batch,

                    ligand_pos=None, 
                    ligand_v=None,
                    org_ligand_pos= org_ligand_pos,
                    #ligand_v=batch.ligand_atom_feature_full, #
                    #batch_ligand=batch.ligand_element_batch



                    #sample params
                    guide_mode=guide_mode,
                    value_model=value_model,
                    type_grad_weight=type_grad_weight,
                    pos_grad_weight=pos_grad_weight,

                    protein_pos=batch.protein_pos,
                    protein_v=batch.protein_atom_feature.float(),
                    batch_protein=batch_protein,

                    init_ligand_pos=init_ligand_pos,
                    init_ligand_v=init_ligand_v,
                    batch_ligand=batch_ligand,

                    num_steps=num_steps,
                    center_pos_mode=center_pos_mode,

                    ligand_bond_index = batch.ligand_bond_index, #[2, 582]
                    ligand_bond_type  = batch.ligand_bond_type,
                    ligand_bond_type_batch = batch.ligand_bond_type_batch,

                    batch_center_pos = batch_center_pos,

                    y = init_ligand_pos,

                    protein_element = batch.protein_element,
                    ligand_element  = batch.ligand_element,

                    ligand_atom_isring  = batch.ligand_atom_isring,
                    ligand_atom_isO     = batch.ligand_atom_isO,
                    ligand_atom_isN     = batch.ligand_atom_isN,

                    protein_atom_isring = batch.protein_atom_isring,
                    protein_atom_isO    = batch.protein_atom_isO,
                    protein_atom_isN    = batch.protein_atom_isN,

                        
                    cross_lig_isring_flag = batch.protein_cross_lig_isring_flag,
                    cross_lig_isO_flag = batch.protein_cross_lig_isO_flag,
                    cross_lig_isN_flag = batch.protein_cross_lig_isN_flag,

                    cross_pro_isring_flag = batch.protein_cross_pro_isring_flag,
                    cross_pro_isO_flag = batch.protein_cross_pro_isO_flag,
                    cross_pro_isN_flag = batch.protein_cross_pro_isN_flag,

                    cross_ligand    = batch.protein_cross_ligand,
                    cross_protein   = batch.protein_cross_protein,
                    cross_distance  = b_protein_cross_distance,

                    cross_bond_index = b_link_e.T,
                    cross_bond_type = batch.protein_link_t, 
                    cross_bond_index_reverse = b_link_e_reverse.T, 
                    cross_bond_type_reverse = batch.protein_link_t_reverse,

                    protein_coords_predict = batch.protein_coords_predict,

                    complex_mol = batch.complex_mol,

                    #protein_element = batch.protein_element,
                    protein_element_batch = batch.protein_element_batch,
                    protein_link_t_batch = batch.protein_link_t_batch,
                    protein_link_t_reverse_batch = batch.protein_link_t_reverse_batch,
                    
                    #ligand_element = batch.ligand_element,
                    ligand_element_batch = batch.ligand_element_batch,

                    rd_pos = batch.ligand_rd_pos.float(),
                    
                    #ligand_emb = batch.protein_ligand_emb,
                    #pocket_emb = batch.protein_pocket_emb,
                    
                    

                )
                

            elif ckpt.model.diffusion_mode == 'DDPM':
                r = model.sample_diffusion(
                    config = ckpt.model,

                    model = model, 
                    protein_atom_feature_dim=protein_atom_feature_dim,  #
                    ligand_atom_feature_dim =ligand_atom_feature_dim,   #


                    #ground truth
                    #protein_pos=gt_protein_pos,
                    #protein_v=batch.protein_atom_feature.float(),
                    affinity=None, #
                    #batch_protein=batch.protein_element_batch,

                    ligand_pos=None, 
                    ligand_v=None,
                    org_ligand_pos= org_ligand_pos,
                    #ligand_v=batch.ligand_atom_feature_full, #
                    #batch_ligand=batch.ligand_element_batch



                    #sample params
                    guide_mode=guide_mode,
                    value_model=value_model,
                    type_grad_weight=type_grad_weight,
                    pos_grad_weight=pos_grad_weight,

                    protein_pos=batch.protein_pos,
                    protein_v=batch.protein_atom_feature.float(),
                    batch_protein=batch_protein,

                    init_ligand_pos=init_ligand_pos,
                    init_ligand_v=init_ligand_v,
                    batch_ligand=batch_ligand,

                    num_steps=num_steps,
                    center_pos_mode=center_pos_mode,

                    ligand_bond_index = batch.ligand_bond_index, #[2, 582]
                    ligand_bond_type  = batch.ligand_bond_type,
                    ligand_bond_type_batch = batch.ligand_bond_type_batch,

                    batch_center_pos = batch_center_pos,

                    y = init_ligand_pos,

                    protein_element = batch.protein_element,
                    ligand_element  = batch.ligand_element,

                    ligand_atom_isring  = batch.ligand_atom_isring,
                    ligand_atom_isO     = batch.ligand_atom_isO,
                    ligand_atom_isN     = batch.ligand_atom_isN,

                    protein_atom_isring = batch.protein_atom_isring,
                    protein_atom_isO    = batch.protein_atom_isO,
                    protein_atom_isN    = batch.protein_atom_isN,

                    
                    cross_lig_isring_flag = batch.protein_cross_lig_isring_flag,
                    cross_lig_isO_flag = batch.protein_cross_lig_isO_flag,
                    cross_lig_isN_flag = batch.protein_cross_lig_isN_flag,

                    cross_pro_isring_flag = batch.protein_cross_pro_isring_flag,
                    cross_pro_isO_flag = batch.protein_cross_pro_isO_flag,
                    cross_pro_isN_flag = batch.protein_cross_pro_isN_flag,

                    cross_ligand    = batch.protein_cross_ligand,
                    cross_protein   = batch.protein_cross_protein,
                    cross_distance  = b_protein_cross_distance,

                    cross_bond_index = b_link_e.T,
                    cross_bond_type = batch.protein_link_t, 
                    cross_bond_index_reverse = b_link_e_reverse.T, 
                    cross_bond_type_reverse = batch.protein_link_t_reverse,

                    protein_coords_predict = batch.protein_coords_predict,
                    complex_mol = batch.complex_mol,

                    #protein_element = batch.protein_element,
                    protein_element_batch = batch.protein_element_batch,
                    protein_link_t_batch = batch.protein_link_t_batch,
                    protein_link_t_reverse_batch = batch.protein_link_t_reverse_batch,
                    
                    #ligand_element = batch.ligand_element,
                    ligand_element_batch = batch.ligand_element_batch,

                    rd_pos = batch.ligand_rd_pos.float(),
                    
                    #ligand_emb = batch.protein_ligand_emb,
                    #pocket_emb = batch.protein_pocket_emb,

                    )


            ligand_pos_list = copy.deepcopy(r['pos_traj'])


            #，，，？
            ligand_pos, ligand_v, ligand_pos_traj, ligand_v_traj = r['pos'], r['v'], r['pos_traj'], r['v_traj']
            ligand_v0_traj, ligand_vt_traj = r['v0_traj'], r['vt_traj']
            exp_traj = r['exp_traj'] #2
            exp_atom_traj = r['exp_atom_traj']

            coords_predict = r['coords_predict'].cpu().numpy().astype(np.float64)

            # unbatch exp
            if guide_mode == 'joint' or guide_mode == 'pdbbind_random' or guide_mode == 'valuenet' or guide_mode == 'wo':
                #all_pred_exp += exp_traj[-1]
                #all_pred_exp_traj += exp_traj
                pass
            
            # unbatch pos，ligand_pos2，，32。
            # 。，。mod == ref,
            #print('ligand_pos.shape:', ligand_pos.shape) #ligand_pos.shape: torch.Size([111, 3]).，. 37 * 3 == 111, ok
            ligand_cum_atoms = np.cumsum([0] + ligand_num_atoms) #，0
            #print('ligand_cum_atoms:', ligand_cum_atoms) #ligand_cum_atoms: [  0  37  74 111]

            ligand_pos_array = ligand_pos.cpu().numpy().astype(np.float64)
            all_pred_pos += [ligand_pos_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in
                                range(n_data)]  # num_samples * [num_atoms_i, 3]
            
            all_coords_predict += [coords_predict[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in
                                range(n_data)]  # num_samples * [num_atoms_i, 3]


            #all_pred_pos_traj
            all_step_pos = [[] for _ in range(n_data)]
            ##print('ligand_pos_traj.shape:', ligand_pos_traj) #list, 
            for p in ligand_pos_traj:  # step_i
                p_array = p.cpu().numpy().astype(np.float64)
                for k in range(n_data):
                    all_step_pos[k].append(p_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
            all_step_pos = [np.stack(step_pos) for step_pos in
                            all_step_pos]  # num_samples * [num_steps, num_atoms_i, 3]
            all_pred_pos_traj += [p for p in all_step_pos]

            
            # unbatch v
            ligand_v_array = ligand_v.cpu().numpy()
            all_pred_v += [ligand_v_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in range(n_data)]

            all_step_v = unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms)
            all_pred_v_traj += [v for v in all_step_v]
            '''
            all_step_v0 = unbatch_v_traj(ligand_v0_traj, n_data, ligand_cum_atoms)
            all_pred_v0_traj += [v for v in all_step_v0]
            all_step_vt = unbatch_v_traj(ligand_vt_traj, n_data, ligand_cum_atoms)
            all_pred_vt_traj += [v for v in all_step_vt]
            '''

            #all_step_exp_atom = unbatch_v_traj(exp_atom_traj, n_data, ligand_cum_atoms)
            #all_pred_exp_atom_traj += [v for v in all_step_exp_atom]
            
                
            t2 = time.time()
            time_list.append(t2 - t1)
            current_i += n_data
        
        
    #all_pred_exp = torch.stack(all_pred_exp,dim=0).numpy()
    #all_pred_exp_traj = torch.stack(all_pred_exp_traj,dim=0).numpy()
        
    return all_pred_pos, all_pred_v, all_pred_exp, all_pred_pos_traj, all_pred_v_traj, all_pred_exp_traj, all_pred_v0_traj, all_pred_vt_traj, all_pred_exp_atom_traj, time_list, ligand_pos_list, all_coords_predict




def concat_conformation(base_path, step_list, num = 200, data_name = None):
    base = base_path
    #print('base:', base)
    os.makedirs(base + '/concat/', exist_ok=True)
    #print('base_path:', base_path)
    #1~25

    file_name = []

    for i in step_list:
        n = str(i)
        file_name.append(n)

    data_dict = {}
    
    for n in [data_name]:
    #for n in file_name:
        mol_list = []
        for stp in step_list:
            file = base + '/step' + str(stp) + '/' + f'gen_{n}_ligand.sdf'  #gen-0.sdf
            #print('file:', file)

            #if not os.path.exists(file):
                #continue

            mols  = Chem.SDMolSupplier(file, removeHs=True)
            mol_list.append(mols[0]) #
        
        data_dict[n] = mol_list

        # 
        # 
        source_file1 = base + '/step' + str(stp) + '/' + f'origin_{n}_ligand.sdf'

        # 
        destination_folder = base + '/concat/'

        #  shutil.copy() 
        #print('source_file1:', source_file1)
        #print('destination_folder:', destination_folder)

        shutil.copy(source_file1, destination_folder)

        '''
        try:
            shutil.copy(source_file1, destination_folder)
        except FileNotFoundError:
            break
        '''

    

    #print('data_dict_num:', len(data_dict))
    #

    for n in data_dict:
        mols = data_dict[n]
        #print('len(mols):', len(mols))
        file = base  + '/concat/' + f'gen_{n}_ligand.sdf'
        #print('file:', file)
        supp=Chem.SDWriter(file)
        for mol in mols:
            try:
                supp.write(mol)
            except Exception as e:
                print(e)
                continue
        supp.close()






def calculate_distance_matrix(A, B):
    """
    

    :
    A (torch.Tensor):  (n, 3) 
    B (torch.Tensor):  (m, 3) 

    :
    torch.Tensor:  (n, m) 
    """
    # A  (n, 3)，B  (m, 3)
    #  A  B 
    diff = A.unsqueeze(1) - B.unsqueeze(0)  # diff  (n, m, 3)
    dist_matrix = torch.sqrt(torch.sum(diff**2, dim=2))  # dist_matrix  (n, m)
    return dist_matrix




def histplot_rmsd(data, base_save_path, name):
    #print('k_name:', k_name)
    data = np.array(data)



    # 
    #sns.histplot(data, bins=30, kde=True, color='blue')
    sns.histplot(data, kde=True, color='blue')

    # 
    plt.title(f'{name.upper()} Histogram')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    # 
    #plt.show()
    save_path = os.path.join(base_save_path, f'{name}.png')
    #print('save_path:', save_path)
    plt.savefig(f'{save_path}')
    plt.close()


def is_file_exist_and_not_empty(filepath):
    # 
    if not os.path.exists(filepath):
        return False
    
    # （）
    if not os.path.isfile(filepath):
        return False
    
    # 0
    if os.path.getsize(filepath) > 0:
        return True
    else:
        return False


def sample_main(name_step = None, data_flag = None, sample_num = 10, data_path = None, data_split = None, data_name = None, args = None, name_list = None, model = None, batch_size = None):
    if batch_size != None:
        args.batch_size = batch_size

    # , ，，
    #if args.rm_outdir and os.path.exists(name_step):
        # 
        #shutil.rmtree(name_step)
        #print(f"：{name_step}")


    args.result_path = args.out_dir
        
    if args.out_dir:
        result_path = os.path.join(args.out_dir)
    else:
        result_path = args.result_path
    
    print('result_path:', result_path)
    
    os.makedirs(result_path, exist_ok=True)
    #shutil.copyfile(args.config, os.path.join(result_path, 'sample.yml'))
    logger = misc.get_logger('sampling', log_dir=result_path)
    


    # Load config
    config = misc.load_config(args.config)
    logger.info(config)
    #misc.seed_all(config.sample.seed)
    setup_seed_for_local_rank(42)
    
    #train
    if batch_size != None:
        config.model['pdbbind_random'] = 'test_premodel/final.pt'


    if args.ckpt:
        config.model['pdbbind_random'] = args.ckpt

    #，
    if args.conf_num != None:
        config.sample.num_samples = args.conf_num
    
    #config.sample.num_samples = args.conf_num


    # Load checkpoint,，
    if args.guide_mode == 'joint': #
        ckpt = torch.load(config.model['joint_ckpt'], map_location=args.device)
        value_ckpt = None
    elif args.guide_mode == 'pdbbind_random': #pdbbind，
        try:
            ckpt = torch.load(config.model['pdbbind_random'], map_location=args.device, weights_only=False) #2.6：weights_only=False
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            ckpt = torch.load(config.model['pdbbind_random'], map_location=args.device)
        #ckpt = torch.load(config.model['pdbbind_random'], map_location='cpu', weights_only=False)
        value_ckpt = None
    elif args.guide_mode == 'vina':
        ckpt = torch.load(config.model['policy_ckpt'], map_location=args.device)
        value_ckpt = None
    elif args.guide_mode == 'valuenet':
        ckpt = torch.load(config.model['policy_ckpt'], map_location=args.device)
        value_ckpt = torch.load(config.model['value_ckpt'], map_location=args.device)
    elif args.guide_mode == 'wo':
        ckpt = torch.load(config.model['policy_ckpt'], map_location=args.device)
        value_ckpt = None
    else:
        raise NotImplementedError
    
    logger.info(f"Training Config: {ckpt['config']}")
    logger.info(f"args: {args}")


    if ckpt['config'].model.diffusion_mode == 'DDPM':
        print('DDPM')
    elif ckpt['config'].model.diffusion_mode == 'CM':
        print('Consistency Model')
    
    
    # Transforms
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_atom_mode = ckpt['config'].data.transform.ligand_atom_mode
    ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)
    transform = Compose([
        protein_featurizer,
        ligand_featurizer,
        trans.FeaturizeLigandBond(),
    ])
    

    # Load dataset
    #path = '/mnt/home/fanzhiguang/47/CrossDocked2020/data/posebusters428'
    #split = '' #，，，，
    #data_flag = 'new_test' #，，None，  data_flag = None
    #data_flag = None
    if data_flag == 'new_test' or data_flag == 'old_test':
        ckpt['config'].data.path = data_path
        ckpt['config'].data.split = data_split
    print("ckpt['config'].data:", ckpt['config'].data)

    test_set_list = [] #40
    
    if GP.cross_distance_num == 'best':
        for it in list(range(10000))[:1]:
            #None，
            if GP.single_cross_distance_id:
                cross_distance_id = GP.single_cross_distance_id
            elif GP.cross_distance_num == None or GP.cross_distance_num == 'best':
                cross_distance_id = GP.cross_distance_num
            else:
                #， ，40/3 = 14, 3，14
                lenths = math.ceil(config.sample.num_samples / GP.cross_distance_num)
                id_list = []
                for j in range(GP.cross_distance_num):
                    id_list.extend([j] * lenths)
                cross_distance_id = id_list[it]

            print('cross_distance_id:', cross_distance_id)
            dataset, subsets = get_dataset(  #
                config=ckpt['config'].data,  #， #../{GP.data_type}/data/pdbbind2020_r10
                transform=transform,
                data_flag = data_flag, #，，
                data_name = data_name,
                cross_distance_num = cross_distance_id  # #it/best
            )
            



            test_set = subsets['test']
            
            train_set, val_set = subsets['train'], subsets['valid']
            logger.info(f'Train: {len(train_set)}')
            logger.info(f'Valid: {len(val_set)}')
            logger.info(f'Test: {len(test_set)}')

            if name_list:
                test_set = [x for x in test_set if x is not None and x.name in set(name_list)]
            else:
                test_set = [x for x in test_set if x is not None]
            print('None')
            logger.info(f'Test: {len(test_set)}')

            test_set_list.append(test_set)
            
        test_set_list = test_set_list * config.sample.num_samples
        
    else:
        for it in list(range(10000))[:config.sample.num_samples]:
            #None，
            if GP.single_cross_distance_id:
                cross_distance_id = GP.single_cross_distance_id
            elif GP.cross_distance_num == None or GP.cross_distance_num == 'best':
                cross_distance_id = GP.cross_distance_num
            else:
                #， ，40/3 = 14, 3，14
                lenths = math.ceil(config.sample.num_samples / GP.cross_distance_num)
                id_list = []
                for j in range(GP.cross_distance_num):
                    id_list.extend([j] * lenths)
                cross_distance_id = id_list[it]

            print('cross_distance_id:', cross_distance_id)
            dataset, subsets = get_dataset(  #
                config=ckpt['config'].data,  #， #../{GP.data_type}/data/pdbbind2020_r10
                transform=transform,
                data_flag = data_flag, #，，
                data_name = data_name,
                cross_distance_num = cross_distance_id  # #it/best
            )
            


            if args.data_flag == 'old_test':
                test_set = subsets['train']
            else:
                test_set = subsets['test'] #
                #test_set = subsets['valid']

            
            train_set, val_set = subsets['train'], subsets['valid']
            logger.info(f'Train: {len(train_set)}')
            logger.info(f'Valid: {len(val_set)}')
            logger.info(f'Test: {len(test_set)}')

            if name_list:
                test_set = [x for x in test_set if x is not None and x.name in set(name_list)]
            else:
                test_set = [x for x in test_set if x is not None]
            print('None')
            logger.info(f'Test: {len(test_set)}')

            test_set_list.append(test_set)
    





    logger.info('Building model...')

    # Load model
    model = ScorePosNet3D(
        ckpt['config'].model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim,
        
        #equiformer_args = ckpt['equiformer'],
        equiformer_args = ckpt['config'].equiformer,
        escn_args = ckpt['config'].escn,

    ).to(args.device)
    model.load_state_dict(ckpt['model'])
    

    
    model.cuda(local_rank)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True, output_device=local_rank, broadcast_buffers=False) 
    #model = torch.compile(model, mode='reduce-overhead', dynamic=True, fullgraph=True, backend='inductor') #torch.compile2.0
    try:
        model = torch.compile(model, mode='reduce-overhead', dynamic=True, fullgraph=True, backend='inductor') #torch.compile2.0
    except Exception as e:
        print('not use pytorch2.0 compile, skip')
        pass
    
    model = model.module
    

    print(f'# trainable parameters: {misc.count_parameters(model) / 1e6:.4f} M') #，
    print(f'# not trainable parameters: {misc.count_non_grad_parameters(model) / 1e6:.4f} M') #，

    consistency_sampling_and_editing = ConsistencySamplingAndEditing(
                    sigma_min = GP.sigma_min, # minimum std of noise
                    sigma_data = GP.sigma_data, # std of the data
                    )

    
    if value_ckpt is not None: #None，valuenet
        # value model
        value_model = ScorePosNet3D(
            value_ckpt['config'].model,
            protein_atom_feature_dim=protein_featurizer.feature_dim,
            ligand_atom_feature_dim=ligand_featurizer.feature_dim
        ).to(args.device)
        value_model.load_state_dict(value_ckpt['model'])
    else:
        value_model = None

    args.protein_max_atom_num = None
    args.ligand_max_atom_num  = None
    #args.equiformer = ckpt['args'].equiformer
    #print('equiformer state:', args.equiformer)

    '''
    if dist.get_rank() == 0:
        if args.out_dir: #None
            dir_path = result_path
        else:
            dir_path = result_path
        
        
        print('dir_path:', dir_path) #tmpresault
        #exit()
    
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)   # 
            print(f": {dir_path}")
            #print('gen file exist, skip')
            #continue
    
        
        
        os.makedirs(dir_path, exist_ok=True)
        
    dist.barrier()
    '''




    def safe_rmtree(path, retries=2, delay=0.5):
        if not os.path.exists(path):
            return

        for i in range(retries):
            try:
                shutil.rmtree(path)
                return
            except OSError as e:
                print(f"[rank0] : {path}, error: {e}, retry {i + 1}/{retries}")
                time.sleep(delay)

        # 
        shutil.rmtree(path, ignore_errors=True)


    # ， rank 
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    if (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0:
        if args.out_dir:
            dir_path = result_path
        else:
            dir_path = result_path

        dir_path = os.path.abspath(dir_path)

        print('dir_path:', dir_path)

        if os.path.exists(dir_path):
            safe_rmtree(dir_path)
            print(f": {dir_path}")

        os.makedirs(dir_path, exist_ok=True)

    # ， rank 
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


    
    #print('test_set:', test_set)
    for i in tqdm(list(range(len(test_set)))[args.si: args.ei]): 
        #data = test_set[args.data_id] #，，，
        data = copy.deepcopy(test_set[i])
        #print('data:', data)
        
        ligand_filename =data.ligand_filename
        #print('ligand_filename:', ligand_filename)
        complex_name = ligand_filename.split('/')[-2]
        
        #print('complex_name:', complex_name)
        if args.out_dir:
            bs_dir = os.path.join(args.out_dir, f'{complex_name}', f'step{GP.final_timesteps-1}')
        else:
            bs_dir = os.path.join(result_path, f'{complex_name}', f'step{GP.final_timesteps-1}')
        
        exit_ligand_file = os.path.join(bs_dir, f'gen_ligand_{complex_name}.sdf')
        #print('bs_dir:', bs_dir)
        #exit()
        
        #print('exit_ligand_file:', exit_ligand_file)
        if is_file_exist_and_not_empty(exit_ligand_file):
            print('gen file exist, skip')
            continue
        

            
        #if name_list and data.name in set(name_list):
            #continue
        #print('data:', data)
        #exit()
        ##print('origin_ligand.shape:', data.ligand_pos.shape) #origin_ligand.shape: torch.Size([37, 3]),torch.Size([21, 3])
        ##print('origin_protein.shape:', data.protein_pos.shape)#origin_protein.shape: torch.Size([430, 3])
        try:
            pred_pos, pred_v, pred_exp, pred_pos_traj, pred_v_traj, pred_exp_traj, pred_v0_traj, pred_vt_traj, pred_exp_atom_traj, time_list, ligand_pos_list, coords_predict = sample_diffusion_ligand(
                model, data, config.sample.num_samples,
                batch_size=args.batch_size, device=args.device,
                num_steps=config.sample.num_steps,
                center_pos_mode=config.sample.center_pos_mode,
                sample_num_atoms=config.sample.sample_num_atoms,
                guide_mode=args.guide_mode,
                value_model=value_model,
                type_grad_weight=args.type_grad_weight,
                pos_grad_weight=args.pos_grad_weight,
                args = args,
                config = config,
                consistency_sampling_and_editing = consistency_sampling_and_editing,
                protein_atom_feature_dim = protein_featurizer.feature_dim, 
                ligand_atom_feature_dim  = ligand_featurizer.feature_dim,
                ckpt = ckpt['config'],
                data_list = [dt[i] for dt in test_set_list]
            )
        except IndexError as e:
            print('error:', exit_ligand_file)
            print(e)
            continue

    
    
    
        result = {
            "pred_pos": pred_pos,
            "pred_v": pred_v,
            "pred_exp": pred_exp,
            "pred_pos_traj": pred_pos_traj,
            "pred_v_traj": pred_v_traj,
            "pred_exp_traj": pred_exp_traj,
            "pred_v0_traj": pred_v0_traj,
            "pred_vt_traj": pred_vt_traj,
            "pred_exp_atom_traj": pred_exp_atom_traj,
            "time_list": time_list,
            "ligand_pos_list": ligand_pos_list,
            "coords_predict": coords_predict,
        }

        result_list = [pred_pos, pred_v, pred_exp, pred_pos_traj, pred_v_traj, pred_exp_traj, pred_v0_traj, pred_vt_traj, pred_exp_atom_traj, time_list, ligand_pos_list, coords_predict]
        logger.info('Sample done!')
        
        results_gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(results_gathered, result_list)
        #print('results_gathered num:', len(results_gathered))
        dist.barrier()
        if dist.get_rank() == 0:
            for rk in results_gathered:
                pred_pos, pred_v, pred_exp, pred_pos_traj, pred_v_traj, pred_exp_traj, pred_v0_traj, pred_vt_traj, pred_exp_atom_traj, time_list, ligand_pos_list, coords_predict = rk


                #print('save_gen_ligand')
                #sdfpdb
                #protein_filename='BSD_ASPTE_1_130_0/2z3h_A_rec_1wn6_bst_lig_tt_docked_3_pocket10.pdb',
                #ligand_filename='BSD_ASPTE_1_130_0/2z3h_A_rec_1wn6_bst_lig_tt_docked_3.sdf',

                #protein_filename='refined-set/5l8c/5l8c_pocket10.pdb',
                #ligand_filename='refined-set/5l8c/5l8c_ligand.sdf',

                protein_filename=data.protein_filename
                ligand_filename =data.ligand_filename
                complex_name = ligand_filename.split('/')[-2] #pdbbind，crossdock
                #print('ligand_filename:', ligand_filename)


                #target_dir = os.path.join(result_path, f'result_{i}') #
                if args.out_dir:
                    target_dir = os.path.join(args.out_dir, f'{complex_name}')
                else:
                    target_dir = os.path.join(result_path, f'{complex_name}')
                
                data_path = os.path.join(ckpt['config'].data.path, 'test_set')
                
                
                s_dir = os.path.join(args.data_name)

                dir_name = ligand_filename.split('/')[-2]
                source_dir = os.path.join(args.data_name, args.data_name, dir_name)
            
                print('target_dir:',target_dir)
                print('s_dir:', s_dir)
                print('source_dir:', source_dir)
                
                #target_dir = os.path.join(result_path, f'result_{args.data_id}') os.path.dirname(file_path)
                #if os.path.exists(target_dir):
                    # If it exists, remove it
                    #shutil.rmtree(target_dir)
                #shutil.copytree(source_dir, target_dir)

                os.makedirs(target_dir, exist_ok=True)
                #torch.save(result, os.path.join(target_dir, f'{i}.pt'))

                pred_ligand_pos_ = copy.deepcopy(pred_pos)



                origin_ligand_file = f'{ligand_filename}'


                dt_mol_list = Chem.SDMolSupplier(origin_ligand_file)
                origin_mol = dt_mol_list[0]

                base_target = target_dir
                #print('base_target:', base_target)


                #for i in pred_pos_traj:
                    #print('i:', i.shape)
                #exit()
                #print('type(pred_pos_traj):', type(pred_pos_traj)) #<class 'list'>
                #print('len(pred_pos_traj):', len(pred_pos_traj)) # 3
                #print('type(pred_pos_traj[0]):', type(pred_pos_traj[0])) #<class 'numpy.ndarray'>
                #print('pred_pos_traj[0].shape:', pred_pos_traj[0].shape) #(25, 50, 3)

                #exit()
                print('pred_pos_traj num:', np.array(pred_pos_traj).shape) #(5, 15, 29, 3)
                #exit()
                for i, step in enumerate(list(range(len(pred_pos_traj[0])))[:]):
                    #，
                    if GP.multi_step:
                        #
                        pass
                    else:
                        if step != len(pred_pos_traj[0]) - 1:
                            continue 

                    #print('i:', i)
                    #print('step:', step)
                    #print('pred_pos_traj:', len(pred_pos_traj))
                    #print('pred_pos_traj[i]:', pred_pos_traj[0][i].shape)
                    pred_ligand_pos = np.array(copy.deepcopy(pred_pos_traj[0][i]))
                    #print('pred_ligand_pos:', pred_ligand_pos.shape)

                    target_dir = os.path.join(base_target)
                    print('target_dir:', target_dir) #tmpresault/5SD5/step14
                    os.makedirs(target_dir, exist_ok=True)
                    #exit()

                    try:
                        new_ligand_file = os.path.join(target_dir, f'origin_{complex_name}_ligand.sdf')
                        f1 = open(new_ligand_file, "w")
                        supp=Chem.SDWriter(f1)
                        #new_mol = Chem.SDMolSupplier(os.path.join(os.path.dirname(target_dir), f'{complex_name}_ligand.sdf'))[0]
                        new_mol = Chem.SDMolSupplier(os.path.join(source_dir, f'{complex_name}_ligand.sdf'))[0]
                        mol2 = Chem.RemoveHs(new_mol)
                        supp.write(mol2)
                        supp.close()    #

                    except Exception as e:
                        print('error:', e)
                        continue


                    new_ligand_file = os.path.join(target_dir, f'gen_{complex_name}_ligand.sdf')
                    f1 = open(new_ligand_file, "a")
                    print('new_ligand_file:', new_ligand_file)
                    supp=Chem.SDWriter(f1)

                    for j in range(len(pred_pos_traj)):
                        try:
                            new_mol = Change_Mol_D3coord(origin_mol, copy.deepcopy(pred_pos_traj[j][i]))
                        except Exception:
                            continue
                        mol2 = Chem.RemoveHs(new_mol)
                        try:
                            supp.write(mol2)
                        except Exception as e:
                            print(e)
                            continue
                    supp.close()    #

                    #exit()



                    # 
                    shutil.copy2(os.path.join(source_dir, f'{complex_name}_protein.pdb'), os.path.join(target_dir, f'{complex_name}_protein.pdb'))
                    shutil.copy2(os.path.join(source_dir, f'{complex_name}_protein_400.pdb'), os.path.join(target_dir, f'{complex_name}_protein_400.pdb'))


            

            if GP.multi_step:
                step_num_list = list(range(len(pred_pos_traj[0])))[:]
                num = 2
                data_name = complex_name
                
                try:
                    concat_conformation(base_target, step_num_list, num, data_name)
                except Exception as e:
                    pass
                







if __name__ == '__main__':
    #exclude_keys = ['protein_cross_distance']
    collate_exclude_keys = ['ligand_nbh_list']
    #
    for step in [1, 5, 10, 15, 20, 25, 50, 100][1:2]: #，？
        '''
            CUDA_VISIBLE_DEVICES=0 python scripts/sample_diffusion.py --config ./configs/sampling.yml -i 0 --guide_mode pdbbind_random \
                --type_grad_weight 100 --pos_grad_weight 25 --result_path ./cd2020_pro_0_res
        '''
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', type=str, default='./configs/sampling.yml')
        parser.add_argument('-i', '--data_id', type=int, default=81) #
        parser.add_argument('--device', type=str, default='cuda')
        parser.add_argument('--batch_size', type=int, default=None)
        parser.add_argument('--guide_mode', type=str, default='pdbbind_random', choices=['joint', 'pdbbind_random', 'vina', 'valuenet', 'wo'])  
        parser.add_argument('--type_grad_weight', type=float, default=0) #，
        parser.add_argument('--pos_grad_weight', type=float, default=0)
        parser.add_argument('--result_path', type=str, default='./test_package') #
        parser.add_argument('--log_name', type=str, default='')
        parser.add_argument('--data_flag', type=str, default='old_test', choices=['new_test', 'old_test'], help = 'use new or old data test') 
        parser.add_argument('--data_name', type=str, default='tmpdata')
        parser.add_argument('--sample_num', type=int, default=10000)
        parser.add_argument('--diffusion', type=str, default='cm', choices=['cm', 'ddpm'])
        parser.add_argument('--gnn', type=str, default='equiformer', choices=['equiformer', 'egnn', 'escn'])
        parser.add_argument('--rm_outdir', action='store_false') #，
        parser.add_argument('--conf_num', type=int, default=5)
        parser.add_argument('--test_name', type=str, default='')

        parser.add_argument('--si', type=int, default=0)
        parser.add_argument('--ei', type=int, default=500000000000)
        
        parser.add_argument('--out_dir', type=str, default=None)
        parser.add_argument('--step', type=int, default=None) 
        
        parser.add_argument('--ckpt', type=str, default=None)
        
        parser.add_argument('--user_test_data_dir', type=str, default='tmpdata')

        args = parser.parse_args() #，
        
        
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"]) #, GPU，CPU

        # 1) 
        #torch.distributed.init_process_group(backend="gloo", init_method='env://', rank=local_rank, world_size=world_size)
        torch.distributed.init_process_group(backend="nccl", init_method='env://', rank=local_rank, world_size=world_size)
        
        # 2） gpu
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        args.device = device
        args.rank = local_rank

        print('local_rank:', local_rank)
        print('device:', device)
        
        setup_seed_for_local_rank(42)
        
        
        
        if args.step:
            step = args.step
            GP.fm_step =args.step
            
        
        if args.batch_size is None:
            args.batch_size = GP.sample_batch_size

        sample_num = args.sample_num
        data_flag = args.data_flag        #posebusters，，None
        data_name = args.data_name     #，
        if data_flag == 'new_test' or data_flag == 'old_test':
            if data_name == ['posebustersv1', 'posebustersv2', 'posebusters_glide']:
                GP.max_atoms = 64 #，

        data_path  = f'{data_name}/' #，"posebusters"
        data_split = f'{data_name}_split.pt' 

        '''
        try:
            name_list = []l
            #posebustersv1_name_valid.txt
            with open(f'../{GP.data_type}/data/{data_name}/{data_name}_name_valid.txt') as f:
                for i in f:
                    name_list.append(i.strip())
        except FileNotFoundError as e:
            print(e)
            name_list = None
        '''
        name_list = None


        GP.final_timesteps = step
        GP.consistency_training_steps = step
        output_dir = f'{args.data_name}'
        sample_main(output_dir, data_flag, sample_num, data_path, data_split, data_name, args, name_list)

        
        #
        '''
        os.makedirs(os.path.join(output_dir, 'model'), exist_ok=True)
        shutil.copyfile(args.config, os.path.join(output_dir, 'model', os.path.basename(args.config)))
        shutil.copytree('./models', os.path.join(output_dir, 'model', 'models'), dirs_exist_ok=True)  
        shutil.copytree('./KGDiff', os.path.join(output_dir, 'model', 'KGDiff'), dirs_exist_ok=True)
        shutil.copytree('./EcDock', os.path.join(output_dir, 'model', 'EcDock'), dirs_exist_ok=True)
        shutil.copytree('./configs', os.path.join(output_dir, 'model', 'configs'), dirs_exist_ok=True)
        shutil.copytree('./ocp', os.path.join(output_dir, 'model', 'ocp'), dirs_exist_ok=True)
        '''
    
    

    dist.barrier()
    

    if dist.get_rank() == 0:
        random_flag = False
        for step in [1, 5, 10, 15, 20, 25, 50, 100][1:2]:
            if args.step:
                step = args.step
            #rmsd
            model = 'ecdock'
            data_name = args.data_name  #pdb2020
            gnn  = args.gnn  #ecdock，, equiformer
            diffusion = args.diffusion #ecdock，， CM/DDPM
            mode = '' #
            if model == 'ecdock':    #posebusters_ecdock_cm_equiformer_step1
                model_name = f'{args.data_name}'
                
                if args.out_dir:
                    file_path  = f'{args.out_dir}'
                else:
                    file_path  = f'../ECDock_sample_dir/{model_name}' #
                step = step - 1


            name_list = []

            #posebustersv1_name_valid.txt
            with open(f'{data_name}/{data_name}_name.txt') as f:
                for i in f:
                    name_list.append(i.strip())

            #，，，
            poor_name_list = []
            '''
            with open(f'../{GP.data_type}/data/{data_name}/{data_name}_poor_name.txt') as f:
                for i in f:
                    poor_name_list.append(i.strip())
            '''
                
            #sdf,truth_mollist，gen_mol2list，
            truth_mol, gen_mol, data_name_list = read_file(file_path, mode, flag = 'sdf', num = 10000  + 2, step = step, model = model, name_list = name_list, poor_name_list = poor_name_list)  #，rdkit mol, step
            print('truth_mol, gen_mol:', len(truth_mol), len(gen_mol))
            assert len(truth_mol) == len(gen_mol)
            

            #rmsd。401/3/5/10/40，rmsd
            resault_dict = {}
            boxplot_data_list = [] #1,5,40
            for num in [1, 3, 5, 10, 25, 40, args.conf_num][:]:
                data_dict = rmsds(truth_mol, gen_mol, num, data_name_list, random_flag) #，num
                resault_dict[num] = ['rate, rate_min, rate_mean, rmsd_mean, rmsd_std, rsmd_mid, rmsd_max, rmsd_min:', data_dict['all']]
                if num in [1, 5, 10, 25, 40, args.conf_num]:
                    boxplot_data_list.append(data_dict['data'])

            #exit()
            print(json.dumps(resault_dict, indent=4))
            #JSON
            #path = 'resault'
            path = file_path
            os.makedirs(path, exist_ok=True)

            file_name = f'{model_name}_evaluate_resault.json'
            with open(os.path.join(path, file_name), 'w') as file:
                json.dump(resault_dict, file, indent=4)
            

            #
            save_path = os.path.join(path, f'{model_name}_boxplot.png')
            boxplot(boxplot_data_list, save_path, model_name)
        
    dist.destroy_process_group()