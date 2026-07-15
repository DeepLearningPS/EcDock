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


def sample_diffusion_ligand(model, data, num_samples, batch_size=1, device='cuda',
                            num_steps=None, center_pos_mode='protein',
                            sample_num_atoms='ref',guide_mode='joint',
                            value_model=None,
                            protein_atom_feature_dim = None,
                            ligand_atom_feature_dim  = None, 
                            protein_element = None,
                            ligand_element  = None,
                            ckpt = None,
                            consistency_sampling_and_editing = None,
        type_grad_weight=1.,pos_grad_weight=1., args = None, config = None):
    
    all_pred_pos, all_pred_v, all_pred_exp = [], [], []
    all_pred_pos_traj, all_pred_v_traj, all_pred_exp_traj, all_pred_exp_atom_traj = [], [], [], []
    all_pred_pos_traj_dict = []
    all_pred_v0_traj, all_pred_vt_traj = [], []
    time_list = []
    num_batch = int(np.ceil(num_samples / batch_size))
    current_i = 0
    for i in tqdm(range(num_batch)):
        n_data = batch_size if i < num_batch - 1 else num_samples - batch_size * (num_batch - 1) #，
        batch = Batch.from_data_list([data.clone() for _ in range(n_data)], follow_batch=FOLLOW_BATCH).to(device)

        t1 = time.time()
        with torch.no_grad():
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
            org_ligand_pos = copy.deepcopy(batch.ligand_pos)

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
                    total_training_steps=GP.final_timesteps,


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
                    )


            ligand_pos_list = copy.deepcopy(r['pos_traj'])


            #，，，？
            ligand_pos, ligand_v, ligand_pos_traj, ligand_v_traj = r['pos'], r['v'], r['pos_traj'], r['v_traj']
            ligand_v0_traj, ligand_vt_traj = r['v0_traj'], r['vt_traj']
            exp_traj = r['exp_traj'] #2
            exp_atom_traj = r['exp_atom_traj']

            

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
        
    return all_pred_pos, all_pred_v, all_pred_exp, all_pred_pos_traj, all_pred_v_traj, all_pred_exp_traj, all_pred_v0_traj, all_pred_vt_traj, all_pred_exp_atom_traj, time_list, ligand_pos_list




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
            file = base + '/step' + str(stp) + '/' + f'gen_ligand_{n}.sdf'  #gen-0.sdf
            #print('file:', file)

            #if not os.path.exists(file):
                #continue

            mols  = Chem.SDMolSupplier(file, removeHs=True)
            mol_list.append(mols[0]) #
        
        data_dict[n] = mol_list

        # 
        # 
        source_file1 = base + '/step' + str(stp) + '/' + f'origin_ligand_{n}.sdf'

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
        file = base  + '/concat/' + f'gen_ligand_{n}.sdf'
        #print('file:', file)
        supp=Chem.SDWriter(file)
        for mol in mols:
            supp.write(mol)
        supp.close()



def main(name_step = None, data_flag = None, sample_num = 10, single_test = False, data_path = None, data_split = None):
    '''
        CUDA_VISIBLE_DEVICES=0 python scripts/sample_diffusion.py --config ./configs/sampling.yml -i 0 --guide_mode pdbbind_random \
            --type_grad_weight 100 --pos_grad_weight 25 --result_path ./cd2020_pro_0_res
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/sampling.yml')
    parser.add_argument('-i', '--data_id', type=int, default=81) #
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--guide_mode', type=str, default='pdbbind_random', choices=['joint', 'pdbbind_random', 'vina', 'valuenet', 'wo'])  
    parser.add_argument('--type_grad_weight', type=float, default=0) #，
    parser.add_argument('--pos_grad_weight', type=float, default=0)
    parser.add_argument('--result_path', type=str, default='./test_package') #
    parser.add_argument('--log_name', type=str, default='')
    parser.add_argument('--protein', type=str, default='')
    parser.add_argument('--ligand', type=str, default='')

    args = parser.parse_args()


    if name_step != None:
        args.result_path = name_step
    result_path = args.result_path
    os.makedirs(result_path, exist_ok=True)
    shutil.copyfile(args.config, os.path.join(result_path, 'sample.yml'))
    logger = misc.get_logger('sampling', log_dir=result_path)

    # Load config
    config = misc.load_config(args.config)
    logger.info(config)
    misc.seed_all(config.sample.seed)

    # Load checkpoint,，
    if args.guide_mode == 'joint': #
        ckpt = torch.load(config.model['joint_ckpt'], map_location=args.device)
        value_ckpt = None
    elif args.guide_mode == 'pdbbind_random': #pdbbind，
        ckpt = torch.load(config.model['pdbbind_random'], map_location=args.device)
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
    if data_flag == 'new_test':
        ckpt['config'].data.path = data_path
        ckpt['config'].data.split = data_split 
    print("ckpt['config'].data:", ckpt['config'].data)
    dataset, subsets = get_dataset(  #
        config=ckpt['config'].data,  #， #../CrossDocked2020/data/pdbbind2020_r10
        transform=transform,
        data_flag = data_flag, #，，
        single_test = single_test,
        protein = args.protein, 
        ligand = args.ligand,
    )
    if ckpt['config'].data.name == 'pl':
        test_set = subsets['test'] #
        #test_set = subsets['train']
    elif ckpt['config'].data.name == 'pdbbind':
        test_set = subsets['test']
        #test_set = subsets['train']
    else:
        raise ValueError

    
    train_set, val_set = subsets['train'], subsets['valid']

    test_set = [x for x in test_set if x is not None]

    logger.info(f'Train: {len(train_set)}')
    logger.info(f'Valid: {len(val_set)}')
    logger.info(f'Test: {len(test_set)}')

    '''
    print('None')
    train_set, val_set, test_set = [x for x in train_set if x is not None], [x for x in val_set if x is not None], [x for x in test_set if x is not None]

    logger.info(f'Train: {len(train_set)}')
    logger.info(f'Valid: {len(val_set)}')
    logger.info(f'Test: {len(test_set)}')

    #，test_set，
    
    
    raw_path = ckpt['config'].data.path #../CrossDocked2020/data/pdbbind2020_r10
    #print('raw_path:', raw_path) #raw_path: ../CrossDocked2020/data/pdbbind2020_r10
    new_path = os.path.dirname(raw_path) + '/pdb2020_test'
    os.makedirs(new_path, exist_ok=True) #../CrossDocked2020/data

    name_list = []
    for c in test_set:
        #protein_filename='v2020-other-PL/5wj6/5wj6_pocket10.pdb',
        #ligand_filename='v2020-other-PL/5wj6/5wj6_ligand.sdf',
        sr = os.path.join(raw_path, os.path.dirname(c['protein_filename']))
        tg = os.path.join(new_path, '/'.join(os.path.dirname(c['protein_filename']).split('/')[1:]))
        #print('sr:', sr)
        #print('tg:', tg)
        #shutil.copytree(sr, tg, dirs_exist_ok=True) 
        name_list.append('/'.join(os.path.dirname(c['protein_filename']).split('/')[1:]))
    
    #
    with open(os.path.dirname(new_path) + '/pdb2020_test_name.txt', 'w') as f:
        for i in name_list:
            f.write(i + '\n')

    name_list = []
    for c in train_set:
        #protein_filename='v2020-other-PL/5wj6/5wj6_pocket10.pdb',
        #ligand_filename='v2020-other-PL/5wj6/5wj6_ligand.sdf',
        sr = os.path.join(raw_path, os.path.dirname(c['protein_filename']))
        tg = os.path.join(new_path, '/'.join(os.path.dirname(c['protein_filename']).split('/')[1:]))
        #print('sr:', sr)
        #print('tg:', tg)
        #shutil.copytree(sr, tg, dirs_exist_ok=True) 
        name_list.append('/'.join(os.path.dirname(c['protein_filename']).split('/')[1:]))
    
    #
    with open(os.path.dirname(new_path) + '/pdb2020_train_name.txt', 'w') as f:
        for i in name_list:
            f.write(i + '\n')


    name_list = []
    for c in val_set:
        #protein_filename='v2020-other-PL/5wj6/5wj6_pocket10.pdb',
        #ligand_filename='v2020-other-PL/5wj6/5wj6_ligand.sdf',
        sr = os.path.join(raw_path, os.path.dirname(c['protein_filename']))
        tg = os.path.join(new_path, '/'.join(os.path.dirname(c['protein_filename']).split('/')[1:]))
        #print('sr:', sr)
        #print('tg:', tg)
        #shutil.copytree(sr, tg, dirs_exist_ok=True) 
        name_list.append('/'.join(os.path.dirname(c['protein_filename']).split('/')[1:]))
    
    #
    with open(os.path.dirname(new_path) + '/pdb2020_valide_name.txt', 'w') as f:
        for i in name_list:
            f.write(i + '\n')

    

    skip_it = np.array([472]) #，,，，，equiformer


    #

    datas = [train_set, val_set, test_set]

    protein_num = []
    ligand_num  = []
    all_num     = []

    for nm in datas:
        for dt in nm:
            protein_num.append(len(dt.protein_element))
            ligand_num.append(len(dt.ligand_element))
            all_num.append(len(dt.protein_element) + len(dt.ligand_element))

    

    #  Counter 
    counter = Counter(all_num)

    #  sorted() 
    sorted_counter = sorted(counter.items(), key=lambda x: x[0])
    sorted_counter = dict(sorted_counter)   
    print('sorted_counter num:', len(sorted_counter)) #819

    with open('atom_num_count.txt', 'w')as f:
        for k in sorted_counter:
            f.write(f'{k}: {sorted_counter[k]}\n')
    
    #for k in list(sorted_counter.keys())[:50]:
        #print(f'{k}: {sorted_counter[k]}')

    np_all_num = np.array(all_num)

    index = np_all_num > 1000
    print('atom num > 1000 graph num:', len(np_all_num[index]))#68, 8, 

    index = np_all_num > 800
    print('atom num > 800 graph num:', len(np_all_num[index])) #231, 16

    index = np_all_num > 700
    print('atom num > 700 graph num:', len(np_all_num[index])) #611, 16

    index = np_all_num > 600
    print('atom num > 600 graph num:', len(np_all_num[index])) #2260, 16

    #atom_num_list = [600, 800, 1000]
    
    # 
    categories = list(sorted_counter.keys())
    values     = list(sorted_counter.values())

    # 
    plt.bar(categories, values)

    # 
    plt.title('atom num bar')
    plt.xlabel('Categories')
    plt.ylabel('Values')

    # 
    #plt.show()
    
    #
    plt.savefig('atom_num_bar.png')
    print('all_num')

    # 
    plt.hist(all_num, bins=len(sorted_counter)//10, edgecolor='black', alpha=0.7)

    plt.savefig('atom_num_hit.png')
    
    exit()
    '''
    


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


    for i in list(range(len(test_set)))[:sample_num]: 
        #data = test_set[args.data_id] #，，，
        data = copy.deepcopy(test_set[i])
        #print('data:', data)
        #exit()
        ##print('origin_ligand.shape:', data.ligand_pos.shape) #origin_ligand.shape: torch.Size([37, 3]),torch.Size([21, 3])
        ##print('origin_protein.shape:', data.protein_pos.shape)#origin_protein.shape: torch.Size([430, 3])
        pred_pos, pred_v, pred_exp, pred_pos_traj, pred_v_traj, pred_exp_traj, pred_v0_traj, pred_vt_traj, pred_exp_atom_traj, time_list, ligand_pos_list = sample_diffusion_ligand(
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
        )
        result = {
            'data': data,
            'pred_ligand_pos': pred_pos,
            'pred_ligand_v': pred_v,
            'pred_exp': pred_exp,
            'pred_ligand_pos_traj': pred_pos_traj,
            'pred_ligand_v_traj': pred_v_traj,
            'pred_exp_traj': pred_exp_traj,
            'pred_exp_atom_traj': pred_exp_atom_traj,
            'time': time_list
        }
        logger.info('Sample done!')

        #print('save_gen_ligand')
        #sdfpdb
        #protein_filename='BSD_ASPTE_1_130_0/2z3h_A_rec_1wn6_bst_lig_tt_docked_3_pocket10.pdb',
        #ligand_filename='BSD_ASPTE_1_130_0/2z3h_A_rec_1wn6_bst_lig_tt_docked_3.sdf',

        #protein_filename='refined-set/5l8c/5l8c_pocket10.pdb',
        #ligand_filename='refined-set/5l8c/5l8c_ligand.sdf',

        protein_filename=data.protein_filename
        ligand_filename =data.ligand_filename
        complex_name = ligand_filename.split('/')[1] #pdbbind，crossdock
        #print('ligand_filename:', ligand_filename)


        #target_dir = os.path.join(result_path, f'result_{i}') #
        target_dir = os.path.join(result_path, f'{complex_name}')
        data_path = os.path.join(ckpt['config'].data.path, 'test_set')
        

        s_dir = os.path.dirname(ckpt['config'].data.path)
        if 'v2020-other-PL' in ligand_filename: ##pdb2020
            #print('use v2020-other-PL test')
            dir_name = '/'.join(ligand_filename.split('/')[:-1])
            source_dir = os.path.join(s_dir, 'pdbbind2020_r10', dir_name)
        elif 'refined-set' in ligand_filename: ##pdb2020
            #print('use refined-set test')
            dir_name = '/'.join(ligand_filename.split('/')[:-1])
            source_dir = os.path.join(s_dir, 'pdbbind2020_r10', dir_name) 
        elif data_flag == 'new_test' and single_test:
            dir_name = '/'.join(ligand_filename.split('/')[:-1])
            source_dir = os.path.join(s_dir, 'tmp_test', dir_name)
        elif data_flag == 'new_test':
            dir_name = '/'.join(ligand_filename.split('/')[:-1])
            source_dir = os.path.join(s_dir, 'posebusters', dir_name)
            #print('source_dir:', source_dir)
        else:
            #print('use test_set test')
            dir_name = ligand_filename.split('/')[0]
            source_dir = os.path.join(s_dir, 'test_set', dir_name)

        
        #target_dir = os.path.join(result_path, f'result_{args.data_id}') os.path.dirname(file_path)
        if os.path.exists(target_dir):
            # If it exists, remove it
            shutil.rmtree(target_dir)
        shutil.copytree(source_dir, target_dir)

        os.makedirs(target_dir, exist_ok=True)
        torch.save(result, os.path.join(target_dir, f'{i}.pt'))

        pred_ligand_pos_ = copy.deepcopy(pred_pos)


        if 'v2020-other-PL' in ligand_filename:
            origin_ligand_file = f'{s_dir}/pdbbind2020_r10/{ligand_filename}'
        elif 'refined-set' in ligand_filename:
            origin_ligand_file = f'{s_dir}/pdbbind2020_r10/{ligand_filename}'
        elif data_flag == 'new_test' and single_test:
            origin_ligand_file = f'{s_dir}/tmp_test/{ligand_filename}'
        elif data_flag == 'new_test':
            origin_ligand_file = f'{s_dir}/posebusters/{ligand_filename}'
        else:
            origin_ligand_file = f'{s_dir}/test_set/{ligand_filename}'

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

        for i, step in enumerate(list(range(len(pred_pos_traj[0])))[:]):
            #print('i:', i)
            #print('step:', step)
            #print('pred_pos_traj:', len(pred_pos_traj))
            #print('pred_pos_traj[i]:', pred_pos_traj[0][i].shape)
            pred_ligand_pos = np.array(copy.deepcopy(pred_pos_traj[0][i]))
            #print('pred_ligand_pos:', pred_ligand_pos.shape)

            target_dir = os.path.join(base_target, f'step{step}')
            #print('target_dir:', target_dir)
            os.makedirs(target_dir, exist_ok=True)

            try:
                new_ligand_file = os.path.join(target_dir, f'origin_ligand_{complex_name}.sdf')
                supp=Chem.SDWriter(new_ligand_file)
                new_mol = Change_Mol_D3coord(origin_mol, data.ligand_pos)
                mol2 = Chem.RemoveHs(new_mol)
                supp.write(mol2)
                supp.close()    #
            except Exception as e:
                continue


            new_ligand_file = os.path.join(target_dir, f'gen_ligand_{complex_name}.sdf')
            conformer = origin_mol.GetConformer() #conformer.GetAtomPosition(atom_idx)
            origin_pos  = origin_mol.GetConformer().GetPositions() #GetPositions(),'s'
            origin_pos2 = data.ligand_pos #


            ##print('origin_pos[:2]:', origin_pos[:2])
            ##print('origin_pos2[:2]:', origin_pos2[:2])
            supp=Chem.SDWriter(new_ligand_file)

            for j in range(len(pred_pos_traj)):
                new_mol = Change_Mol_D3coord(origin_mol, copy.deepcopy(pred_pos_traj[j][i]))
                mol2 = Chem.RemoveHs(new_mol)
                try:
                    supp.write(mol2)
                except Exception:
                    continue
            supp.close()    #


            #  XYZ 
            '''
            3
            Water molecule
            O 0.0 0.0 0.0
            H 0.757 0.586 0.0
            H -0.757 0.586 0.0

            '''

            
            #xyz？，,xyz，，
            symbols = [atom.GetSymbol() for atom in origin_mol.GetAtoms()]
            filename = os.path.join(target_dir, f'gen_ligand_{complex_name}.xyz')
            num_atoms = pred_ligand_pos[0].shape[0]
            with open(filename, 'w') as xyz_file:
                #print('type(pred_pos_traj):', type(pred_pos_traj))
                for j in range(len(pred_pos_traj)): #[sample_n, atom_n, 3]
                    pos_i = pred_pos_traj[j][i]
                    #pred_atom_type = transforms.get_atomic_number_from_index(atom_types, mode=args.atom_enc_mode)
                    xyz_file.write(f"{len(pos_i)}\n")
                    xyz_file.write("\n")
                    #print('type(pos_i):', type(pos_i))
                    for pos, id_atom in zip(pos_i, symbols):
                        #print('type(pos):', type(pos))
                        xyz_file.write(f"{id_atom} {round(pos[0], 4)} {round(pos[1], 4)} {round(pos[2], 4)}\n")
                    #break
                    
                    #xyz_file.write('\n')
                    #xyz_file.write(f"{num_atoms}\n")
                    #xyz_file.write("Generated by RDKit\n")
            
        
        '''
        type(pred_pos_traj): <class 'list'>
        type(pos_i): <class 'numpy.ndarray'>
        type(pos): <class 'numpy.ndarray'>
        '''
        
        step_num_list = list(range(len(pred_pos_traj[0])))[:]
        num = 2
        data_name = complex_name
        
        concat_conformation(base_target, step_num_list, num, data_name)







if __name__ == '__main__':
    #
    sample_num = 1
    single_test = True
    data_path = '../CrossDocked2020/data/tmp_test'
    data_split = '../CrossDocked2020/data/tmp_data_split.pt'
    for step in [1, 5, 10, 15, 25, 50][:1]: #，？
        GP.final_timesteps = step
        GP.consistency_training_steps = step

        data_flag = 'new_test' #posebusters，，None
        if data_flag == 'new_test':
            GP.max_atoms = 64
        #main(f'ecdock_cm_equiformer_posebusters424_step{step}', data_flag)
        directory = f'tmp_test/tmp_data_ecdock_cm_equiformer_step{step}'
        # 
        if os.path.exists(directory):
            # 
            shutil.rmtree(directory)
            print(f"：{directory}")
        else:
            print(f"：{directory}")

        main(f'{directory}', data_flag, sample_num, single_test, data_path, data_split)
    

        
    


    #rmsd
    model = 'ecdock'
    data_name = 'tmp_data'  #pdb2020
    step = 1
    gnn  = 'equiformer'  #ecdock，, equiformer
    diffusion = 'cm' #ecdock，， CM/DDPM
    mode = '' #
    if model == 'ecdock':    #posebusters_ecdock_cm_equiformer_step1
        file_path = f'tmp_test/{data_name}_ecdock_{diffusion}_{gnn}_step{step}' #，sdfpickle，
        model_name = data_name + '_' + model + '_' + diffusion + '_' + gnn + f'_step{step}' #
        step = step - 1

    #sdf,truth_mollist，gen_mol2list，
    truth_mol, gen_mol = read_file(file_path, mode, flag = 'sdf', num = 10000  + 2, step = step, model = model)  #，rdkit mol, step
    print('truth_mol, gen_mol:', len(truth_mol), len(gen_mol))
    assert len(truth_mol) == len(gen_mol)
    

    #rmsd。401/3/5/10/40，rmsd
    resault_dict = {}
    boxplot_data_list = [] #1,5,40
    for num in [1, 3, 5, 10, 25, 40][:]:
        data_dict = rmsds(truth_mol, gen_mol, num) #，num
        resault_dict[num] = ['rate, rmsd_mean, rmsd_std, rsmd_mid, rmsd_max, rmsd_min:', data_dict['all']]
        if num in [1, 5, 25, 40]:
            boxplot_data_list.append(data_dict['data'])

    #exit()
    print(resault_dict)
    #JSON
    path = 'tmp_test/resault'
    os.makedirs(path, exist_ok=True)

    file_path = f'{model_name}_evaluate_resault.json'
    with open(os.path.join(path, file_path), 'w') as file:
        json.dump(resault_dict, file, indent=4)
    

    #
    save_path = os.path.join(path, f'{model_name}_boxplot.png')
    boxplot(boxplot_data_list, save_path, model_name)