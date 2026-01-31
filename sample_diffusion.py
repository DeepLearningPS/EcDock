import argparse
import os
import shutil
import time
import sys
sys.path.append(os.path.abspath('./'))\



# EcConf
import numpy as np 
from rdkit import Chem
from EcConf.graphs import * 
from EcConf.utils import *
from EcConf.model import *  #这个导致的批量问题，因为torch的DataLoader和pyg的DataLoader同名了，所以要么注释掉，要么放在pyg的前面
from EcConf.comparm import *
#from EcConf.model import ConsistencySamplingAndEditing



import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.transforms import Compose
from torch_scatter import scatter_sum, scatter_mean
from tqdm.auto import tqdm

#from torch_geometric.data import DataLoader #这一步过不去
from torch_geometric.loader import DataLoader #继承了torch的DataLoader, 同名，别调用错了, 目前找不大处理形状不一样数据的方法，list也不返回

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

import warnings
warnings.filterwarnings("ignore")

local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"]) #进程数量, 对应所使用的GPU数量，而不是CPU的线程数量

collate_exclude_keys = ['ligand_nbh_list']

def setup_seed_for_local_rank(base_seed=42):
    """
    在每个 GPU（local_rank）上设置独立随机种子，避免生成相同噪声。
    在多卡并行环境中，随机数设置对训练没啥影响，但对采样有影响，因为都是同一个数据的多个构象，数据的输出是一样的，导致随机数是一样的
    """
    # local_rank 由 torchrun 自动注入环境变量
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))


    # 每个 rank 使用不同的种子
    seed = base_seed + local_rank
    #print(f"[Rank {local_rank}] 设置随机种子：{seed}")

    # 设置随机种子，确保每个 rank 的随机流不同
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 可选
    os.environ["PYTHONHASHSEED"] = str(seed)

    return seed, local_rank, world_size

def rebulit_rdkitmol(file):
    # 使用 PandasPdb 读取 PDB 文件
    ppdb = PandasPdb().read_pdb(file)
    
    # 获取原子的坐标数据框
    atom_df = ppdb.df['ATOM']
    
    # 根据指定坐标列表筛选原子
    selected_atoms = atom_df[['x_coord', 'y_coord', 'z_coord']]
    print('selected_atoms:', selected_atoms)

    # 使用 PandasPdb 数据创建一个新的 3D RDKit Mol 对象
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

    # 如果需要对新创建的分子进行后续操作，可以在这里继续
    if mol is None:
        raise ValueError("RDKit 创建 3D 分子对象失败，请检查输入数据。")

    return mol



def biopandas_read_proetein(file):
    #我们需要用来判断哪些原子是O,N, 哪些环上，因此需要判断使用rdkit来读取pdb文件，而不是直接读取文本文件，但不确定rdkit读取的顺序和从文本读取的顺序一致，因此验证一个问题
    #通过坐标大小来验证是否一致。如果不一致，依旧以文本顺序为主，然而制作一个rdkit顺序和文本顺序的映射，用于标识该原子是否在环上
    #使用biopands读取蛋白，rdkit读取蛋白时，部分原子无法读取

    # 读取 PDB 文件
    pdb = PandasPdb().read_pdb(file)
    # 获取 ATOM 数据
    atom_df = pdb.df['ATOM']

    # 去除具有相同坐标的重复原子
    # 首先，构建一个唯一坐标的标识符
    atom_df['coords'] = atom_df[['x_coord', 'y_coord', 'z_coord']].apply(tuple, axis=1)

    # 然后，通过删除重复的坐标来去除重复原子
    unique_atom_df = atom_df.drop_duplicates(subset='coords')

    # 删除临时的坐标列
    unique_atom_df = unique_atom_df.drop(columns=['coords'])

    # 重置索引
    unique_atom_df = unique_atom_df.reset_index(drop=True)  

    # 重置 'line_idx' 列的值，从 0 开始顺序编号
    unique_atom_df['line_idx'] = range(len(unique_atom_df))

    # 将去重后的数据保存回 PDB 格式
    #pdb.df['ATOM'] = unique_atom_df
    #pdb.to_pdb(path='output_protein.pdb', records=None)

    atom_df = unique_atom_df

    # 选择 'ATOM' 行并提取 x, y, z 坐标
    coordinates_df = atom_df[['x_coord', 'y_coord', 'z_coord']]
    # 转换为 NumPy 矩阵
    coordinates_matrix = coordinates_df.to_numpy()  
    #print('atom_df:', atom_df)

    return coordinates_matrix






def Change_Mol_D3coord(inputmol,coords):
    '''
        用生成的坐标替换原始的构象坐标，inputmol分子构象，coords：坐标
    '''
    molobj=copy.deepcopy(inputmol)
    conformer=molobj.GetConformer()
    id=conformer.GetId()
    for cid,xyz in enumerate(coords):
        ##print(xyz[0],xyz[1],xyz[2],type(xyz))
        conformer.SetAtomPosition(cid,Point3D(float(xyz[0]),float(xyz[1]),float(xyz[2]))) #更新构象每个原子的坐标
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
    
    #如果使用unmol的40个不同的距离矩阵，则需要提前获取一下，然后各个批量的，因为同一个复合物的40个构象保存好40个对应的距离矩阵是一样的，所以用第一个就可以
    target_protein_cross_distance = data.protein_cross_distance #此时数据量是40

    for i in tqdm(range(num_batch)):
        sub_data_list = data_list[i * batch_size : (i + 1) * batch_size]
        #print('data:', data)
        # 遍历并打印所有属性名和值
        #for attr, value in data._store._mapping.__dict__.items():
        #for attr, value in data._store._mapping.items():
            #print(f"属性名: {attr}")


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
        n_data = batch_size if i < num_batch - 1 else num_samples - batch_size * (num_batch - 1) #不够一个批量，则有多少为多少
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
        #print('ligand_bond_type:', batch.ligand_bond_type.shape) #bond_index（pyg的内部机制）自动按list连接了，但link_e没有，怎么实现呢？实现不了，只能改数据形状，

        #print('protein_protein_link_e:', batch.protein_link_e.shape)
        #print('protein_protein_link_t:', batch.protein_link_t.shape)

        #print('data:', data)
        #raise Exception('test')
        b_protein_cross_distance = []
        #b_cross_bond_index, b_cross_bond_type, b_cross_bond_index_reverse, b_cross_bond_type_reverse = [], [], [], []
        if isinstance(batch.protein_cross_distance, list):
            #print('多个距离矩阵：', len(batch.protein_cross_distance)) #5
            #print('1个距离矩阵：', batch.protein_cross_distance[0].shape) #多个距离矩阵： 5
            #assert np.allclose(batch.protein_cross_distance[0], batch.protein_cross_distance[1], atol=0.02)
            #exit()

            for i in batch.protein_cross_distance: #protein_cross_distances是一个list
                tg = torch.from_numpy(i).cuda()
                #print('tg.shape:', tg.shape)
                #print('tg[:3]:', tg[:3])
                b_protein_cross_distance.append(torch.from_numpy(i).cuda())

                #ii = torch.stack(list(i), dim = 0) #集合转list，再连接，恢复成张量
                #print('set ii.shape:', ii.shape)
                #print('set ii[:3]:', ii[:3])
                #b_protein_cross_distance.append(ii.cuda())

        else:
            print('单个距离矩阵：', batch.protein_cross_distance.shape)
            #exit()
            b_protein_cross_distance.append(batch.protein_cross_distance.cuda())
            #b_cross_bond_index.append(batch.cross_bond_index.cuda())
            #b_cross_bond_type.append(batch.cross_bond_type.cuda())
            #b_cross_bond_index_reverse.append(batch.cross_bond_index_reverse.cuda()) 
            #b_cross_bond_type_reverse.append(batch.cross_bond_type_reverse.cuda())
        
        #b_link_e = batch.protein_link_e.cuda()
        #b_link_e_reverse = batch.protein_link_e_reverse.cuda()
        
        
        #numpy 版本导致的
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
            #截断原子的索引范围。模式不同，截断的方法也不同。mod == ref,则表示生成的原子数量等于配体
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
            
            #现在面临一个问题，在训练阶段构建KNN图时，我们用真实的坐标来固定KNN图，这点和使用全连接图来固定连接表是一样的，好处在于减少计算成本。然而
            #在采样过程中，由于不知道真实的坐标，这个时候只能采样随机坐标的方式来构建KNN图，但是这样会不会有问题？或者说，如果想在采样过程中还想固定
            #KNN，那么我们能不能在KNN图和全连接图之间建立映射关系呢？或者更直接的就是蛋白使用KNN图，配体使用全连接图，这样就实现了固定邻接表的要求了
            #主要蛋白的坐标是已知，所以不用管它，且在训练过程中，我们不更新蛋白坐标

            if args.data_name == 'Glide' or args.data_name == 'Vina':
                org_ligand_pos = copy.deepcopy(batch.ligand_pos) #这里的ligand_pos实际上就是glide生成的, 其实这里换成ligand_rd_pos也行，不影响模型
            else:
                #org_ligand_pos = copy.deepcopy(batch.ligand_pos)
                org_ligand_pos = copy.deepcopy(batch.ligand_rd_pos.float()) #要换成非ground truth测试数据

            # init ligand pos
            '''
            center_pos = scatter_mean(batch.protein_pos, batch_protein, dim=0)
            batch_center_pos = center_pos[batch_ligand]
            init_ligand_pos = batch_center_pos + torch.randn_like(batch_center_pos)
            '''

            #如果以蛋白为中心作为初始值，则对接生成的配体是有问题的。所有会集中在一点。
            center_pos = scatter_mean(batch.protein_pos, batch_protein, dim=0)
            batch_center_pos = center_pos[batch_ligand]
            init_ligand_pos = batch_center_pos + torch.randn_like(batch_center_pos)  #以蛋白质心 + 正太分布
            #init_ligand_pos = batch.ligand_pos + torch.randn_like(batch_center_pos) #原始坐标 + 正太分布，这种方法不可取，因为采样生成的过程中，坐标未知
            #init_ligand_pos = batch.ligand_pos #无噪音试试
            #init_ligand_pos = torch.randn_like(batch_center_pos) #纯正态分布，有些结构生成不了

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
            #原子类型是已知的，所以这里不用初始化，直接使用真实的值
            init_ligand_v = batch.ligand_atom_feature_full #这里的原子类型是加了芳香原子之后，由原来的8个变成了13个

            '''
            
            # 8个原子类型的映射关系
            str2id_atom_encoder = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'P': 15, 'S': 16, 'Cl': 17}
            id2str_atom_decoder = {v: k for k, v in atom_encoder.items()}
            id2str_atom_decoder = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl'}
            
            self.atom_types=[1,6,7,8,9,15,16,17]


            #print('ligand_element:', batch.ligand_element) #存在的是原子在周期表的索引
            
            ligand_element: tensor([6, 8, 6, 8, 6, 6, 6, 6, 6, 6, 6, 8, 6, 6, 6, 6, 6, 6, 6, 8, 6, 6, 8, 6,
            8, 6, 6, 6, 6, 6, 6, 6, 8, 6, 6, 6, 6, 6, 6, 6, 8, 6, 6, 8, 6, 8, 6, 6,
            6, 6, 6, 6, 6, 8, 6, 6, 6, 6, 6, 6, 6, 8, 6], device='cuda:0')

            
            ##print('ligand_atom_feature_full:', batch.ligand_atom_feature_full) #总共有8种原子类型, 
            #这里的原子类型是加了芳香原子之后，由原来的8个变成了13个，编号从1~12


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

            #给蛋白加噪？为了让生成的配体更稳定
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
                    protein_atom_feature_dim=protein_atom_feature_dim,  #蛋白的原子类型数量
                    ligand_atom_feature_dim =ligand_atom_feature_dim,   #配体的原子类型数量


                    #ground truth
                    #protein_pos=gt_protein_pos,
                    #protein_v=batch.protein_atom_feature.float(),
                    affinity=None, #真实的亲和度
                    #batch_protein=batch.protein_element_batch,

                    ligand_pos=None, 
                    ligand_v=None,
                    org_ligand_pos= org_ligand_pos,
                    #ligand_v=batch.ligand_atom_feature_full, #真实的原子类型
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
                    protein_atom_feature_dim=protein_atom_feature_dim,  #蛋白的原子类型数量
                    ligand_atom_feature_dim =ligand_atom_feature_dim,   #配体的原子类型数量


                    #ground truth
                    #protein_pos=gt_protein_pos,
                    #protein_v=batch.protein_atom_feature.float(),
                    affinity=None, #真实的亲和度
                    #batch_protein=batch.protein_element_batch,

                    ligand_pos=None, 
                    ligand_v=None,
                    org_ligand_pos= org_ligand_pos,
                    #ligand_v=batch.ligand_atom_feature_full, #真实的原子类型
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


            #值得注意的是，在神经网络中，配体和蛋白分组减去了蛋白质心的，最后对接或保存结构的时候是否需要再加上质心了？目前来看是不需要的
            ligand_pos, ligand_v, ligand_pos_traj, ligand_v_traj = r['pos'], r['v'], r['pos_traj'], r['v_traj']
            ligand_v0_traj, ligand_vt_traj = r['v0_traj'], r['vt_traj']
            exp_traj = r['exp_traj'] #是一个2维度张量
            exp_atom_traj = r['exp_atom_traj']

            coords_predict = r['coords_predict'].cpu().numpy().astype(np.float64)

            # unbatch exp
            if guide_mode == 'joint' or guide_mode == 'pdbbind_random' or guide_mode == 'valuenet' or guide_mode == 'wo':
                #all_pred_exp += exp_traj[-1]
                #all_pred_exp_traj += exp_traj
                pass
            
            # unbatch pos，预测出来的分子ligand_pos坐标形状是2维度，这是把当前批量生成的分子给连接在一起，由原来的3维度变成了2维度。而我们则需要在这个图上截取不同数量的原子
            # 截断原子的索引范围。模式不同，截断的方法也不同。mod == ref,则表示生成的原子数量等于配体
            #print('ligand_pos.shape:', ligand_pos.shape) #ligand_pos.shape: torch.Size([111, 3]).关于对接，我们希望生成配体的坐标形状等于真实的配体坐标. 37 * 3 == 111, ok
            ligand_cum_atoms = np.cumsum([0] + ligand_num_atoms) #截断原子的取值范围，加上开始的下标索引0
            #print('ligand_cum_atoms:', ligand_cum_atoms) #ligand_cum_atoms: [  0  37  74 111]

            ligand_pos_array = ligand_pos.cpu().numpy().astype(np.float64)
            all_pred_pos += [ligand_pos_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in
                                range(n_data)]  # num_samples * [num_atoms_i, 3]
            
            all_coords_predict += [coords_predict[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in
                                range(n_data)]  # num_samples * [num_atoms_i, 3]


            #弄清楚all_pred_pos_traj是啥
            all_step_pos = [[] for _ in range(n_data)]
            ##print('ligand_pos_traj.shape:', ligand_pos_traj) #是一个list, 且形状不一样
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
    
    #如果使用unmol的40个不同的距离矩阵，则需要提前获取一下，然后各个批量的，因为同一个复合物的40个构象保存好40个对应的距离矩阵是一样的，所以用第一个就可以
    target_protein_cross_distance = data.protein_cross_distance #此时数据量是40


        
    test_sampler = torch.utils.data.distributed.DistributedSampler(data_list, shuffle = False, num_replicas=world_size, rank=local_rank) #打乱数据在sample控制

    test_loader =DataLoader(
        data_list,
        batch_size=batch_size,
        shuffle=False, #有了sampler，则数据的打乱不要在DataLoader设置了，由sampler控制。所以这里shuffle必须为False
        #num_workers=config.train.num_workers,
        num_workers=10,
        follow_batch=FOLLOW_BATCH,  #FOLLOW_BATCH = ('protein_element', 'ligand_element', 'ligand_bond_type',) #对这些数据形成批量id，命名为数据名_batch
        exclude_keys=collate_exclude_keys,  #表示在组合批量时候，把邻接表给排除。问题是如果排除邻接表，按怎么批量传递给GNN邻接表？这一步操作在哪？
        sampler=test_sampler,
        pin_memory=True,
        prefetch_factor=10, #预加载内存，可以扩大点
        #collate_fn=custom_collate, #这种事针对pyg.Data类的数据的，但现在的传递的是自定义的类，不起作用 <class 'KGDiff.datasets.pl_data.ProteinLigandData'>
        #exclude_keys = exclude_keys
    )

    test_loader = iter(test_loader)
    for batch in tqdm(test_loader):
        n_data = len(batch)
        batch = batch.to(device)
        b_protein_cross_distance = []
        if isinstance(batch.protein_cross_distance, list):
            for i in batch.protein_cross_distance: #protein_cross_distances是一个list
                tg = torch.from_numpy(i).cuda()
                b_protein_cross_distance.append(torch.from_numpy(i).cuda())

        else:
            print('单个距离矩阵：', batch.protein_cross_distance.shape)
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
            #截断原子的索引范围。模式不同，截断的方法也不同。mod == ref,则表示生成的原子数量等于配体
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
            
            #现在面临一个问题，在训练阶段构建KNN图时，我们用真实的坐标来固定KNN图，这点和使用全连接图来固定连接表是一样的，好处在于减少计算成本。然而
            #在采样过程中，由于不知道真实的坐标，这个时候只能采样随机坐标的方式来构建KNN图，但是这样会不会有问题？或者说，如果想在采样过程中还想固定
            #KNN，那么我们能不能在KNN图和全连接图之间建立映射关系呢？或者更直接的就是蛋白使用KNN图，配体使用全连接图，这样就实现了固定邻接表的要求了
            #主要蛋白的坐标是已知，所以不用管它，且在训练过程中，我们不更新蛋白坐标

            if args.data_name == 'Glide' or args.data_name == 'Vina':
                org_ligand_pos = copy.deepcopy(batch.ligand_pos) #这里的ligand_pos实际上就是glide生成的, 其实这里换成ligand_rd_pos也行，不影响模型
            else:
                #org_ligand_pos = copy.deepcopy(batch.ligand_pos)
                org_ligand_pos = copy.deepcopy(batch.ligand_rd_pos.float()) #要换成非ground truth测试数据

            # init ligand pos
            '''
            center_pos = scatter_mean(batch.protein_pos, batch_protein, dim=0)
            batch_center_pos = center_pos[batch_ligand]
            init_ligand_pos = batch_center_pos + torch.randn_like(batch_center_pos)
            '''

            #如果以蛋白为中心作为初始值，则对接生成的配体是有问题的。所有会集中在一点。
            center_pos = scatter_mean(batch.protein_pos, batch_protein, dim=0)
            batch_center_pos = center_pos[batch_ligand]
            init_ligand_pos = batch_center_pos + torch.randn_like(batch_center_pos)  #以蛋白质心 + 正太分布
            #init_ligand_pos = batch.ligand_pos + torch.randn_like(batch_center_pos) #原始坐标 + 正太分布，这种方法不可取，因为采样生成的过程中，坐标未知
            #init_ligand_pos = batch.ligand_pos #无噪音试试
            #init_ligand_pos = torch.randn_like(batch_center_pos) #纯正态分布，有些结构生成不了

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
            #原子类型是已知的，所以这里不用初始化，直接使用真实的值
            init_ligand_v = batch.ligand_atom_feature_full #这里的原子类型是加了芳香原子之后，由原来的8个变成了13个

            '''
            
            # 8个原子类型的映射关系
            str2id_atom_encoder = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'P': 15, 'S': 16, 'Cl': 17}
            id2str_atom_decoder = {v: k for k, v in atom_encoder.items()}
            id2str_atom_decoder = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl'}
            
            self.atom_types=[1,6,7,8,9,15,16,17]


            #print('ligand_element:', batch.ligand_element) #存在的是原子在周期表的索引
            
            ligand_element: tensor([6, 8, 6, 8, 6, 6, 6, 6, 6, 6, 6, 8, 6, 6, 6, 6, 6, 6, 6, 8, 6, 6, 8, 6,
            8, 6, 6, 6, 6, 6, 6, 6, 8, 6, 6, 6, 6, 6, 6, 6, 8, 6, 6, 8, 6, 8, 6, 6,
            6, 6, 6, 6, 6, 8, 6, 6, 6, 6, 6, 6, 6, 8, 6], device='cuda:0')

            
            ##print('ligand_atom_feature_full:', batch.ligand_atom_feature_full) #总共有8种原子类型, 
            #这里的原子类型是加了芳香原子之后，由原来的8个变成了13个，编号从1~12


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

            #给蛋白加噪？为了让生成的配体更稳定
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
                    protein_atom_feature_dim=protein_atom_feature_dim,  #蛋白的原子类型数量
                    ligand_atom_feature_dim =ligand_atom_feature_dim,   #配体的原子类型数量


                    #ground truth
                    #protein_pos=gt_protein_pos,
                    #protein_v=batch.protein_atom_feature.float(),
                    affinity=None, #真实的亲和度
                    #batch_protein=batch.protein_element_batch,

                    ligand_pos=None, 
                    ligand_v=None,
                    org_ligand_pos= org_ligand_pos,
                    #ligand_v=batch.ligand_atom_feature_full, #真实的原子类型
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
                    protein_atom_feature_dim=protein_atom_feature_dim,  #蛋白的原子类型数量
                    ligand_atom_feature_dim =ligand_atom_feature_dim,   #配体的原子类型数量


                    #ground truth
                    #protein_pos=gt_protein_pos,
                    #protein_v=batch.protein_atom_feature.float(),
                    affinity=None, #真实的亲和度
                    #batch_protein=batch.protein_element_batch,

                    ligand_pos=None, 
                    ligand_v=None,
                    org_ligand_pos= org_ligand_pos,
                    #ligand_v=batch.ligand_atom_feature_full, #真实的原子类型
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


            #值得注意的是，在神经网络中，配体和蛋白分组减去了蛋白质心的，最后对接或保存结构的时候是否需要再加上质心了？目前来看是不需要的
            ligand_pos, ligand_v, ligand_pos_traj, ligand_v_traj = r['pos'], r['v'], r['pos_traj'], r['v_traj']
            ligand_v0_traj, ligand_vt_traj = r['v0_traj'], r['vt_traj']
            exp_traj = r['exp_traj'] #是一个2维度张量
            exp_atom_traj = r['exp_atom_traj']

            coords_predict = r['coords_predict'].cpu().numpy().astype(np.float64)

            # unbatch exp
            if guide_mode == 'joint' or guide_mode == 'pdbbind_random' or guide_mode == 'valuenet' or guide_mode == 'wo':
                #all_pred_exp += exp_traj[-1]
                #all_pred_exp_traj += exp_traj
                pass
            
            # unbatch pos，预测出来的分子ligand_pos坐标形状是2维度，这是把当前批量生成的分子给连接在一起，由原来的3维度变成了2维度。而我们则需要在这个图上截取不同数量的原子
            # 截断原子的索引范围。模式不同，截断的方法也不同。mod == ref,则表示生成的原子数量等于配体
            #print('ligand_pos.shape:', ligand_pos.shape) #ligand_pos.shape: torch.Size([111, 3]).关于对接，我们希望生成配体的坐标形状等于真实的配体坐标. 37 * 3 == 111, ok
            ligand_cum_atoms = np.cumsum([0] + ligand_num_atoms) #截断原子的取值范围，加上开始的下标索引0
            #print('ligand_cum_atoms:', ligand_cum_atoms) #ligand_cum_atoms: [  0  37  74 111]

            ligand_pos_array = ligand_pos.cpu().numpy().astype(np.float64)
            all_pred_pos += [ligand_pos_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in
                                range(n_data)]  # num_samples * [num_atoms_i, 3]
            
            all_coords_predict += [coords_predict[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in
                                range(n_data)]  # num_samples * [num_atoms_i, 3]


            #弄清楚all_pred_pos_traj是啥
            all_step_pos = [[] for _ in range(n_data)]
            ##print('ligand_pos_traj.shape:', ligand_pos_traj) #是一个list, 且形状不一样
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
    #将1~25内生成数据的第一个构象连接一起

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
            mol_list.append(mols[0]) #取第一个即可
        
        data_dict[n] = mol_list

        # 把原始的构象和蛋白复制到新的文件夹里面
        # 源文件路径
        source_file1 = base + '/step' + str(stp) + '/' + f'origin_ligand_{n}.sdf'

        # 目标文件夹路径
        destination_folder = base + '/concat/'

        # 使用 shutil.copy() 复制文件
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
    #写入文件

    for n in data_dict:
        mols = data_dict[n]
        #print('len(mols):', len(mols))
        file = base  + '/concat/' + f'gen_ligand_{n}.sdf'
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
    计算两个坐标矩阵之间的欧氏距离

    参数:
    A (torch.Tensor): 大小为 (n, 3) 的坐标矩阵
    B (torch.Tensor): 大小为 (m, 3) 的坐标矩阵

    返回:
    torch.Tensor: 大小为 (n, m) 的距离矩阵
    """
    # A 的形状为 (n, 3)，B 的形状为 (m, 3)
    # 计算 A 的每个点与 B 的每个点之间的距离
    diff = A.unsqueeze(1) - B.unsqueeze(0)  # diff 的形状为 (n, m, 3)
    dist_matrix = torch.sqrt(torch.sum(diff**2, dim=2))  # dist_matrix 的形状为 (n, m)
    return dist_matrix




def histplot_rmsd(data, base_save_path, name):
    #print('k_name:', k_name)
    data = np.array(data)



    # 绘制直方图
    #sns.histplot(data, bins=30, kde=True, color='blue')
    sns.histplot(data, kde=True, color='blue')

    # 添加标题和标签
    plt.title(f'{name.upper()} Histogram')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    # 显示图形
    #plt.show()
    save_path = os.path.join(base_save_path, f'{name}.png')
    #print('save_path:', save_path)
    plt.savefig(f'{save_path}')
    plt.close()


def is_file_exist_and_not_empty(filepath):
    # 检查文件是否存在
    if not os.path.exists(filepath):
        return False
    
    # 检查是否是文件（不是目录）
    if not os.path.isfile(filepath):
        return False
    
    # 检查文件大小是否大于0
    if os.path.getsize(filepath) > 0:
        return True
    else:
        return False


def sample_main(name_step = None, data_flag = None, sample_num = 10, data_path = None, data_split = None, data_name = None, args = None, name_list = None, model = None, batch_size = None):
    if batch_size != None:
        args.batch_size = batch_size

    # 检查目录是否存在, 保证为空，一次只生成一个复合物，一次只评估一个复合物
    #if args.rm_outdir and os.path.exists(name_step):
        # 删除目录
        #shutil.rmtree(name_step)
        #print(f"已删除目录：{name_step}")

    if name_step != None:
        args.result_path = name_step
        
    if args.out_dir:
        result_path = os.path.join(args.out_dir, os.path.basename(args.result_path), )
    else:
        result_path = args.result_path
    
    os.makedirs(result_path, exist_ok=True)
    shutil.copyfile(args.config, os.path.join(result_path, 'sample.yml'))
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

    #如果指定了构象数量，则使用
    if args.conf_num != None:
        config.sample.num_samples = args.conf_num
    
    #config.sample.num_samples = args.conf_num


    # Load checkpoint,不同的模式，选择不同的模型
    if args.guide_mode == 'joint': #这是从整理好的药物设计数据来生成分子的
        ckpt = torch.load(config.model['joint_ckpt'], map_location=args.device)
        value_ckpt = None
    elif args.guide_mode == 'pdbbind_random': #从pdbbind直接来生成分子，我们用这个做对接
        try:
            ckpt = torch.load(config.model['pdbbind_random'], map_location=args.device, weights_only=False) #2.6之后新增：weights_only=False允许加载模型之外的字典参数
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
        print('使用DDPM')
    elif ckpt['config'].model.diffusion_mode == 'CM':
        print('使用Consistency Model')
    
    
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
    #split = '' #这里存放训练验证测试的数据集分割，我们需要自己制作一个，训练，验证为空，测试为所需要的集合
    #data_flag = 'new_test' #使用特殊的测试集，则开启，否则赋值为None，  data_flag = None
    #data_flag = None
    if data_flag == 'new_test':
        ckpt['config'].data.path = data_path
        ckpt['config'].data.split = data_split
    print("ckpt['config'].data:", ckpt['config'].data)

    test_set_list = [] #存放40个距离矩阵对应的数据集
    
    if GP.cross_distance_num == 'best':
        for it in list(range(10000))[:1]:
            #如果改参数不为None，则启动单距离矩阵
            if GP.single_cross_distance_id:
                cross_distance_id = GP.single_cross_distance_id
            elif GP.cross_distance_num == None or GP.cross_distance_num == 'best':
                cross_distance_id = GP.cross_distance_num
            else:
                #分组，向上取整 ，如40/3 = 14, 分3组，每组的长度是14
                lenths = math.ceil(config.sample.num_samples / GP.cross_distance_num)
                id_list = []
                for j in range(GP.cross_distance_num):
                    id_list.extend([j] * lenths)
                cross_distance_id = id_list[it]

            print('cross_distance_id:', cross_distance_id)
            dataset, subsets = get_dataset(  #数据来自于训练模型中保存的配置文件
                config=ckpt['config'].data,  #如果单独传递新的测试集，则改这里 #../{GP.data_type}/data/pdbbind2020_r10
                transform=transform,
                data_flag = data_flag, #加一个数据标志位，如果是仅仅使用新的测试，则特殊处理
                data_name = data_name,
                cross_distance_num = cross_distance_id  # #it/best
            )
            


            if args.data_flag == 'old_test':
                test_set = subsets['train']
            else:
                test_set = subsets['test'] #这里我们用训练集来测试
                #test_set = subsets['valid']

            
            train_set, val_set = subsets['train'], subsets['valid']
            logger.info(f'Train: {len(train_set)}')
            logger.info(f'Valid: {len(val_set)}')
            logger.info(f'Test: {len(test_set)}')

            if name_list:
                test_set = [x for x in test_set if x is not None and x.name in set(name_list)]
            else:
                test_set = [x for x in test_set if x is not None]
            print('去除None')
            logger.info(f'Test: {len(test_set)}')

            test_set_list.append(test_set)
            
        test_set_list = test_set_list * config.sample.num_samples
        
    else:
        for it in list(range(10000))[:config.sample.num_samples]:
            #如果改参数不为None，则启动单距离矩阵
            if GP.single_cross_distance_id:
                cross_distance_id = GP.single_cross_distance_id
            elif GP.cross_distance_num == None or GP.cross_distance_num == 'best':
                cross_distance_id = GP.cross_distance_num
            else:
                #分组，向上取整 ，如40/3 = 14, 分3组，每组的长度是14
                lenths = math.ceil(config.sample.num_samples / GP.cross_distance_num)
                id_list = []
                for j in range(GP.cross_distance_num):
                    id_list.extend([j] * lenths)
                cross_distance_id = id_list[it]

            print('cross_distance_id:', cross_distance_id)
            dataset, subsets = get_dataset(  #数据来自于训练模型中保存的配置文件
                config=ckpt['config'].data,  #如果单独传递新的测试集，则改这里 #../{GP.data_type}/data/pdbbind2020_r10
                transform=transform,
                data_flag = data_flag, #加一个数据标志位，如果是仅仅使用新的测试，则特殊处理
                data_name = data_name,
                cross_distance_num = cross_distance_id  # #it/best
            )
            


            if args.data_flag == 'old_test':
                test_set = subsets['train']
            else:
                test_set = subsets['test'] #这里我们用训练集来测试
                #test_set = subsets['valid']

            
            train_set, val_set = subsets['train'], subsets['valid']
            logger.info(f'Train: {len(train_set)}')
            logger.info(f'Valid: {len(val_set)}')
            logger.info(f'Test: {len(test_set)}')

            if name_list:
                test_set = [x for x in test_set if x is not None and x.name in set(name_list)]
            else:
                test_set = [x for x in test_set if x is not None]
            print('去除None')
            logger.info(f'Test: {len(test_set)}')

            test_set_list.append(test_set)
    

    #测试保存的口袋蛋白和我们模型正在使用数据集中原子顺序是否一致，允许误差在0.02。另外验证通过rdkit和from biopandas.pdb import PandasPdb读取口袋蛋白的原子顺序和数量是否一致
    '''计算距离矩阵和参考的之间的rmsd'''

    #计算参考配体到蛋白的距离矩阵
    true_distance_list   = []
    unimol_distance_list = []
    rsmd_list            = []

    '''
    dt: ProteinLigandData(
    protein_element=[125],
    protein_molecule_name='pocket',
    protein_pos=[125, 3],
    protein_is_backbone=[125],
    protein_atom_name=[125],
    protein_atom_to_aa_type=[125],
    protein_atom_isring=[125],
    protein_atom_isO=[125],
    protein_atom_isN=[125],
    protein_cross_lig_isring_flag=[13],
    protein_cross_lig_isO_flag=[13],
    protein_cross_lig_isN_flag=[13],
    protein_cross_pro_isring_flag=[125],
    protein_cross_pro_isO_flag=[125],
    protein_cross_pro_isN_flag=[125],
    protein_cross_ligand=[13, 3],
    protein_cross_protein=[125, 3],
    protein_cross_distance=[13, 125],
    protein_link_e=[291, 2],
    protein_link_t=[291],
    protein_link_e_reverse=[291, 2],
    protein_link_t_reverse=[291],
    protein_coords_predict=[13, 3],
    protein_mask_protein_pos=[300],
    protein_fill_protein_pos=[300, 3],
    ligand_smiles='CNC(=O)c1scc2c1OCCO2',
    ligand_element=[13],
    ligand_pos=[13, 3],
    ligand_mol=<rdkit.Chem.rdchem.Mol object at 0x7f007714ef20>,
    ligand_rd_mol=<rdkit.Chem.rdchem.Mol object at 0x7f007714e660>,
    ligand_rd_pos=[13, 3],
    ligand_bond_index=[2, 28],
    ligand_bond_type=[28],
    ligand_center_of_mass=[3],
    ligand_atom_feature=[13, 8],
    ligand_hybridization=[13],
    ligand_fill_coords=[250, 3],
    ligand_fill_zmats=[250, 4],
    ligand_fill_masks=[250],
    ligand_fill_atom_order=[13],
    ligand_atom_isring=[13],
    ligand_atom_isO=[13],
    ligand_atom_isN=[13],
    ligand_nbh_list={
        0=[1],
        1=[2],
        2=[3],
        3=[3],
        4=[3],
        5=[3],
        6=[2],
        7=[2],
        8=[2],
        9=[2],
        10=[2],
        11=[2],
        12=[1],
    },
    complex_mol=<rdkit.Chem.rdchem.Mol object at 0x7f0077164a40>,
    protein_filename='posebustersv1/5S8I/5S8I_protein.pdb',
    ligand_filename='posebustersv1/5S8I/5S8I_ligand.sdf',
    affinity=0.0,
    name='5S8I',
    id='5S8I',
    protein_atom_feature=[125, 27],
    ligand_atom_feature_full=[13],
    ligand_bond_feature=[28, 5]
    )
    '''
    
    unimol_coords_distance_list = []
    rsmd_list2 = []
    #取第一个最好的
    for dt in test_set_list[0]:
        #print('dt:', dt)
        #exit()
        unimol_distance_list.append(dt.protein_cross_distance)
        A, B     = dt.protein_cross_ligand, dt.protein_cross_protein
        tmp_dis  = calculate_distance_matrix(A, B)
        true_distance_list.append(tmp_dis.detach().cpu().numpy())

        unimol_coords_distance = calculate_distance_matrix(dt.protein_coords_predict, B)
        unimol_coords_distance_list.append(unimol_coords_distance.detach().cpu().numpy())

    for i_dis, j_dis in zip(unimol_distance_list, true_distance_list):
        assert i_dis.shape == j_dis.shape
        rmsd_ = np.sqrt(np.sum((i_dis.reshape(-1) - j_dis.reshape(-1)) ** 2) / i_dis.reshape(-1).shape[0])
        rsmd_list.append(rmsd_)

    
    for i_dis, j_dis in zip(unimol_coords_distance_list, true_distance_list):
        assert i_dis.shape == j_dis.shape
        rmsd_ = np.sqrt(np.sum((i_dis.reshape(-1) - j_dis.reshape(-1)) ** 2) / i_dis.reshape(-1).shape[0]) #把距离矩阵变成向量，再算rmsd更适合
        rsmd_list2.append(rmsd_)

    #绘制分布图
    histplot_rmsd(rsmd_list, base_save_path = './', name = 'unimol_to_true_distance_rmsd')
    print('unimol_distance to true_distance mean_rsmd:',np.mean(rsmd_list)) #unimol_distance to true_distance mean_rsmd: 3.7892649648973666

    histplot_rmsd(rsmd_list2, base_save_path = './', name = 'unimol_coords_to_true_distance_rmsd')
    print('unimol_coords_distance to true_distance mean_rsmd:',np.mean(rsmd_list2)) # 2.8978240091030654

    #exit()
    
    '''
    print('去除None')
    train_set, val_set, test_set = [x for x in train_set if x is not None], [x for x in val_set if x is not None], [x for x in test_set if x is not None]

    logger.info(f'Train: {len(train_set)}')
    logger.info(f'Valid: {len(val_set)}')
    logger.info(f'Test: {len(test_set)}')

    #把测试的数据所在的目录单独保存一份，根据test_set所保存的文件名，可以获取路径
    
    
    raw_path = ckpt['config'].data.path #../{GP.data_type}/data/pdbbind2020_r10
    #print('raw_path:', raw_path) #raw_path: ../{GP.data_type}/data/pdbbind2020_r10
    new_path = os.path.dirname(raw_path) + '/pdb2020_test'
    os.makedirs(new_path, exist_ok=True) #../{GP.data_type}/data

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
    
    #保存对应的复合物名字
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
    
    #保存对应的复合物名字
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
    
    #保存对应的复合物名字
    with open(os.path.dirname(new_path) + '/pdb2020_valide_name.txt', 'w') as f:
        for i in name_list:
            f.write(i + '\n')

    

    skip_it = np.array([472]) #某些图很大，直接跳过,可能需要统计数据集，将原子数量过多的，去掉，否则equiformer难以运行


    #统计每一个复合物的原子数量

    datas = [train_set, val_set, test_set]

    protein_num = []
    ligand_num  = []
    all_num     = []

    for nm in datas:
        for dt in nm:
            protein_num.append(len(dt.protein_element))
            ligand_num.append(len(dt.ligand_element))
            all_num.append(len(dt.protein_element) + len(dt.ligand_element))

    

    # 使用 Counter 类统计列表中每个元素的出现频率
    counter = Counter(all_num)

    # 使用 sorted() 函数对统计结果按照原子数量从大到小排序
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
    print('atom num > 1000 graph num:', len(np_all_num[index]))#68, 采样8, 或者直接过滤掉

    index = np_all_num > 800
    print('atom num > 800 graph num:', len(np_all_num[index])) #231, 采样16

    index = np_all_num > 700
    print('atom num > 700 graph num:', len(np_all_num[index])) #611, 采样16

    index = np_all_num > 600
    print('atom num > 600 graph num:', len(np_all_num[index])) #2260, 采样16

    #atom_num_list = [600, 800, 1000]
    
    # 示例数据
    categories = list(sorted_counter.keys())
    values     = list(sorted_counter.values())

    # 创建柱状图
    plt.bar(categories, values)

    # 添加标题和标签
    plt.title('atom num bar')
    plt.xlabel('Categories')
    plt.ylabel('Values')

    # 显示图形
    #plt.show()
    
    #保存图片
    plt.savefig('atom_num_bar.png')
    print('all_num')

    # 创建频率直方图
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
    

    
    model.cuda(local_rank)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True, output_device=local_rank, broadcast_buffers=False) 
    #model = torch.compile(model, mode='reduce-overhead', dynamic=True, fullgraph=True, backend='inductor') #torch.compile在2.0后才能用
    try:
        model = torch.compile(model, mode='reduce-overhead', dynamic=True, fullgraph=True, backend='inductor') #torch.compile在2.0后才能用
    except Exception as e:
        print('not use pytorch2.0 compile, skip')
        pass
    
    model = model.module
    

    print(f'# trainable parameters: {misc.count_parameters(model) / 1e6:.4f} M') #只统计带有梯度更新的，不要没参与梯度更新的
    print(f'# not trainable parameters: {misc.count_non_grad_parameters(model) / 1e6:.4f} M') #只统计带有梯度更新的，不要没参与梯度更新的

    consistency_sampling_and_editing = ConsistencySamplingAndEditing(
                    sigma_min = GP.sigma_min, # minimum std of noise
                    sigma_data = GP.sigma_data, # std of the data
                    )

    
    if value_ckpt is not None: #默认是None，仅仅valuenet模型有效
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


    if dist.get_rank() == 0:
        if args.out_dir: #默认是None
            dir_path = os.path.join(args.out_dir, os.path.basename(result_path))
        else:
            dir_path = result_path
        
        
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)   # 删除整个目录及其子文件
            print(f"已删除目录: {dir_path}")
            #print('gen file exist, skip')
            #continue
        
        
        os.makedirs(dir_path, exist_ok=True)
        
    dist.barrier()
    
    for i in tqdm(list(range(len(test_set)))[args.si: args.ei]): 
        #data = test_set[args.data_id] #只取一个蛋白靶点，如果想要更多，则需要写个循环或者多次生成，每次指定不同的复合物
        data = copy.deepcopy(test_set[i])
        
        ligand_filename =data.ligand_filename
        complex_name = ligand_filename.split('/')[1]
        
        
        if args.out_dir:
            bs_dir = os.path.join(args.out_dir, os.path.basename(result_path), f'{complex_name}', f'step{GP.final_timesteps-1}')
        else:
            bs_dir = os.path.join(result_path, f'{complex_name}', f'step{GP.final_timesteps-1}')
        
        exit_ligand_file = os.path.join(bs_dir, f'gen_ligand_{complex_name}.sdf')
        
        #print('exit_ligand_file:', exit_ligand_file)
        #if is_file_exist_and_not_empty(exit_ligand_file):
            #print('gen file exist, skip')
            #continue
        

            
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
                #保存配体和蛋白sdf和pdb
                #protein_filename='BSD_ASPTE_1_130_0/2z3h_A_rec_1wn6_bst_lig_tt_docked_3_pocket10.pdb',
                #ligand_filename='BSD_ASPTE_1_130_0/2z3h_A_rec_1wn6_bst_lig_tt_docked_3.sdf',

                #protein_filename='refined-set/5l8c/5l8c_pocket10.pdb',
                #ligand_filename='refined-set/5l8c/5l8c_ligand.sdf',

                protein_filename=data.protein_filename
                ligand_filename =data.ligand_filename
                complex_name = ligand_filename.split('/')[1] #至少pdbbind数据是这样，crossdock要另外说
                #print('ligand_filename:', ligand_filename)


                #target_dir = os.path.join(result_path, f'result_{i}') #数字编号
                if args.out_dir:
                    target_dir = os.path.join(args.out_dir, os.path.basename(result_path), f'{complex_name}')
                else:
                    target_dir = os.path.join(result_path, f'{complex_name}')
                
                data_path = os.path.join(ckpt['config'].data.path, 'test_set')
                
                
                s_dir = os.path.dirname(ckpt['config'].data.path)
                #print('s_dir:', s_dir) #../{GP.data_type}/data/posebustersv1
                if 'v2020-other-PL' in ligand_filename: ##使用pdb2020训练时采用的测试方法
                    #print('use v2020-other-PL test')
                    dir_name = '/'.join(ligand_filename.split('/')[:-1])
                    source_dir = os.path.join(s_dir, 'pdbbind2020_r10', dir_name)
                elif 'refined-set' in ligand_filename: ##使用pdb2020训练时采用的测试方法
                    #print('use refined-set test')
                    dir_name = '/'.join(ligand_filename.split('/')[:-1])
                    source_dir = os.path.join(s_dir, 'pdbbind2020_r10', dir_name)
                elif data_flag == 'new_test':
                    dir_name = ligand_filename.split('/')[1]
                    source_dir = os.path.join(s_dir, args.data_name, dir_name)
                elif 'pdbbind2020_r10' in ligand_filename:
                    dir_name = ligand_filename.split('/')[1]
                    source_dir = os.path.join(s_dir, args.data_name, dir_name)
                    #print('source_dir:', source_dir)
                else:
                    #print('use test_set test')
                    dir_name = ligand_filename.split('/')[0]
                    source_dir = os.path.join(s_dir, 'test_set', dir_name)

                
                #target_dir = os.path.join(result_path, f'result_{args.data_id}') os.path.dirname(file_path)
                #if os.path.exists(target_dir):
                    # If it exists, remove it
                    #shutil.rmtree(target_dir)
                #shutil.copytree(source_dir, target_dir)

                os.makedirs(target_dir, exist_ok=True)
                #torch.save(result, os.path.join(target_dir, f'{i}.pt'))

                pred_ligand_pos_ = copy.deepcopy(pred_pos)


                if 'v2020-other-PL' in ligand_filename:
                    origin_ligand_file = f'{s_dir}/pdbbind2020_r10/{ligand_filename}'
                elif 'refined-set' in ligand_filename:
                    origin_ligand_file = f'{s_dir}/pdbbind2020_r10/{ligand_filename}'
                elif data_flag == 'new_test':
                    origin_ligand_file = f'{s_dir}/{ligand_filename}'
                elif 'pdbbind2020_r10' in ligand_filename:
                    origin_ligand_file = f'{s_dir}/{ligand_filename}'
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
                    #保留轨迹意义不大了，所以只要最后一步即可
                    if GP.multi_step:
                        #保留轨迹
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

                    target_dir = os.path.join(base_target, f'step{step}')
                    #print('target_dir:', target_dir)
                    os.makedirs(target_dir, exist_ok=True)
                    if GP.glide_vina:
                        try:
                            new_ligand_file1 = os.path.join(target_dir, f'origin_ligand_{complex_name}.sdf')
                            new_ligand_file2 = os.path.join(target_dir, f'glide_gen_ligand_{complex_name}.sdf')
                            f1 = open(new_ligand_file1, "a")
                            f2 = open(new_ligand_file2, "a")
                            supp1 = Chem.SDWriter(f1)
                            supp2 = Chem.SDWriter(f2)
                            #new_mol1 = Chem.SDMolSupplier(os.path.join(os.path.dirname(target_dir), f'origin_ligand_{complex_name}.sdf'))[0]
                            #new_mol2_list = Chem.SDMolSupplier(os.path.join(os.path.dirname(target_dir), f'{complex_name}_ligand.sdf'))
                            new_mol1 = Chem.SDMolSupplier(os.path.join(source_dir, f'origin_ligand_{complex_name}.sdf'))[0]
                            new_mol2_list = Chem.SDMolSupplier(os.path.join(source_dir, f'{complex_name}_ligand.sdf'))
                            
                            mol1 = Chem.RemoveHs(new_mol1)
                            supp1.write(mol1)
                            supp1.close()

                            for new_mol2 in new_mol2_list:
                                mol2 = Chem.RemoveHs(new_mol2)
                                supp2.write(mol2) 
                            supp2.close()#需要手动关闭

                        except Exception as e:
                            continue

                    else:
                        try:
                            new_ligand_file = os.path.join(target_dir, f'origin_ligand_{complex_name}.sdf')
                            f1 = open(new_ligand_file, "a")
                            supp=Chem.SDWriter(f1)
                            #new_mol = Chem.SDMolSupplier(os.path.join(os.path.dirname(target_dir), f'{complex_name}_ligand.sdf'))[0]
                            new_mol = Chem.SDMolSupplier(os.path.join(source_dir, f'{complex_name}_ligand.sdf'))[0]
                            mol2 = Chem.RemoveHs(new_mol)
                            supp.write(mol2)
                            supp.close()    #需要手动关闭

                        except Exception as e:
                            continue


                    new_ligand_file = os.path.join(target_dir, f'gen_ligand_{complex_name}.sdf')
                    conformer = origin_mol.GetConformer() #conformer.GetAtomPosition(atom_idx)
                    origin_pos  = origin_mol.GetConformer().GetPositions() #GetPositions(),'s'表示获取所有分子
                    origin_pos2 = data.ligand_pos #这个数据是保真的


                    ##print('origin_pos[:2]:', origin_pos[:2])
                    ##print('origin_pos2[:2]:', origin_pos2[:2])
                    f1 = open(new_ligand_file, "a")
                    supp=Chem.SDWriter(f1)

                    for j in range(len(pred_pos_traj)):
                        try:
                            new_mol = Change_Mol_D3coord(origin_mol, copy.deepcopy(pred_pos_traj[j][i]))
                        except Exception:
                            continue
                        mol2 = Chem.RemoveHs(new_mol)
                        try:
                            supp.write(mol2)
                        except Exception:
                            continue
                    supp.close()    #需要手动关闭



                    #写入构建连接表使用的coords_predict到sdf
                    coords_predict_file = os.path.join(target_dir, f'connection_coords{complex_name}.sdf')
                    f1 = open(coords_predict_file, "a")
                    supp=Chem.SDWriter(f1)
                    #print('coords_predict:', coords_predict.shape)
                    #print(pred_ligand_pos.shape) #(13, 3)
                    #print(coords_predict.shape)  #(130, 3)
                    #for j in range(len(pred_pos_traj)):
                    #构建连接表的都是同一个距离矩阵，所以保存一个即可
                    try:
                        new_mol_ = Change_Mol_D3coord(origin_mol, copy.deepcopy(coords_predict[0].reshape([-1, pred_ligand_pos.shape[0], 3])[0]))
                        mol2_ = Chem.RemoveHs(new_mol_)
                        supp.write(mol2_)
                    except Exception as e:
                        print(e)
                    supp.close()    #需要手动关闭

                    # 复制蛋白
                    shutil.copy2(os.path.join(source_dir, f'{complex_name}_protein.pdb'), os.path.join(target_dir, f'{complex_name}_protein.pdb'))


                    # 写入 XYZ 文件
                    '''
                    3
                    Water molecule
                    O 0.0 0.0 0.0
                    H 0.757 0.586 0.0
                    H -0.757 0.586 0.0

                    '''

                    
                    #xyz文件的保存是不是有问题？第一个效果可以，之后杂乱,一个xyz文件只能保存一个分子，所以如果要求保存多个构象，则需要分别存储
                    symbols = [atom.GetSymbol() for atom in origin_mol.GetAtoms()]
                    filename = os.path.join(target_dir, f'gen_ligand_{complex_name}.xyz')
                    num_atoms = pred_ligand_pos[0].shape[0]
                    with open(filename, 'a') as xyz_file:
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
                    
            

            if GP.multi_step:
                step_num_list = list(range(len(pred_pos_traj[0])))[:]
                num = 2
                data_name = complex_name
                
                concat_conformation(base_target, step_num_list, num, data_name)
                







if __name__ == '__main__':
    #exclude_keys = ['protein_cross_distance']
    collate_exclude_keys = ['ligand_nbh_list']
    #采样生成
    for step in [1, 5, 10, 15, 20, 25, 50, 100][1:2]: #不能循环了，是什么情况？出现加载配置文件时编码错误
        '''
            CUDA_VISIBLE_DEVICES=0 python scripts/sample_diffusion.py --config ./configs/sampling.yml -i 0 --guide_mode pdbbind_random \
                --type_grad_weight 100 --pos_grad_weight 25 --result_path ./cd2020_pro_0_res
        '''
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', type=str, default='./configs/sampling.yml')
        parser.add_argument('-i', '--data_id', type=int, default=81) #数据开始的位置
        parser.add_argument('--device', type=str, default='cuda')
        parser.add_argument('--batch_size', type=int, default=None)
        parser.add_argument('--guide_mode', type=str, default='pdbbind_random', choices=['joint', 'pdbbind_random', 'vina', 'valuenet', 'wo'])  
        parser.add_argument('--type_grad_weight', type=float, default=0) #注意下，这里的权重如何使用的
        parser.add_argument('--pos_grad_weight', type=float, default=0)
        parser.add_argument('--result_path', type=str, default='./test_package') #分子生成的路径
        parser.add_argument('--log_name', type=str, default='')
        parser.add_argument('--data_flag', type=str, default='old_test', choices=['new_test', 'old_test'], help = 'use new or old data test') 
        parser.add_argument('--data_name', type=str, default='pdbbind2020')
        parser.add_argument('--sample_num', type=int, default=10000)
        parser.add_argument('--diffusion', type=str, default='cm', choices=['cm', 'ddpm'])
        parser.add_argument('--gnn', type=str, default='equiformer', choices=['equiformer', 'egnn', 'escn'])
        parser.add_argument('--rm_outdir', action='store_false') #默认是删除的，即如果不指定这个参数
        parser.add_argument('--conf_num', type=int, default=5)
        parser.add_argument('--test_name', type=str, default='')

        parser.add_argument('--si', type=int, default=0)
        parser.add_argument('--ei', type=int, default=500)
        
        parser.add_argument('--out_dir', type=str, default=None)
        parser.add_argument('--step', type=int, default=None) 
        
        parser.add_argument('--ckpt', type=str, default=None)

        args = parser.parse_args() #参数一定要放在循环内，这样保证每一步的采样的命令行参数一样
        
        
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"]) #进程数量, 对应所使用的GPU数量，而不是CPU的线程数量

        # 1) 初始化
        #torch.distributed.init_process_group(backend="gloo", init_method='env://', rank=local_rank, world_size=world_size)
        torch.distributed.init_process_group(backend="nccl", init_method='env://', rank=local_rank, world_size=world_size)
        
        # 2） 配置每个进程的gpu
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
        data_flag = args.data_flag        #posebusters数据集，则执行这个，否则设置为None或者不指定
        data_name = args.data_name     #注意，新的数据集要和标志位一起修改
        if data_flag == 'new_test':
            if data_name == ['posebustersv1', 'posebustersv2', 'posebusters_glide']:
                GP.max_atoms = 64 #根据不同的数据集，设置配体的最大原子数量

        data_path  = f'../{GP.data_type}/data/{data_name}/' #如果要使用新的数据集，把这里的"posebusters"给替换掉即可
        data_split = f'../{GP.data_type}/data/{data_name}/{data_name}_split.pt' 

        '''
        try:
            name_list = []
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
        output_dir = f'../ECDock_sample_dir/{args.data_name}_ecdock_{args.diffusion}_{args.gnn}_step{step}_{GP.interaction_stype}_limit{GP.cross_distance_cutoff}ai_{args.test_name}'
        sample_main(output_dir, data_flag, sample_num, data_path, data_split, data_name, args, name_list)

        
        #把模型复制一份
        '''
        os.makedirs(os.path.join(output_dir, 'model'), exist_ok=True)
        shutil.copyfile(args.config, os.path.join(output_dir, 'model', os.path.basename(args.config)))
        shutil.copytree('./models', os.path.join(output_dir, 'model', 'models'), dirs_exist_ok=True)  
        shutil.copytree('./KGDiff', os.path.join(output_dir, 'model', 'KGDiff'), dirs_exist_ok=True)
        shutil.copytree('./EcConf', os.path.join(output_dir, 'model', 'EcConf'), dirs_exist_ok=True)
        shutil.copytree('./configs', os.path.join(output_dir, 'model', 'configs'), dirs_exist_ok=True)
        shutil.copytree('./ocp', os.path.join(output_dir, 'model', 'ocp'), dirs_exist_ok=True)
        '''
    
    

    dist.barrier()
    

    if dist.get_rank() == 0:
        random_flag = False
        for step in [1, 5, 10, 15, 20, 25, 50, 100][1:2]:
            if args.step:
                step = args.step
            #评估rmsd
            model = 'ecdock'
            data_name = args.data_name  #pdb2020
            gnn  = args.gnn  #ecdock时，采用不同的神经网络, equiformer
            diffusion = args.diffusion #ecdock时，采用不同的扩散模型， CM/DDPM
            mode = '' #不用赋值
            if model == 'ecdock':    #posebusters_ecdock_cm_equiformer_step1
                model_name = f'{args.data_name}_ecdock_{args.diffusion}_{args.gnn}_step{step}_{GP.interaction_stype}_limit{GP.cross_distance_cutoff}ai_{args.test_name}' #这个目录下，可以存放是sdf也可以是pickle，路径别忘了改
                
                if args.out_dir:
                    file_path  = f'{args.out_dir}/{model_name}'
                else:
                    file_path  = f'../ECDock_sample_dir/{model_name}' #记得改名字
                step = step - 1


            name_list = []

            #posebustersv1_name_valid.txt
            with open(f'../{GP.data_type}/data/{data_name}/{data_name}_name.txt') as f:
                for i in f:
                    name_list.append(i.strip())

            #效果差的，复杂的分子，为了节约时间，我们平常评估这些即可
            poor_name_list = []
            '''
            with open(f'../{GP.data_type}/data/{data_name}/{data_name}_poor_name.txt') as f:
                for i in f:
                    poor_name_list.append(i.strip())
            '''
                
            #读取配体的sdf文件,truth_mol是一维list，gen_mol是一个2维度list，存放整个测试集的结果
            truth_mol, gen_mol, data_name_list = read_file(file_path, mode, flag = 'sdf', num = 10000  + 2, step = step, model = model, name_list = name_list, poor_name_list = poor_name_list)  #读取所有数据，并转化成rdkit mol对象, step值别忘了改
            print('truth_mol, gen_mol:', len(truth_mol), len(gen_mol))
            assert len(truth_mol) == len(gen_mol)
            

            #计算rmsd。从生成的40个分子的中随机选择1/3/5/10/40的，拿过来看rmsd成功率
            resault_dict = {}
            boxplot_data_list = [] #保留1,5,40结果用于绘制箱线图
            for num in [1, 3, 5, 10, 25, 40, args.conf_num][:]:
                data_dict = rmsds(truth_mol, gen_mol, num, data_name_list, random_flag) #对于每一条数据，随机挑选num个进行测试
                resault_dict[num] = ['rate, rate_min, rate_mean, rmsd_mean, rmsd_std, rsmd_mid, rmsd_max, rmsd_min:', data_dict['all']]
                if num in [1, 5, 10, 25, 40, args.conf_num]:
                    boxplot_data_list.append(data_dict['data'])

            #exit()
            print(json.dumps(resault_dict, indent=4))
            #保存字典为JSON文件
            #path = 'resault'
            path = file_path
            os.makedirs(path, exist_ok=True)

            file_name = f'{model_name}_evaluate_resault.json'
            with open(os.path.join(path, file_name), 'w') as file:
                json.dump(resault_dict, file, indent=4)
            

            #绘制箱线图
            save_path = os.path.join(path, f'{model_name}_boxplot.png')
            boxplot(boxplot_data_list, save_path, model_name)
        
    dist.destroy_process_group()
