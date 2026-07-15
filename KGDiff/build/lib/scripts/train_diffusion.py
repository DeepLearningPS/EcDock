import os
import sys
import argparse
import shutil
import numpy as np
import pandas as pd
import torch
import torch.utils.tensorboard
import seaborn as sns
# sns.set_theme(style="darkgrid")

import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score
from scipy import stats

from torch.nn.utils import clip_grad_norm_





# EcDock
import numpy as np 
from rdkit import Chem
from EcDock.graphs import * 
from EcDock.utils import *
from EcDock.model import *  #，torchDataLoaderpygDataLoader，，pyg
from EcDock.comparm import *





#from torch_geometric.data import DataLoader #
from torch_geometric.loader import DataLoader #torchDataLoader, ，, ，list

from torch_geometric.transforms import Compose
from torch_geometric.data import Data

from tqdm.auto import tqdm #
import sys
sys.path.append(os.path.abspath('./'))
import KGDiff.utils.misc as misc
import KGDiff.utils.train as utils_train
import KGDiff.utils.transforms as trans #

from KGDiff.datasets import get_dataset
from KGDiff.datasets.pl_data import FOLLOW_BATCH
from models.molopt_score_model import ScorePosNet3D
import logging
import pprint 

from ordered_set import OrderedSet
import copy
from rdkit import Chem
from rdkit.Chem import AllChem


import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import torch.multiprocessing as mp

import random

from collections import Counter
import matplotlib.pyplot as plt

from torch_geometric.data import Data

try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
except Exception:
    pass

import dill
#import triton #torch.complie triton


# 
#torch.distributed.init_process_group(backend='nccl')


np.set_printoptions(suppress=True, precision=4)
torch.set_printoptions(sci_mode=False, precision=4)

def get_auroc(y_true, y_pred, feat_mode):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    avg_auroc = 0.
    possible_classes = set(y_true)
    for c in possible_classes:
        auroc = roc_auc_score(y_true == c, y_pred[:, c])
        avg_auroc += auroc * np.sum(y_true == c)
        mapping = {
            'basic': trans.MAP_INDEX_TO_ATOM_TYPE_ONLY,
            'add_aromatic': trans.MAP_INDEX_TO_ATOM_TYPE_AROMATIC,
            'full': trans.MAP_INDEX_TO_ATOM_TYPE_FULL
        }
        logging.info(f'atom: {mapping[feat_mode][c]} \t auc roc: {auroc:.4f}')
    return avg_auroc / len(y_true)

def get_pearsonr(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return stats.pearsonr(y_true, y_pred)


def generate_3d_conformer_from_smiles(smiles):
    #  SMILES 
    ##print(smiles)
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    ##print(mol)
    flag = False

    # 
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    try:
        AllChem.UFFOptimizeMolecule(mol)
    except Exception as e:
        ##print(e)
        flag = True
        return mol, flag

    return mol, flag


def save_sdf(mol, output_sdf):
    #  SDF 
    writer = Chem.SDWriter(output_sdf)
    #  SDF 
    writer.write(mol)
    #  SDF 
    writer.close()
    #print(f"SDF ：{output_sdf}")


def add_knn_ligand_pos(data):
    knn_data = []
    fail_count = 0
    for dt in data:
        #rdkit3DKNN
        # 
        ##print('dt:', dt)
        mol, fail_flag = generate_3d_conformer_from_smiles(dt.ligand_smiles)

        if fail_flag == True:
            dt.knn_ligand_pos = dt.ligand_pos #rdkit，
            knn_data.append(dt)
            fail_count += 1
        else:
            conformer      = mol.GetConformer()
            knn_ligand_pos = conformer.GetPositions() #KNN
            knn_ligand_pos = torch.FloatTensor(knn_ligand_pos)
            centor = knn_ligand_pos.mean(dim = 0) - dt.ligand_pos.mean(dim = 0)
            dt.knn_ligand_pos = knn_ligand_pos - centor
            knn_data.append(dt)
    
    return knn_data, fail_count


def set_seed(seed):
    torch.manual_seed(seed)  #  PyTorch 
    torch.cuda.manual_seed_all(seed)  #  GPU 
    np.random.seed(seed)  #  NumPy 
    random.seed(seed)  #  Python 
    torch.backends.cudnn.deterministic = True  #  CuDNN 
    torch.backends.cudnn.benchmark = True


# collate
def custom_collate(batch):
    # （ 'z'）
    exclude_keys = ['protein_cross_distance']

    # 
    batch_data = {}
    
    keys = batch[0].keys #pyg2.1.0，，
    # 
    for key in keys:
        if key in exclude_keys:
            # ，
            batch_data[key] = [getattr(data, key) for data in batch]
        else:
            # ，
            batch_data[key] = torch.cat([getattr(data, key) for data in batch], dim=0)

    return batch_data




def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/training.yml')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--logdir', type=str, default='./logs_diffusion')
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--value_only', action='store_true')
    parser.add_argument('--train_report_iter', type=int, default=200)
    parser.add_argument('--load_model_path', type=str, default=None)
    parser.add_argument('--log_name', type=str, default='') #
    try:
        parser.add_argument("--local-rank", type=int,  help='rank in current node') #，torch, local-rank，local_rank
    except Exception:
        parser.add_argument("--local_rank", type=int,  help='rank in current node')

    # 
    seed = 2024
    set_seed(seed)

    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"]) #, GPU，CPU

    # 1) 
    torch.distributed.init_process_group(backend="nccl", init_method='env://', rank=local_rank, world_size=world_size)
    
    # 2） gpu
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    args.device = device
    args.rank = local_rank

    print('local_rank:', local_rank)
    print('device:', device)
    


    # load ckpt
    if args.ckpt:
        #print(f'loading {args.ckpt}...')
        ckpt = torch.load(args.ckpt, map_location=args.device)
        config = ckpt['config']
        config = misc.load_config(args.config) #，
    else:
        # Load configs
        config = misc.load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    misc.seed_all(config.train.seed)

    # Logging
    log_dir = misc.get_new_log_dir(args.logdir, prefix= args.log_name + config_name, tag=args.tag)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    vis_dir = os.path.join(log_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)
    logger = misc.get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    print(args)
    print(config)


    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    shutil.copytree('./models', os.path.join(log_dir, 'models'), dirs_exist_ok=True)  
    shutil.copytree('./KGDiff', os.path.join(log_dir, 'KGDiff'), dirs_exist_ok=True)
    shutil.copytree('./EcDock', os.path.join(log_dir, 'EcDock'), dirs_exist_ok=True)
    shutil.copytree('./configs', os.path.join(log_dir, 'configs'), dirs_exist_ok=True)
    shutil.copytree('./ocp', os.path.join(log_dir, 'ocp'), dirs_exist_ok=True)


    #
    if torch.cuda.get_device_properties(local_rank).total_memory / 1000**3 >= 38:
        config.train.batch_size         = 16
        config.equiformer.batch_size    = 6
        config.escn.batch_size          = 4
    elif torch.cuda.get_device_properties(local_rank).total_memory / 1000**3 >= 24:
        config.train.batch_size         = 8
        config.equiformer.batch_size    = 2
        config.escn.batch_size          = 2
    else:
        config.train.batch_size         = 4
        config.equiformer.batch_size    = 2
        config.escn.batch_size          = 1



    #
    #EGNNEquiformer，2，16，8
    if torch.cuda.get_device_properties(local_rank).total_memory / 1000**3 >= 38:
        grad_num = 1
    elif torch.cuda.get_device_properties(local_rank).total_memory / 1000**3 >= 24:
        grad_num = 1
    else:
        grad_num = 1 #，，. consistency，，

    #equiformer
    if config.model.model_mode == 'equiformer':
        lmax_list = config.equiformer.lmax_list
        num_resolutions = len(lmax_list)
        num_coefficients = 0
        for i in range(num_resolutions):
            num_coefficients = num_coefficients + int((lmax_list[i] + 1) ** 2) #（），

        print('num_coefficients = 49 ?:', num_coefficients)
        config.train.batch_size = config.equiformer.batch_size
        config.model.hidden_dim = config.equiformer.attn_hidden_channels * num_coefficients # num_coefficients49


    #escn
    if config.model.model_mode == 'escn':
        lmax_list = config.escn.lmax_list
        num_resolutions = len(lmax_list)
        num_coefficients = 0
        for i in range(num_resolutions):
            num_coefficients = num_coefficients + int((lmax_list[i] + 1) ** 2) #（），

        print('num_coefficients = 49 ?:', num_coefficients)
        config.train.batch_size = config.escn.batch_size
        config.model.hidden_dim = config.escn.sphere_channels * num_coefficients # num_coefficients49

    
    # Transforms
    protein_featurizer = trans.FeaturizeProteinAtom() #，
    ligand_featurizer = trans.FeaturizeLigandAtom(config.data.transform.ligand_atom_mode) #
    transform_list = [
        protein_featurizer,
        ligand_featurizer,
        trans.FeaturizeLigandBond(), #,，?，ligand_bond_feature
        trans.NormalizeVina(config.data.name) #vine
    ]
    
    if config.data.transform.random_rot:
        transform_list.append(trans.RandomRotation())
    transform = Compose(transform_list)

    # Datasets and loaders
    print('Loading dataset...')
    #raise Exception('stop0')

    #dataset，subdets、、
    dataset, subsets = get_dataset(
        config=config.data,
        transform=transform,
    )
    #print('data_name:',config.data.name)

    subset_train = []

    if config.data.name == 'pl': # 
        train_set, val_set, test_set = subsets['train'], subsets['test'], []
    elif config.data.name == 'pdbbind':
        train_set, val_set, test_set = list(subsets['train'])[:], list(subsets['valid'])[:], list(subsets['test'])[:]

        train_set, val_set, test_set = [x for x in train_set if x is not None], [x for x in val_set if x is not None], [x for x in test_set if x is not None]


        print(f'Training: {len(train_set)} Validation: {len(val_set)} Test: {len(test_set)}')
        #exit()

        data_dict = defaultdict(lambda: defaultdict(lambda: []))
        #，，
        #pocket_fn = 'v2020-other-PL/4po7/4po7_pocket10.pdb'
        #ligand_fn = 'v2020-other-PL/4po7/4po7_ligand.sdf'

        for n, datas in zip(['train', 'val', 'test'], [train_set, val_set, test_set]):
            for dt in datas:
                pocket_fn    = dt.protein_filename 
                ligand_fn    = dt.ligand_filename
                complex_name = os.path.basename(ligand_fn).split('_')[0]  
                #print('complex_name:', complex_name)
                #data_dict[n].append([complex_name, pocket_fn, ligand_fn])
                data_dict[n]['name'].append(complex_name)
                data_dict[n]['protein_file'].append(pocket_fn)
                data_dict[n]['ligand_file'].append(ligand_fn)


        with open('pdbbind2020_dataname.pickle', 'wb')as f:
            dill.dump(data_dict, f)

        #exit()
        #, [14000,7000,3500,2000].
        #，

        if torch.cuda.get_device_properties(local_rank).total_memory / 1000**3 >= 38:
            lens = 14000
        elif torch.cuda.get_device_properties(local_rank).total_memory / 1000**3 >= 24:
            lens = 7000
        else:
            lens = 3500

        args.train_set_num  = len(train_set)
        args.val_set_num    = len(val_set)
        args.test_set_num   = len(test_set)

        sub_num = math.ceil(len(train_set) / lens)
        for i in range(sub_num):
            subset_train.append(train_set[i*lens: (i+1)*lens])

        #train_set, val_set, test_set = subsets['train'], subsets['val'], subsets['test']
        #train_set, val_set, test_set = list(subsets['train'])[:100], list(subsets['train'])[:10], list(subsets['train'])[:10]

        #，，，, ？
        #GPU，CPU，GPU，，numpy，list，CPU，
        #GPU tensor，，
        
    else:
        raise ValueError
    

    print(f'Training: {len(train_set)} Validation: {len(val_set)} Test: {len(test_set)}')

    #collate_exclude_keys = ['ligand_nbh_list']

    if local_rank == 0:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_set, shuffle = False, num_replicas=world_size, rank=local_rank)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_set, shuffle = False, num_replicas=world_size, rank=local_rank)

        val_loader = DataLoader(val_set, config.train.batch_size, shuffle=False, num_workers=2, 
                                #collate_fn=custom_collate, 
                                #exclude_keys = exclude_keys,
                                follow_batch=FOLLOW_BATCH, exclude_keys=collate_exclude_keys, sampler=val_sampler, pin_memory=True, prefetch_factor=2)
        test_loader = DataLoader(test_set, config.train.batch_size, shuffle=False, num_workers=2, 
                                #collate_fn=custom_collate, 
                                #exclude_keys = exclude_keys,
                                follow_batch=FOLLOW_BATCH, exclude_keys=collate_exclude_keys, sampler=test_sampler, pin_memory=True, prefetch_factor=2)



    # Model
    print('Building model...')

    model = ScorePosNet3D(
        config.model,
        protein_atom_feature_dim=protein_featurizer.feature_dim, #27，+
        ligand_atom_feature_dim=ligand_featurizer.feature_dim,  #
        equiformer_args = config.equiformer,
        escn_args = config.escn,
    )
    

    ema_model = ScorePosNet3D(
        config.model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim,
        equiformer_args = config.equiformer,
        escn_args = config.escn,
    )


    consistency_training = ConsistencyTraining(
        sigma_min=GP.sigma_min,
        sigma_max=GP.sigma_max,
        sigma_data=GP.sigma_data,
        rho=GP.rho,
        initial_timesteps=GP.initial_timesteps,
        final_timesteps=GP.final_timesteps
        )
    

    if config.model.diffusion_mode == 'DDPM':
        print('DDPM')
    elif config.model.diffusion_mode == 'CM':
        print('Consistency Model')

    model.cuda(local_rank)
    ema_model.cuda(local_rank)

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    ema_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(ema_model)

    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True, output_device=local_rank, broadcast_buffers=False) 
    ema_model = nn.parallel.DistributedDataParallel(ema_model, device_ids=[local_rank], find_unused_parameters=True, output_device=local_rank, broadcast_buffers=False) 
    #find_unused_parameters=True ，GPU，，
    #backend: ['cudagraphs', 'inductor', 'onnxrt', 'openxla', 'openxla_eval', 'tvm']
    #mode: default, reduce-overhead, max-autotune

    try:
        model = torch.compile(model, mode='max-autotune', dynamic=True, fullgraph=True, backend='inductor') #torch.compile2.0
        ema_model = torch.compile(ema_model, mode='max-autotune', dynamic=True, fullgraph=True, backend='inductor') #torch.compile2.0
    except Exception as e:
        print('not use pytorch2.0 compile, skip')
        pass

    model = model.module
    ema_model = ema_model.module

    

    # #print(model)
    print(f'protein feature dim: {protein_featurizer.feature_dim} ligand feature dim: {ligand_featurizer.feature_dim}')
    logger.info(f'# trainable parameters: {misc.count_parameters(model) / 1e6:.4f} M') #，
    logger.info(f'# not trainable parameters: {misc.count_non_grad_parameters(model) / 1e6:.4f} M') #，

    # Optimizer and scheduler
    optimizer = utils_train.get_optimizer(config.train.optimizer, model)
    scheduler = utils_train.get_scheduler(config.train.scheduler, optimizer)

    start_it = 0
    if args.ckpt:
        model.load_state_dict(ckpt['model'])
        ema_model.load_state_dict(ckpt['ema_model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_it = ckpt['iteration']
    
    args.all_protein_max_atom_num = config.data.protein_max_atom_num
    args.all_ligand_max_atom_num  = config.data.ligand_max_atom_num



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

    '''
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


    # 
    plt.hist(all_num, bins=len(sorted_counter)//10, edgecolor='black', alpha=0.7)

    plt.savefig('atom_num_hit.png')

    #exit()
    '''



    def train(gpu, args, it, sub_id, train_loader, ckpt_path, batch_id):
        model.train()
        ema_model.train()

        #for num, (batch, b_protein_cross_distance) in tqdm(enumerate(train_loader), desc='Training'):
        
        for num, batch in enumerate(tqdm(train_loader, desc='Training')):
            #print('type(batch_):', type(batch_))
            #print('batch_ num:', len(batch_))
            #print('batch_:', batch_)
            #batch, b_protein_cross_distance = batch_
            #，17，？？，，17

            '''
            if max(batch.protein_element) > 17 or max(batch.ligand_element) > 17:
                print('batch.protein_element:', batch.protein_element)
                print('batch.ligand_element:', batch.ligand_element)
                print('max(batch.protein_element), max(batch.ligand_element):', max(batch.protein_element), max(batch.ligand_element))
                raise Exception(f'>17')
            else:
                continue
            '''
            
            #
            b_protein_pos=batch.protein_pos.cuda(local_rank, non_blocking=True)
            b_protein_v=batch.protein_atom_feature.float().cuda(local_rank, non_blocking=True)
            b_affinity=batch.affinity.float().cuda(local_rank, non_blocking=True)
            b_batch_protein=batch.protein_element_batch.cuda(local_rank, non_blocking=True)

            b_ligand_pos=batch.ligand_pos.cuda(local_rank, non_blocking=True)
            b_ligand_v=batch.ligand_atom_feature_full.cuda(local_rank, non_blocking=True)
            b_batch_ligand=batch.ligand_element_batch.cuda(local_rank, non_blocking=True)

            b_ligand_bond_index = batch.ligand_bond_index.cuda(local_rank, non_blocking=True) #[2, 582]
            b_ligand_bond_type  = batch.ligand_bond_type.cuda(local_rank, non_blocking=True)
            b_ligand_bond_type_batch = batch.ligand_bond_type_batch.cuda(local_rank, non_blocking=True)

            b_protein_element = batch.protein_element.cuda(local_rank, non_blocking=True)
            b_ligand_element  = batch.ligand_element.cuda(local_rank, non_blocking=True)

            b_ligand_fill_coords =  batch.ligand_fill_coords.cuda(local_rank, non_blocking=True)

            zmats = batch.ligand_fill_zmats.cuda(local_rank, non_blocking=True).view(-1, GP.max_atoms, 4)
            bzids=torch.arange(zmats.shape[0]).view(-1,1).tile((1,zmats.shape[1])).unsqueeze(-1).long().cuda(local_rank, non_blocking=True)
            zmats=torch.concat((bzids,zmats),axis=-1).cuda(local_rank, non_blocking=True).view(-1, 5)
            b_ligand_fill_zmats  =  zmats

            b_ligand_fill_masks  =  batch.ligand_fill_masks.cuda(local_rank, non_blocking=True)
            b_ligand_fill_atom_order    =  batch.ligand_fill_atom_order.cuda(local_rank, non_blocking=True)

            b_ligand_atom_isring, b_ligand_atom_isO, b_ligand_atom_isN = batch.ligand_atom_isring.cuda(local_rank, non_blocking=True), batch.ligand_atom_isO.cuda(local_rank, non_blocking=True), batch.ligand_atom_isN.cuda(local_rank, non_blocking=True)
            b_protein_atom_isring, b_protein_atom_isO, b_protein_atom_isN = batch.protein_atom_isring.cuda(local_rank, non_blocking=True), batch.protein_atom_isO.cuda(local_rank, non_blocking=True), batch.protein_atom_isN.cuda(local_rank, non_blocking=True)
        

            b_protein_cross_lig_isring_flag = batch.protein_cross_lig_isring_flag.cuda(local_rank, non_blocking=True)
            b_protein_cross_lig_isO_flag = batch.protein_cross_lig_isO_flag.cuda(local_rank, non_blocking=True)
            b_protein_cross_lig_isN_flag = batch.protein_cross_lig_isN_flag.cuda(local_rank, non_blocking=True)

            b_protein_cross_pro_isring_flag = batch.protein_cross_pro_isring_flag.cuda(local_rank, non_blocking=True)
            b_protein_cross_pro_isO_flag = batch.protein_cross_pro_isO_flag.cuda(local_rank, non_blocking=True)
            b_protein_cross_pro_isN_flag = batch.protein_cross_pro_isN_flag.cuda(local_rank, non_blocking=True)

            b_protein_cross_ligand    = batch.protein_cross_ligand.cuda(local_rank, non_blocking=True)
            b_protein_cross_protein   = batch.protein_cross_protein.cuda(local_rank, non_blocking=True)
            #print('type(batch.protein_cross_distance):', type(batch.protein_cross_distance)) #protein_cross_distance，pyg，
            #cross_distance，PyG，，set，
            #setlist，：[{tensor([1, 2, 3]), tensor([1, 2, 3])}, {tensor([4, 5]), tensor([4, 5]), tensor([4, 5])}]
            #

            b_protein_cross_distance = []
            if isinstance(batch.protein_cross_distance, list):
                for i in batch.protein_cross_distance: #protein_cross_distanceslist
                    #ii = torch.stack(list(i), dim = 0) #list，，
                    #b_protein_cross_distance.append(ii.cuda(local_rank, non_blocking=True))
                    tg = torch.from_numpy(i).cuda(local_rank, non_blocking=True)
                    b_protein_cross_distance.append(tg)
    
            else:
                b_protein_cross_distance.append(batch.protein_cross_distance.cuda(local_rank, non_blocking=True))
                

            b_cross_bond_index = batch.protein_link_e.T.cuda(local_rank, non_blocking=True)
            b_cross_bond_type = batch.protein_link_t.cuda(local_rank, non_blocking=True)
            b_cross_bond_index_reverse = batch.protein_link_e_reverse.T.cuda(local_rank, non_blocking=True) 
            b_cross_bond_type_reverse = batch.protein_link_t_reverse.cuda(local_rank, non_blocking=True)






            for _ in range(1): #，2
                #batch = batch.cuda(local_rank, non_blocking=False) #batchGPU，，GPU，model(data.cuda())

                # 
                #torch.cuda.synchronize()
                ##print('train_iterator:', train_iterator)
                #batch = next(train_iterator).cuda(local_rank)

                #batch_size = max(batch.protein_element_batch) + 1
                #print('batch_size:', batch_size)
                #batch = next(train_iterator)
                ##print('batch:', batch)
                #exit()
                #exit()
                #？
                #protein_noise = torch.randn_like(batch.protein_pos) * config.train.pos_noise_std
                #gt_protein_pos = batch.protein_pos + protein_noise

                step_loss = defaultdict(list)
                for step in np.array(GP.steps_list) - 1: #
                    st = time.perf_counter()
                    for i in range(grad_num): #
                        #print('batch:', i)
                        #print('batch_size:', batch_size)
                        #？
                        #protein_noise = torch.randn_like(batch.protein_pos) * config.train.pos_noise_std #，
                        #gt_protein_pos = batch.protein_pos + protein_noise
                        #gt_protein_pos = batch.protein_pos

                    #，
                    #with torch.autocast(device_type='cuda'): #
                    #with torch.cuda.amp.autocast():
                        #GPU，GPU,GPU
                        if config.model.diffusion_mode == 'CM':
                            results = consistency_training(
                                #sigma_min=GP.sigma_min,
                                #sigma_max=GP.sigma_max,
                                #rho=GP.rho,
                                #sigma_data=GP.sigma_data,
                                #initial_timesteps=GP.initial_timesteps,
                                #final_timesteps=GP.final_timesteps,
                                online_model=model, 
                                ema_model=ema_model, 
                                current_training_step=step,
                                total_training_steps=GP.final_timesteps,
                                
                                args=args, 
                                config=config.model, 
                                protein_atom_feature_dim=protein_featurizer.feature_dim,
                                ligand_atom_feature_dim=ligand_featurizer.feature_dim,

                                protein_pos=b_protein_pos,
                                protein_v=b_protein_v,
                                affinity=b_affinity,
                                batch_protein=b_batch_protein,

                                ligand_pos=b_ligand_pos,
                                ligand_v=b_ligand_v,
                                batch_ligand=b_batch_ligand,

                                ligand_bond_index = b_ligand_bond_index, #[2, 582]
                                ligand_bond_type  = b_ligand_bond_type,
                                ligand_bond_type_batch = b_ligand_bond_type_batch,

                                protein_element = b_protein_element,
                                ligand_element  = b_ligand_element,

                                ligand_mol = batch.ligand_mol,

                                ligand_fill_coords =  b_ligand_fill_coords,
                                ligand_fill_zmats  =  b_ligand_fill_zmats,
                                ligand_fill_masks  =  b_ligand_fill_masks,
                                ligand_fill_atom_order = b_ligand_fill_atom_order,

                                ligand_atom_isring  = b_ligand_atom_isring,
                                ligand_atom_isO     = b_ligand_atom_isO,
                                ligand_atom_isN     = b_ligand_atom_isN,

                                protein_atom_isring = b_protein_atom_isring,
                                protein_atom_isO    = b_protein_atom_isO,
                                protein_atom_isN    = b_protein_atom_isN,

                                cross_lig_isring_flag = b_protein_cross_lig_isring_flag,
                                cross_lig_isO_flag = b_protein_cross_lig_isO_flag,
                                cross_lig_isN_flag = b_protein_cross_lig_isN_flag,

                                cross_pro_isring_flag = b_protein_cross_pro_isring_flag,
                                cross_pro_isO_flag = b_protein_cross_pro_isO_flag,
                                cross_pro_isN_flag = b_protein_cross_pro_isN_flag,

                                cross_ligand    = b_protein_cross_ligand,
                                cross_protein   = b_protein_cross_protein,
                                cross_distance  = b_protein_cross_distance,

                                    
                                cross_bond_index = b_cross_bond_index,
                                cross_bond_type = b_cross_bond_type, 
                                cross_bond_index_reverse = b_cross_bond_index_reverse, 
                                cross_bond_type_reverse = b_cross_bond_type_reverse,



                                )
                        
                        elif config.model.diffusion_mode == 'DDPM':
                            results = model.get_diffusion_loss(
                                args=args, 
                                config=config.model, 
                                protein_atom_feature_dim=protein_featurizer.feature_dim,
                                ligand_atom_feature_dim=ligand_featurizer.feature_dim,

                                protein_pos=b_protein_pos,
                                protein_v=b_protein_v,
                                affinity=b_affinity,
                                batch_protein=b_batch_protein,

                                ligand_pos=b_ligand_pos,
                                ligand_v=b_ligand_v,
                                batch_ligand=b_batch_ligand,

                                ligand_bond_index = b_ligand_bond_index, #[2, 582]
                                ligand_bond_type  = b_ligand_bond_type,
                                ligand_bond_type_batch = b_ligand_bond_type_batch,

                                protein_element = b_protein_element,
                                ligand_element  = b_ligand_element,

                                ligand_mol = batch.ligand_mol,

                                ligand_fill_coords =  b_ligand_fill_coords,
                                ligand_fill_zmats  =  b_ligand_fill_zmats,
                                ligand_fill_masks  =  b_ligand_fill_masks,
                                ligand_fill_atom_order = b_ligand_fill_atom_order,

                                ligand_atom_isring  = b_ligand_atom_isring,
                                ligand_atom_isO     = b_ligand_atom_isO,
                                ligand_atom_isN     = b_ligand_atom_isN,

                                protein_atom_isring = b_protein_atom_isring,
                                protein_atom_isO    = b_protein_atom_isO,
                                protein_atom_isN    = b_protein_atom_isN,


                                cross_lig_isring_flag = b_protein_cross_lig_isring_flag,
                                cross_lig_isO_flag = b_protein_cross_lig_isO_flag,
                                cross_lig_isN_flag = b_protein_cross_lig_isN_flag,

                                cross_pro_isring_flag = b_protein_cross_pro_isring_flag,
                                cross_pro_isO_flag = b_protein_cross_pro_isO_flag,
                                cross_pro_isN_flag = b_protein_cross_pro_isN_flag,

                                cross_ligand    = b_protein_cross_ligand,
                                cross_protein   = b_protein_cross_protein,
                                cross_distance  = b_protein_cross_distance,

                                
                                cross_bond_index = b_cross_bond_index,
                                cross_bond_type = b_cross_bond_type, 
                                cross_bond_index_reverse = b_cross_bond_index_reverse, 
                                cross_bond_type_reverse = b_cross_bond_type_reverse,
                            )
            

                        if args.value_only:
                            results['loss'] = results['loss_exp']
                            
                        loss, loss_pos, loss_v, loss_exp, loss_dismat, loss_bond, loss_angle, loss_dihedral, rmsd = results['loss'], results['loss_pos'], results['loss_v'],\
                            results['loss_exp'], results['loss_dismat'], results['loss_bond'], results['loss_angle'], results['loss_dihedral'], results['rmsd'],
                        loss = loss / grad_num #n_acc_batch == 1。，，2batch，，2。 loss_angle, loss_dihedral
                        #，，，。2，1/2， 
                        step_loss['loss'].append(loss.item())
                        step_loss['loss_pos'].append(loss_pos.item() / grad_num)
                        step_loss['loss_v'].append(loss_v.item() / grad_num)
                        step_loss['loss_exp'].append(loss_exp.item() / grad_num)
                        step_loss['loss_dismat'].append(loss_dismat.item() / grad_num) #，
                        step_loss['loss_bond'].append(loss_bond.item() / grad_num)
                        step_loss['loss_angle'].append(loss_angle.item() / grad_num)
                        step_loss['loss_dihedral'].append(loss_dihedral.item() / grad_num)
                        step_loss['rmsd'].append(rmsd.item() / grad_num)

                        ##print('loss:', loss)
                        #exit()
                        loss.backward()

                    
                    #，，. Consistency，，
                    #loss.backward()

                    #，，，
                    orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)

                    #，，，
                    optimizer.step()

                    # ，，
                    optimizer.zero_grad(set_to_none=True)  #loss.backward()，optimizer.step()


                    # ，
                    #torch.cuda.empty_cache()

                    #ema，
                    #timesteps_schedule
                    num_timesteps=timesteps_schedule(step,GP.final_timesteps,initial_timesteps=GP.initial_timesteps,final_timesteps=GP.final_timesteps)
                    #num_timesteps=improved_timesteps_schedule(step,GP.final_timesteps,initial_timesteps=GP.initial_timesteps,final_timesteps=GP.final_timesteps)
                    ema_decay_rate = ema_decay_rate_schedule(
                                            num_timesteps,
                                            initial_ema_decay_rate=0.95,
                                            initial_timesteps=2,
                                        )
                    ##print('ema_decay_rate:', ema_decay_rate)

                    #st2 = time.time()
                    update_ema_model(ema_model, model,ema_decay_rate) #
                    #end2 = time.time()
                    #print('update_ema_model time s:', round(end2 - st2, 4))
                    
                    end = time.perf_counter()
                    #print('a batch time s:', round(end - st, 4)) #3.2, 0.4,onlineemamodel
                    if torch.distributed.get_rank() == 0:
                        if num % 100 == 0:
                            torch.save({
                                'config': config,
                                'model': model.state_dict(),
                                'ema_model': ema_model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'scheduler': scheduler.state_dict(),
                                'iteration': it,
                                'args': args,
                                'equiformer': config.equiformer,
                                'escn': config.escn,
                            }, ckpt_path)


                    if torch.distributed.get_rank() == 0:
                        logger.info(
                            '[Train] Step %d | iter %d | subdata %d | Loss %.6f (pos %.6f | v %.6f | exp %.6f | dismat %.6f | bond %.6f | angle %.6f | dihedral %.6f | rmsd %.6f)' % (
                                step, it, sub_id, loss, loss_pos, loss_v, loss_exp, loss_dismat, loss_bond, loss_angle, loss_dihedral, rmsd
                            ))
                        
                if torch.distributed.get_rank() == 0:
                    logger.info(
                        '[Train] Iter %d | subdata %d | Loss %.6f (pos %.6f | v %.6f | exp %.6f | dismat %.6f | bond %.6f | angle %.6f | dihedral %.6f | rmsd %.6f)' % (
                            batch_id + num + 1, sub_id, np.mean(step_loss['loss'][-1]), np.mean(step_loss['loss_pos'][-1]), np.mean(step_loss['loss_v'][-1]), np.mean(step_loss['loss_exp'][-1]), 
                            np.mean(step_loss['loss_dismat'][-1]), np.mean(step_loss['loss_bond'][-1]), np.mean(step_loss['loss_angle'][-1]), np.mean(step_loss['loss_dihedral'][-1]), np.mean(step_loss['rmsd'][-1]), 
                        )
                    )

        #if torch.distributed.get_rank() == 0:
        for k, v in results.items():
            if torch.is_tensor(v) and v.squeeze().ndim == 0:
                writer.add_scalar(f'train/{k}', v, it)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
        writer.add_scalar('train/grad', orig_grad_norm, it)
        writer.flush()


    def validate(local_rank, args, it):  
        # fix time steps
        with torch.no_grad():
            model.eval()
            ema_model.eval()
            step_all_loss = defaultdict(list)
            for num, batch in enumerate(tqdm(val_loader, desc='Validate')):

                #
                b_protein_pos=batch.protein_pos.cuda(local_rank, non_blocking=True)
                b_protein_v=batch.protein_atom_feature.float().cuda(local_rank, non_blocking=True)
                b_affinity=batch.affinity.float().cuda(local_rank, non_blocking=True)
                b_batch_protein=batch.protein_element_batch.cuda(local_rank, non_blocking=True)

                b_ligand_pos=batch.ligand_pos.cuda(local_rank, non_blocking=True)
                b_ligand_v=batch.ligand_atom_feature_full.cuda(local_rank, non_blocking=True)
                b_batch_ligand=batch.ligand_element_batch.cuda(local_rank, non_blocking=True)

                b_ligand_bond_index = batch.ligand_bond_index.cuda(local_rank, non_blocking=True) #[2, 582]
                b_ligand_bond_type  = batch.ligand_bond_type.cuda(local_rank, non_blocking=True)
                b_ligand_bond_type_batch = batch.ligand_bond_type_batch.cuda(local_rank, non_blocking=True)

                b_protein_element = batch.protein_element.cuda(local_rank, non_blocking=True)
                b_ligand_element  = batch.ligand_element.cuda(local_rank, non_blocking=True)

                b_ligand_fill_coords =  batch.ligand_fill_coords.cuda(local_rank, non_blocking=True)

                zmats = batch.ligand_fill_zmats.cuda(local_rank, non_blocking=True).view(-1, GP.max_atoms, 4)
                bzids=torch.arange(zmats.shape[0]).view(-1,1).tile((1,zmats.shape[1])).unsqueeze(-1).long().cuda(local_rank, non_blocking=True)
                zmats=torch.concat((bzids,zmats),axis=-1).cuda(local_rank, non_blocking=True).view(-1, 5)
                b_ligand_fill_zmats  =  zmats

                b_ligand_fill_masks  =  batch.ligand_fill_masks.cuda(local_rank, non_blocking=True)
                b_ligand_fill_atom_order    =  batch.ligand_fill_atom_order.cuda(local_rank, non_blocking=True)

                b_ligand_atom_isring, b_ligand_atom_isO, b_ligand_atom_isN = batch.ligand_atom_isring.cuda(local_rank, non_blocking=True), batch.ligand_atom_isO.cuda(local_rank, non_blocking=True), batch.ligand_atom_isN.cuda(local_rank, non_blocking=True)
                b_protein_atom_isring, b_protein_atom_isO, b_protein_atom_isN = batch.protein_atom_isring.cuda(local_rank, non_blocking=True), batch.protein_atom_isO.cuda(local_rank, non_blocking=True), batch.protein_atom_isN.cuda(local_rank, non_blocking=True)
            

                b_protein_cross_lig_isring_flag = batch.protein_cross_lig_isring_flag.cuda(local_rank, non_blocking=True)
                b_protein_cross_lig_isO_flag = batch.protein_cross_lig_isO_flag.cuda(local_rank, non_blocking=True)
                b_protein_cross_lig_isN_flag = batch.protein_cross_lig_isN_flag.cuda(local_rank, non_blocking=True)

                b_protein_cross_pro_isring_flag = batch.protein_cross_pro_isring_flag.cuda(local_rank, non_blocking=True)
                b_protein_cross_pro_isO_flag = batch.protein_cross_pro_isO_flag.cuda(local_rank, non_blocking=True)
                b_protein_cross_pro_isN_flag = batch.protein_cross_pro_isN_flag.cuda(local_rank, non_blocking=True)

                b_protein_cross_ligand    = batch.protein_cross_ligand.cuda(local_rank, non_blocking=True)
                b_protein_cross_protein   = batch.protein_cross_protein.cuda(local_rank, non_blocking=True)
                
                #print('type(batch.protein_cross_distance):', type(batch.protein_cross_distance)) #protein_cross_distance，pyg，
                #cross_distance，PyG，，set，
                #setlist，：[{tensor([1, 2, 3]), tensor([1, 2, 3])}, {tensor([4, 5]), tensor([4, 5]), tensor([4, 5])}]
                #

                b_protein_cross_distance = []
                if isinstance(batch.protein_cross_distance, list):
                    for i in batch.protein_cross_distance: #protein_cross_distanceslist
                        #ii = torch.stack(list(i), dim = 0) #list，，
                        #b_protein_cross_distance.append(ii.cuda(local_rank, non_blocking=True))
                        tg = torch.from_numpy(i).cuda(local_rank, non_blocking=True)
                        b_protein_cross_distance.append(tg)
        
                else:
                    b_protein_cross_distance.append(batch.protein_cross_distance.cuda(local_rank, non_blocking=True))
                    

                b_cross_bond_index = batch.protein_link_e.T.cuda(local_rank, non_blocking=True)
                b_cross_bond_type = batch.protein_link_t.cuda(local_rank, non_blocking=True)
                b_cross_bond_index_reverse = batch.protein_link_e_reverse.T.cuda(local_rank, non_blocking=True)
                b_cross_bond_type_reverse = batch.protein_link_t_reverse.cuda(local_rank, non_blocking=True)






                ##print(batch)
                #batch = batch.cuda(local_rank, non_blocking=False)
                # 
                #torch.cuda.synchronize()

                batch_size = batch.num_graphs

                step_loss = defaultdict(list)
                for step in np.array(GP.steps_list) - 1: #
                    #？
                    #protein_noise = torch.randn_like(batch.protein_pos) * config.train.pos_noise_std #
                    #gt_protein_pos = batch.protein_pos + protein_noise
                    #gt_protein_pos = batch.protein_pos
                #with torch.autocast(device_type='cuda'):
                    if config.model.diffusion_mode == 'CM':
                        results = consistency_training(
                            #sigma_min=GP.sigma_min,
                            #sigma_max=GP.sigma_max,
                            #rho=GP.rho,
                            #sigma_data=GP.sigma_data,
                            #initial_timesteps=GP.initial_timesteps,
                            #final_timesteps=GP.final_timesteps,
                            online_model=model, 
                            ema_model=ema_model, 
                            current_training_step=step,
                            total_training_steps=GP.final_timesteps,
                            
                            args=args, 
                            config=config.model, 
                            protein_atom_feature_dim=protein_featurizer.feature_dim,
                            ligand_atom_feature_dim=ligand_featurizer.feature_dim,

                            protein_pos=b_protein_pos,
                            protein_v=b_protein_v,
                            affinity=b_affinity,
                            batch_protein=b_batch_protein,

                            ligand_pos=b_ligand_pos,
                            ligand_v=b_ligand_v,
                            batch_ligand=b_batch_ligand,

                            ligand_bond_index = b_ligand_bond_index, #[2, 582]
                            ligand_bond_type  = b_ligand_bond_type,
                            ligand_bond_type_batch = b_ligand_bond_type_batch,

                            protein_element = b_protein_element,
                            ligand_element  = b_ligand_element,

                            ligand_mol = batch.ligand_mol,

                            ligand_fill_coords =  b_ligand_fill_coords,
                            ligand_fill_zmats  =  b_ligand_fill_zmats,
                            ligand_fill_masks  =  b_ligand_fill_masks,
                            ligand_fill_atom_order  = b_ligand_fill_atom_order,

                            ligand_atom_isring  = b_ligand_atom_isring,
                            ligand_atom_isO     = b_ligand_atom_isO,
                            ligand_atom_isN     = b_ligand_atom_isN,

                            protein_atom_isring = b_protein_atom_isring,
                            protein_atom_isO    = b_protein_atom_isO,
                            protein_atom_isN    = b_protein_atom_isN,

                            cross_lig_isring_flag = b_protein_cross_lig_isring_flag,
                            cross_lig_isO_flag = b_protein_cross_lig_isO_flag,
                            cross_lig_isN_flag = b_protein_cross_lig_isN_flag,

                            cross_pro_isring_flag = b_protein_cross_pro_isring_flag,
                            cross_pro_isO_flag = b_protein_cross_pro_isO_flag,
                            cross_pro_isN_flag = b_protein_cross_pro_isN_flag,

                            cross_ligand    = b_protein_cross_ligand,
                            cross_protein   = b_protein_cross_protein,
                            cross_distance  = b_protein_cross_distance,

                            cross_bond_index = b_cross_bond_index,
                            cross_bond_type = b_cross_bond_type, 
                            cross_bond_index_reverse = b_cross_bond_index_reverse, 
                            cross_bond_type_reverse = b_cross_bond_type_reverse,
                            
                            )
                        
                    elif config.model.diffusion_mode == 'DDPM':
                        results = model.get_diffusion_loss(
                            args=args, 
                            config=config.model, 
                            protein_atom_feature_dim=protein_featurizer.feature_dim,
                            ligand_atom_feature_dim=ligand_featurizer.feature_dim,

                            protein_pos=b_protein_pos,
                            protein_v=b_protein_v,
                            affinity=b_affinity,
                            batch_protein=b_batch_protein,

                            ligand_pos=b_ligand_pos,
                            ligand_v=b_ligand_v,
                            batch_ligand=b_batch_ligand,

                            ligand_bond_index = b_ligand_bond_index, #[2, 582]
                            ligand_bond_type  = b_ligand_bond_type,
                            ligand_bond_type_batch = b_ligand_bond_type_batch,

                            protein_element = b_protein_element,
                            ligand_element  = b_ligand_element,

                            ligand_mol = batch.ligand_mol,

                            ligand_fill_coords =  b_ligand_fill_coords,
                            ligand_fill_zmats  =  b_ligand_fill_zmats,
                            ligand_fill_masks  =  b_ligand_fill_masks,
                            ligand_fill_atom_order  = b_ligand_fill_atom_order,

                            ligand_atom_isring  = b_ligand_atom_isring,
                            ligand_atom_isO     = b_ligand_atom_isO,
                            ligand_atom_isN     = b_ligand_atom_isN,

                            protein_atom_isring = b_protein_atom_isring,
                            protein_atom_isO    = b_protein_atom_isO,
                            protein_atom_isN    = b_protein_atom_isN,



                            cross_lig_isring_flag = b_protein_cross_lig_isring_flag,
                            cross_lig_isO_flag = b_protein_cross_lig_isO_flag,
                            cross_lig_isN_flag = b_protein_cross_lig_isN_flag,

                            cross_pro_isring_flag = b_protein_cross_pro_isring_flag,
                            cross_pro_isO_flag = b_protein_cross_pro_isO_flag,
                            cross_pro_isN_flag = b_protein_cross_pro_isN_flag,

                            cross_ligand    = b_protein_cross_ligand,
                            cross_protein   = b_protein_cross_protein,
                            cross_distance  = b_protein_cross_distance,

                            
                            cross_bond_index = b_cross_bond_index,
                            cross_bond_type = b_cross_bond_type, 
                            cross_bond_index_reverse = b_cross_bond_index_reverse, 
                            cross_bond_type_reverse = b_cross_bond_type_reverse,
                            
                            )
                


                    if args.value_only:
                        results['loss'] = results['loss_exp']
                        
                    loss, loss_pos, loss_v, loss_exp, loss_dismat, loss_bond, loss_angle, loss_dihedral, rmsd = results['loss'], results['loss_pos'], results['loss_v'],\
                            results['loss_exp'], results['loss_dismat'], results['loss_bond'], results['loss_angle'], results['loss_dihedral'], results['rmsd'],
                    step_loss['loss'].append(loss.item())
                    step_loss['loss_pos'].append(loss_pos.item())
                    step_loss['loss_v'].append(loss_v.item())
                    step_loss['loss_exp'].append(loss_exp.item())
                    step_loss['loss_dismat'].append(loss_dismat.item()) 
                    step_loss['loss_bond'].append(loss_bond.item())
                    step_loss['loss_angle'].append(loss_angle.item())
                    step_loss['loss_dihedral'].append(loss_dihedral.item())
                    step_loss['rmsd'].append(rmsd.item())

                    step_all_loss['loss'].append(loss.item())
                    step_all_loss['loss_pos'].append(loss_pos.item())
                    step_all_loss['loss_v'].append(loss_v.item())
                    step_all_loss['loss_exp'].append(loss_exp.item())
                    step_all_loss['loss_dismat'].append(loss_dismat.item())
                    step_all_loss['loss_bond'].append(loss_bond.item())
                    step_all_loss['loss_angle'].append(loss_angle.item())
                    step_all_loss['loss_dihedral'].append(loss_dihedral.item())
                    step_all_loss['rmsd'].append(rmsd.item())


                    if torch.distributed.get_rank() == 0:
                        logger.info(
                            '[Validate] Step %d | iter %d | subdata %d | Loss %.6f (pos %.6f | v %.6f | exp %.6f | dismat %.6f | bond %.6f | angle %.6f | dihedral %.6f | rmsd %.6f)' % (
                                step, it, sub_id, loss, loss_pos, loss_v, loss_exp, loss_dismat, loss_bond, loss_angle, loss_dihedral, rmsd
                            ))

            #epoch        
            if torch.distributed.get_rank() == 0:
                logger.info(
                    '[Validate] Iter %d | subdata %d | Loss %.6f (pos %.6f | v %.6f | exp %.6f | dismat %.6f | bond %.6f | angle %.6f | dihedral %.6f | rmsd %.6f)' % (
                        it, sub_id, np.mean(step_loss['loss']), np.mean(step_loss['loss_pos']), np.mean(step_loss['loss_v']), np.mean(step_loss['loss_exp']), 
                        np.mean(step_loss['loss_dismat']), np.mean(step_loss['loss_bond']), np.mean(step_loss['loss_angle']), np.mean(step_loss['los_dihedral']), np.mean(step_loss['rmsd']), 
                    )
                )


            if args.value_only:
                return np.mean(step_all_loss['loss_v'])
            
            return np.mean(step_all_loss['loss'])

    
    
    try:
        best_loss, best_iter = None, None
        for it in range(start_it, config.train.max_iters):
            #it，epoch
            #with torch.autograd.detect_anomaly():
            # torch.autograd.detect_anomaly() ，，
            for sub_id, train_set in enumerate(subset_train):
                # 
                #transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
                #train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
                train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle = True, num_replicas=world_size, rank=local_rank) #sample
                #train_sampler = torch.utils.data.distributed.DistributedSampler(train_set) #sample

                #print('train_set[0] type:', type(train_set[0])) # <class 'KGDiff.datasets.pl_data.ProteinLigandData'>
                #print('train_set[0]', train_set[0])
                #print('train_set[0].keys()', train_set[0].keys())
                
                train_loader =DataLoader(
                    train_set,
                    batch_size=config.train.batch_size,
                    shuffle=False, #sampler，DataLoader，sampler。shuffleFalse
                    num_workers=config.train.num_workers,
                    #num_workers=16,
                    follow_batch=FOLLOW_BATCH,  #FOLLOW_BATCH = ('protein_element', 'ligand_element', 'ligand_bond_type',) #id，_batch
                    exclude_keys=collate_exclude_keys,  #，。，GNN？？
                    sampler=train_sampler,
                    pin_memory=True,
                    prefetch_factor=8, #，
                    #collate_fn=custom_collate, #pyg.Data，， <class 'KGDiff.datasets.pl_data.ProteinLigandData'>
                    #exclude_keys = exclude_keys
                )

                ##PYG dataloadercollate_fn，，exclude_keys，，exclude_keys

                train_sampler.set_epoch(it)

                #GPU，GPU,GPU
                #，，，
                ckpt_path = os.path.join(ckpt_dir, 'final.pt')

                epoch_batch_num = math.ceil(args.train_set_num / config.train.batch_size) #batch
                #assert len(train_loader) == len(train_set) / math.ceil(config.train.batch_size) #train_loaderlen，
                sub_batch_num   = math.ceil(len(train_set) / config.train.batch_size)
                batch_id        = it * epoch_batch_num + sub_id * sub_batch_num
                print('epoch_batch_num, sub_batch_num:', epoch_batch_num, sub_batch_num)

                train(local_rank, args, it, sub_id, train_loader, ckpt_path, batch_id) #batch
                if local_rank == 0:
                    torch.save({
                        'config': config,
                        'model': model.state_dict(),
                        'ema_model': ema_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'iteration': it,
                        'args': args,
                        'equiformer': config.equiformer,
                        'escn': config.escn,
                    }, ckpt_path)

                #del train_loader
                #gc.collect()
                #if len(subset_train) > 1:
                    #dist.barrier() #，，PYG，，

            if local_rank == 0:
                #if it % config.train.val_freq == 0 and it != 0 or it == config.train.max_iters:
                val_loss = validate(local_rank, args, it) #epoch，batch，

                if best_loss is None or val_loss < best_loss:
                    if torch.distributed.get_rank() == 0:
                        logger.info(f'[Validate] Best val loss achieved: {val_loss:.6f}')
                    best_loss, best_iter = val_loss, it
                    ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                    torch.save({
                        'config': config,
                        'model': model.state_dict(),
                        'ema_model': ema_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'iteration': it,
                        'args': args,
                        'equiformer': config.equiformer,
                        'escn': config.escn,
                    }, ckpt_path)
                else:
                    if torch.distributed.get_rank() == 0:
                        logger.info(f'[Validate] Val loss is not improved. '
                                    f'Best val loss: {best_loss:.6f} at iter {best_iter}')
                
            #dist.barrier() #，
                    
    except KeyboardInterrupt:
        print('Terminating...')

    
    dist.destroy_process_group()
        

if __name__ == '__main__':

    #['cudagraphs', 'inductor', 'onnxrt', 'openxla', 'openxla_eval', 'tvm']
    #main = torch.compile(main, mode="max-autotune", dynamic=True, fullgraph=True, backend='inductor') #torch.compile2.0
    #torch.compilemodel，，

    ##PYG dataloadercollate_fn，，exclude_keys，，exclude_keys
    #PYG dataloader
    #exclude_keys = ['protein_cross_distance']
    collate_exclude_keys = ['ligand_nbh_list']
    main()
