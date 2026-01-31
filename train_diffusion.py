import os
import sys
import argparse
import shutil
import numpy as np
import numpy
import pandas as pd
import torch
import torch.utils.tensorboard
import seaborn as sns
# sns.set_theme(style="darkgrid")

import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score
from scipy import stats

from torch.nn.utils import clip_grad_norm_





# EcConf
import numpy as np 
from rdkit import Chem
from EcConf.graphs import * 
from EcConf.utils import *
from EcConf.model import *  #这个导致的批量问题，因为torch的DataLoader和pyg的DataLoader同名了，所以要么注释掉，要么放在pyg的前面
from EcConf.comparm import *





#from torch_geometric.data import DataLoader #这一步过不去
try:
    from torch_geometric.loader import DataLoader #继承了torch的DataLoader, 同名，别调用错了, 目前找不大处理形状不一样数据的方法，list也不返回
except Exception as e:
    print(e)
    from torch_geometric.data import DataLoader

from torch_geometric.transforms import Compose
from torch_geometric.data import Data

from tqdm.auto import tqdm #自动选择合适的版本
import sys
sys.path.append(os.path.abspath('./'))
import KGDiff.utils.misc as misc
import KGDiff.utils.train as utils_train
import KGDiff.utils.transforms as trans #这个有问题

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


#导入采样测试代码

from KGDiff.scripts.sample_diffusion import sample_main, boxplot

from KGDiff.scripts.evaluate import rmsds, read_file, train_evaluate

import warnings
warnings.filterwarnings("ignore", message="The IPv6 network addresses.*")
from datetime import timedelta
try:
    scaler = torch.amp.GradScaler()
except Exception as e:
    print(e)
    scaler = torch.cuda.amp.GradScaler()

#import triton #使用torch.complie 不用显示导入triton


# 初始化分布式训练环境
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
    # 从 SMILES 字符串创建分子对象
    ##print(smiles)
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    ##print(mol)
    flag = False

    # 生成三维构象
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    try:
        AllChem.UFFOptimizeMolecule(mol)
    except Exception as e:
        ##print(e)
        flag = True
        return mol, flag

    return mol, flag


def save_sdf(mol, output_sdf):
    # 创建 SDF 文件写入对象
    writer = Chem.SDWriter(output_sdf)
    # 将分子写入 SDF 文件
    writer.write(mol)
    # 关闭 SDF 文件写入对象
    writer.close()
    #print(f"SDF 文件已生成：{output_sdf}")


def add_knn_ligand_pos(data):
    knn_data = []
    fail_count = 0
    for dt in data:
        #使用rdkit生成3D构象用于构建KNN图
        # 生成三维构象
        ##print('dt:', dt)
        mol, fail_flag = generate_3d_conformer_from_smiles(dt.ligand_smiles)

        if fail_flag == True:
            dt.knn_ligand_pos = dt.ligand_pos #如果rdkit推理失败，则使用真实的坐标
            knn_data.append(dt)
            fail_count += 1
        else:
            conformer      = mol.GetConformer()
            knn_ligand_pos = conformer.GetPositions() #专门用于构建KNN图的
            knn_ligand_pos = torch.FloatTensor(knn_ligand_pos)
            centor = knn_ligand_pos.mean(dim = 0) - dt.ligand_pos.mean(dim = 0)
            dt.knn_ligand_pos = knn_ligand_pos - centor
            knn_data.append(dt)
    
    return knn_data, fail_count


def set_seed(seed):
    torch.manual_seed(seed)  # 设置 PyTorch 的随机数种子
    torch.cuda.manual_seed_all(seed)  # 设置所有 GPU 的随机数种子
    np.random.seed(seed)  # 设置 NumPy 的随机数种子
    random.seed(seed)  # 设置 Python 自带的随机数种子
    torch.backends.cudnn.deterministic = True  # 设置 CuDNN 算法为确定性算法
    torch.backends.cudnn.benchmark = True


# 自定义collate函数
def custom_collate(batch):
    # 在这里指定不想连接的分量（比如 'z'）
    exclude_keys = ['protein_cross_distance']

    # 初始化用于存储连接数据的字典
    batch_data = {}
    
    keys = batch[0].keys #使用pyg2.1.0，更新的版本，则不行
    # 处理每个属性
    for key in keys:
        if key in exclude_keys:
            # 对于需要排除的分量，收集成列表
            batch_data[key] = [getattr(data, key) for data in batch]
        else:
            # 对于需要连接的分量，使用默认的方式进行连接
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
    parser.add_argument('--log_name', type=str, default='') #用于区分不同配置的模型
    parser.add_argument('--epoch', type=int, default=2000) #用于区分不同配置的模型
    try:
        parser.add_argument("--local-rank", type=int,  help='rank in current node') #不要提供默认值，有些版本的torch可能不支持, 参数名字是local-rank，有些版本是local_rank
    except Exception:
        parser.add_argument("--local_rank", type=int,  help='rank in current node')

    # 设置随机数种子
    seed = 2024
    set_seed(seed)

    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"]) #进程数量, 对应所使用的GPU数量，而不是CPU的线程数量

    # 1) 初始化
    #torch.distributed.init_process_group(backend="gloo", init_method='env://', rank=local_rank, world_size=world_size, timeout=timedelta(minutes=100))
    torch.distributed.init_process_group(backend="nccl", init_method='env://', rank=local_rank, world_size=world_size, timeout=timedelta(minutes=100))
    
    # 2） 配置每个进程的gpu
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    args.device = device
    args.rank = local_rank

    print('local_rank:', local_rank)
    print('device:', device)
    


    # load ckpt
    if args.ckpt:
        #print(f'loading {args.ckpt}...')
        #torch.serialization.add_safe_globals([EasyDict])
        ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False) #2.6之后新增：weights_only=False允许加载模型之外的字典参数
        config = ckpt['config']
        config = misc.load_config(args.config) #不使用上一轮保存的配置文件，使用新的配置参数
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
    shutil.copytree('./EcConf', os.path.join(log_dir, 'EcConf'), dirs_exist_ok=True)
    shutil.copytree('./configs', os.path.join(log_dir, 'configs'), dirs_exist_ok=True)
    shutil.copytree('./ocp', os.path.join(log_dir, 'ocp'), dirs_exist_ok=True)


    #设置批量大小
    if torch.cuda.get_device_properties(local_rank).total_memory / 1000**3 >= 38:
        config.train.batch_size         = 32
        config.equiformer.batch_size    = 5
        config.escn.batch_size          = 4
    elif torch.cuda.get_device_properties(local_rank).total_memory / 1000**3 >= 30:
        config.train.batch_size         = 32
        config.equiformer.batch_size    = 5
        config.escn.batch_size          = 4
    elif torch.cuda.get_device_properties(local_rank).total_memory / 1000**3 >= 24:
        config.train.batch_size         = 16
        config.equiformer.batch_size    = 3
        config.escn.batch_size          = 2
    else:
        config.train.batch_size         = 8
        config.equiformer.batch_size    = 1
        config.escn.batch_size          = 1



    #设置是否需要梯度累积
    #由于EGNN比Equiformer耗的显存更少，前者通常是后者批量的2倍，因此前者最大批量可以设置成16，后者可以设置成8
    if torch.cuda.get_device_properties(local_rank).total_memory / 1000**3 >= 38:
        grad_num = 1
    elif torch.cuda.get_device_properties(local_rank).total_memory / 1000**3 >= 24:
        grad_num = 1
    else:
        grad_num = 1 #是否需要梯度累积，这个需要看机器，记得修改. 由于consistency模型训练方式的独特性，不容易实现梯度累积，所以先不使用了

    #对equiformer的批量和隐藏维度进行特殊处理
    if config.model.model_mode == 'equiformer':
        lmax_list = config.equiformer.lmax_list
        num_resolutions = len(lmax_list)
        num_coefficients = 0
        for i in range(num_resolutions):
            num_coefficients = num_coefficients + int((lmax_list[i] + 1) ** 2) #球坐标系的谐波函数（其余函数可以由多个该函数来模拟或表示）的阶数，影响填充的注意力数量

        print('num_coefficients = 49 ?:', num_coefficients)
        config.train.batch_size = config.equiformer.batch_size
        config.model.hidden_dim = config.equiformer.attn_hidden_channels * num_coefficients # num_coefficients默认是49


    #对escn的批量和隐藏维度进行特殊处理
    if config.model.model_mode == 'escn':
        lmax_list = config.escn.lmax_list
        num_resolutions = len(lmax_list)
        num_coefficients = 0
        for i in range(num_resolutions):
            num_coefficients = num_coefficients + int((lmax_list[i] + 1) ** 2) #球坐标系的谐波函数（其余函数可以由多个该函数来模拟或表示）的阶数，影响填充的注意力数量

        print('num_coefficients = 49 ?:', num_coefficients)
        config.train.batch_size = config.escn.batch_size
        config.model.hidden_dim = config.escn.sphere_channels * num_coefficients # num_coefficients默认是49

    
    # Transforms
    protein_featurizer = trans.FeaturizeProteinAtom() #原子特征化，即获取原子的初始特征
    ligand_featurizer = trans.FeaturizeLigandAtom(config.data.transform.ligand_atom_mode) #默认添加芳香原子
    transform_list = [
        protein_featurizer,
        ligand_featurizer,
        trans.FeaturizeLigandBond(), #配体键长特征化,这里使用了配体键类型，因此不用再特意加键类型信息了?遗憾的是，ligand_bond_feature并没有在别处使用
        trans.NormalizeVina(config.data.name) #vine标准化
    ]
    
    if config.data.transform.random_rot:
        transform_list.append(trans.RandomRotation())
    transform = Compose(transform_list)





    # Model
    print('Building model...')

    model = ScorePosNet3D(
        config.model,
        protein_atom_feature_dim=protein_featurizer.feature_dim, #等于27，即氨基酸数量+所使用的原子序号数量
        ligand_atom_feature_dim=ligand_featurizer.feature_dim,  #等于所使用的原子序号数量
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


    if GP.ema_exit:
        print('使用传统的ema模型')
    else:
        print('不使用传统的ema模型')






    if config.model.diffusion_mode == 'DDPM':
        print('使用DDPM')
    elif config.model.diffusion_mode == 'CM':
        print('使用Consistency Model')

    model.cuda(local_rank)
    ema_model.cuda(local_rank)

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    ema_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(ema_model)

    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True, output_device=local_rank, broadcast_buffers=False) 
    ema_model = nn.parallel.DistributedDataParallel(ema_model, device_ids=[local_rank], find_unused_parameters=True, output_device=local_rank, broadcast_buffers=False) 
    #find_unused_parameters=True 可以保证在加载模型的时候，不会在第一张GPU上额外多处几个进程，来维护数据，造成资源浪费
    #backend: ['cudagraphs', 'inductor', 'onnxrt', 'openxla', 'openxla_eval', 'tvm']
    #mode: default, reduce-overhead, max-autotune
    
    model = torch.compile(model, mode='reduce-overhead', dynamic=True, fullgraph=True, backend='inductor') #torch.compile在2.0后才能用
    ema_model = torch.compile(ema_model, mode='reduce-overhead', dynamic=True, fullgraph=True, backend='inductor') #torch.compile在2.0后才能用
    
    '''   
    try:
        model = torch.compile(model, mode='max-autotune', dynamic=True, fullgraph=True, backend='inductor') #torch.compile在2.0后才能用
        ema_model = torch.compile(ema_model, mode='max-autotune', dynamic=True, fullgraph=True, backend='inductor') #torch.compile在2.0后才能用
    except Exception as e:
        print('not use pytorch2.0 compile, skip')
        pass
    '''
    model = model.module
    ema_model = ema_model.module

    

    # #print(model)
    print(f'protein feature dim: {protein_featurizer.feature_dim} ligand feature dim: {ligand_featurizer.feature_dim}')
    logger.info(f'# trainable parameters: {misc.count_parameters(model) / 1e6:.4f} M') #只统计带有梯度更新的，不要没参与梯度更新的
    logger.info(f'# not trainable parameters: {misc.count_non_grad_parameters(model) / 1e6:.4f} M') #只统计带有梯度更新的，不要没参与梯度更新的

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
        #start_it = 0
    
    args.all_protein_max_atom_num = config.data.protein_max_atom_num
    args.all_ligand_max_atom_num  = config.data.ligand_max_atom_num











    def load_data(st_i = 0, end_i = 100000000000000000000, num = None):
        # Datasets and loaders
        print('Loading dataset...')
        #raise Exception('stop0')

        #dataset是所有数据集，subdets是训练、验证、测试
        dataset, subsets = get_dataset(
            config=config.data,
            transform=transform,
            data_name = GP.train_data_name,
            st_i = st_i,
            end_i = end_i,
        )
        #print('data_name:',config.data.name)s

        subset_train = []

        if config.data.name == 'pl': # 默认是这个
            train_set, val_set, test_set = subsets['train'], subsets['test'], []
        elif config.data.name == 'pdbbind':
            if num:
                train_set, val_set, test_set = list(subsets['train'])[:num], list(subsets['valid'])[:num], list(subsets['test'])[:num]
            else:
                train_set, val_set, test_set = list(subsets['train'])[:], list(subsets['valid'])[:], list(subsets['test'])[:]
            #train_set, val_set, test_set = subsets['train'], subsets['valid'], subsets['test']

            train_set, val_set, test_set = [x for x in train_set if x is not None], [x for x in val_set if x is not None], [x for x in test_set if x is not None]


            print(f'Training: {len(train_set)} Validation: {len(val_set)} Test: {len(test_set)}')
            #exit()

            data_dict = defaultdict(lambda: defaultdict(lambda: []))
            #保存训练，验证，测试的文件名字
            #pocket_fn = 'v2020-other-PL/4po7/4po7_pocket10.pdb'
            #ligand_fn = 'v2020-other-PL/4po7/4po7_ligand.sdf'
            '''
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
            '''
            #exit()
            #子集, 选择合适的[14000,7000,3500,2000].
            #根据显卡来确定分隔的子集数量，方便保存模型

            if torch.cuda.get_device_properties(local_rank).total_memory / 1000**3 >= 38:
                lens = 500000
            elif torch.cuda.get_device_properties(local_rank).total_memory / 1000**3 >= 24:
                lens = 500000
            else:
                lens = 500000

            args.train_set_num  = len(train_set)
            args.val_set_num    = len(val_set)
            args.test_set_num   = len(test_set)

            sub_num = math.ceil(len(train_set) / lens)
            for i in range(sub_num):
                subset_train.append(train_set[i*lens: (i+1)*lens])

            #train_set, val_set, test_set = subsets['train'], subsets['val'], subsets['test']
            #train_set, val_set, test_set = list(subsets['train'])[:100], list(subsets['train'])[:10], list(subsets['train'])[:10]

            #存在一个问题，当全部的数据集使用时，第一张卡的计算单元会被卡住，计算的非常慢, 为什么呢？
            #出现多GPU突然卡住的问题，主要是数据在CPU上浪费太多时间，GPU利用率很低，主要原因是在计算损失的时候，使用了numpy，list，循环操作等非常耗时间的CPU操作导致，改成
            #GPU tensor操作即可，同时尽可能使用矩阵运算，批量处理
            
        else:
            raise ValueError
        

        print(f'Training: {len(train_set)} Validation: {len(val_set)} Test: {len(test_set)}')

        #collate_exclude_keys = ['ligand_nbh_list']
        '''
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
        '''
        
        return subset_train















    def train(gpu, args, it, sub_id, train_loader, ckpt_path, batch_id):
        model.train()
        ema_model.train()

        #for num, (batch, b_protein_cross_distance) in tqdm(enumerate(train_loader), desc='Training'):
        
        for num, batch in enumerate(tqdm(train_loader, desc='Training')):
            #print('type(batch_):', type(batch_))
            #print('batch_ num:', len(batch_))
            #print('batch_:', batch_)
            #batch, b_protein_cross_distance = batch_
            #一个问题，原子序号有超过17，为什么？是新增的数据导致的？是的，新增数据的导致，新的数据中有原子序号大于17的

            '''
            if max(batch.protein_element) > 17 or max(batch.ligand_element) > 17:
                print('batch.protein_element:', batch.protein_element)
                print('batch.ligand_element:', batch.ligand_element)
                print('max(batch.protein_element), max(batch.ligand_element):', max(batch.protein_element), max(batch.ligand_element))
                raise Exception(f'>17')
            else:
                continue
            '''
            
            #这一部分数据是可以共享的


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

            b_protein_coords_predict = batch.protein_coords_predict.cuda(local_rank, non_blocking=True)
            #print('type(batch.protein_cross_distance):', type(batch.protein_cross_distance)) #每一个数据的protein_cross_distance都不一样，pyg无法组合，因此要么填充
            #cross_distance的形状不一样，PyG无法连接，所以报错，因此一种可行的方法是套一个集合set，之后解析时再特殊处理
            #这是一个包含set的list，形式如下：[{tensor([1, 2, 3]), tensor([1, 2, 3])}, {tensor([4, 5]), tensor([4, 5]), tensor([4, 5])}]
            #所以要解析出来

            b_protein_cross_distance = []
            if isinstance(batch.protein_cross_distance, list):
                for i in batch.protein_cross_distance: #protein_cross_distances是一个list
                    #ii = torch.stack(list(i), dim = 0) #集合转list，再连接，恢复成张量
                    #b_protein_cross_distance.append(ii.cuda(local_rank, non_blocking=True))
                    tg = torch.from_numpy(i).cuda(local_rank, non_blocking=True)
                    b_protein_cross_distance.append(tg)
    
            else:
                b_protein_cross_distance.append(batch.protein_cross_distance.cuda(local_rank, non_blocking=True))
                

            #b_cross_bond_index = batch.protein_link_e.T.cuda(local_rank, non_blocking=True)
            b_cross_bond_type = batch.protein_link_t.cuda(local_rank, non_blocking=True)
            #b_cross_bond_index_reverse = batch.protein_link_e_reverse.T.cuda(local_rank, non_blocking=True) 
            b_cross_bond_type_reverse = batch.protein_link_t_reverse.cuda(local_rank, non_blocking=True)

            
            #b_link_e = batch.protein_link_e.cuda(local_rank, non_blocking=True)
            #b_link_e_reverse = batch.protein_link_e_reverse.cuda(local_rank, non_blocking=True)

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
                
            
            
            
            b_protein_element_batch = batch.protein_element_batch.cuda(local_rank, non_blocking=True)
            b_protein_link_t_batch = batch.protein_link_t_batch.cuda(local_rank, non_blocking=True)
            b_protein_link_t_reverse_batch = batch.protein_link_t_reverse_batch.cuda(local_rank, non_blocking=True)
            b_ligand_element_batch = batch.ligand_element_batch.cuda(local_rank, non_blocking=True)

            b_rd_pos = batch.ligand_rd_pos.cuda(local_rank, non_blocking=True)
            
            #b_ligand_emb = batch.protein_ligand_emb.cuda(local_rank, non_blocking=True)
            #b_pocket_emb = batch.protein_pocket_emb.cuda(local_rank, non_blocking=True)


            torch.cuda.current_stream().synchronize()
            
            
            if config.model.diffusion_mode == 'DDPM':
                GP.steps_list = [1]

            for _ in range(1): #为了提供利用率，数据可以复用2次
                #batch = batch.cuda(local_rank, non_blocking=False) #直接将整个batch都加载到GPU上，太浪费资源了，需要什么数据就把那些数据给加载GPU上，在model(data.cuda())部分控制

                # 等待数据异步传输完成
                #torch.cuda.synchronize()
                ##print('train_iterator:', train_iterator)
                #batch = next(train_iterator).cuda(local_rank)

                #batch_size = max(batch.protein_element_batch) + 1
                #print('batch_size:', batch_size)
                #batch = next(train_iterator)
                ##print('batch:', batch)
                #exit()
                #exit()
                #给蛋白加噪？
                #protein_noise = torch.randn_like(batch.protein_pos) * config.train.pos_noise_std
                #gt_protein_pos = batch.protein_pos + protein_noise
                
                step_loss = defaultdict(list)
                for step in np.array(range(GP.final_timesteps)): #指定步长
                    st = time.perf_counter()
                    for i in range(grad_num): #批量太小做梯度累计
                        #print('batch:', i)
                        #print('batch_size:', batch_size)
                        #给蛋白加噪？为了让生成的配体更稳定
                        #protein_noise = torch.randn_like(batch.protein_pos) * config.train.pos_noise_std #蛋白噪音，先不加
                        #gt_protein_pos = batch.protein_pos + protein_noise
                        #gt_protein_pos = batch.protein_pos

                    #前向使用混合精度，后向不使用
                    #with torch.autocast(device_type='cuda'): #混合精度可能导致梯度消失问题
                    #with torch.cuda.amp.autocast():
                        #速度慢是因为数据加载到GPU的方法不合理，不要一下子把数据全部加载到GPU,需要什么就手动加载什么到GPU
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
                                total_training_steps=GP.consistency_training_steps,
                                
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

                                    
                                cross_bond_index = b_link_e.T,
                                cross_bond_type = b_cross_bond_type, 
                                cross_bond_index_reverse = b_link_e_reverse.T, 
                                cross_bond_type_reverse = b_cross_bond_type_reverse,

                                protein_coords_predict = b_protein_coords_predict,
                                complex_mol = None,

                                protein_element_batch = b_protein_element_batch,
                                protein_link_t_batch = b_protein_link_t_batch,
                                protein_link_t_reverse_batch = b_protein_link_t_reverse_batch,
                                ligand_element_batch = b_ligand_element_batch,

                                rd_pos = b_rd_pos,
                                
                                #ligand_emb = b_ligand_emb,
                                #pocket_emb = b_pocket_emb,



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

                                cross_bond_index = b_link_e.T,
                                cross_bond_type = b_cross_bond_type, 
                                cross_bond_index_reverse = b_link_e_reverse.T, 
                                cross_bond_type_reverse = b_cross_bond_type_reverse,

                                protein_coords_predict = b_protein_coords_predict,
                                complex_mol = None,

                                protein_element_batch = b_protein_element_batch,
                                protein_link_t_batch = b_protein_link_t_batch,
                                protein_link_t_reverse_batch = b_protein_link_t_reverse_batch,
                                ligand_element_batch = b_ligand_element_batch,

                                rd_pos = b_rd_pos,
                                
                                #ligand_emb = b_ligand_emb,
                                #pocket_emb = b_pocket_emb,
                            )
            
                        
            

                        if args.value_only:
                            results['loss'] = results['loss_exp']
                            
                        loss, loss_pos, loss_v, loss_exp, loss_dismat, loss_bond, loss_angle, loss_dihedral, rmsd = results['loss'], results['loss_pos'], results['loss_v'],\
                            results['loss_exp'], results['loss_dismat'], results['loss_bond'], results['loss_angle'], results['loss_dihedral'], results['rmsd'],
                        loss = loss / grad_num #n_acc_batch == 1。如果批量小，则作梯度累计，假如2个batch一次梯度更新，则可以选择在每次迭代之后，损失除以2。 loss_angle, loss_dihedral
                        #当然也可以不立刻梯度更新，而是将来个批量的损失相加，再取均值，最后执行一次梯度更新。前者梯度更新了2次，但损失缩小1/2， 后者梯度更新一次
                        '''
                        step_loss['loss'].append(loss.item())
                        step_loss['loss_pos'].append(loss_pos.item() / grad_num)
                        step_loss['loss_v'].append(loss_v.item() / grad_num)
                        step_loss['loss_exp'].append(loss_exp.item() / grad_num)
                        step_loss['loss_dismat'].append(loss_dismat.item() / grad_num) #梯度累积之后，再取损失均值
                        step_loss['loss_bond'].append(loss_bond.item() / grad_num)
                        step_loss['loss_angle'].append(loss_angle.item() / grad_num)
                        step_loss['loss_dihedral'].append(loss_dihedral.item() / grad_num)
                        step_loss['rmsd'].append(rmsd.item() / grad_num)\
                        '''

                        ##print('loss:', loss)
                        #exit()
                    loss.backward()

                    
                    
                    #梯度更新，如果之后不执行梯度下降，则梯度会累积. 如果想在Consistency上累积梯度，则需要在每一步下面，重复训练
                    #loss.backward()

                    #如果一直是梯度爆炸，则可能要改一下梯度修剪办法，如果想累积梯度，则应该调整该位置
                    orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)

                    #参数更新，梯度下降，如果想累积梯度，则应该调整该位置
                    optimizer.step()
                    
                    # 清空梯度，如果想累积梯度，则应该调整该位置
                    optimizer.zero_grad(set_to_none=True)  #要么放在loss.backward()前面，要么放在optimizer.step()后面


                    # 释放显存，可能降低速度
                    #torch.cuda.empty_cache()

                    #无论是否使用ema，都可以更新它
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
                    update_ema_model(ema_model, model,ema_decay_rate) #速度慢不是这里的问题
                    #end2 = time.time()
                    #print('update_ema_model time s:', round(end2 - st2, 4))
                    
                    end = time.perf_counter()
                    #print('a batch time s:', round(end - st, 4)) #3.2, 模型只用了0.4,包括online和emamodel

                    
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
                                'rate': best_rate,
                                'iter': best_iter,
                            }, ckpt_path)
                    
                    
                    if torch.distributed.get_rank() == 0:
                        logger.info(
                            '[Train] Step %d | iter %d | subdata %d | best_iter %d | best_rate %.6f | Loss %.6f (pos %.6f | v %.6f | exp %.6f | dismat %.6f | bond %.6f | angle %.6f | dihedral %.6f | rmsd %.6f)' % (
                                step, it, sub_id, best_iter, best_rate, loss, loss_pos, loss_v, loss_exp, loss_dismat, loss_bond, loss_angle, loss_dihedral, rmsd
                            ))
                    
                    
                '''        
                if torch.distributed.get_rank() == 0:
                    logger.info(
                        '[Train] Iter %d | subdata %d | Loss %.6f (pos %.6f | v %.6f | exp %.6f | dismat %.6f | bond %.6f | angle %.6f | dihedral %.6f | rmsd %.6f)' % (
                            batch_id + num + 1, sub_id, np.mean(step_loss['loss'][-1]), np.mean(step_loss['loss_pos'][-1]), np.mean(step_loss['loss_v'][-1]), np.mean(step_loss['loss_exp'][-1]), 
                            np.mean(step_loss['loss_dismat'][-1]), np.mean(step_loss['loss_bond'][-1]), np.mean(step_loss['loss_angle'][-1]), np.mean(step_loss['loss_dihedral'][-1]), np.mean(step_loss['rmsd'][-1]), 
                        )
                    )
                '''



        #if torch.distributed.get_rank() == 0:
        for k, v in results.items():
            if torch.is_tensor(v) and v.squeeze().ndim == 0:
                writer.add_scalar(f'train/{k}', v, it)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
        writer.add_scalar('train/grad', orig_grad_norm, it)
        writer.flush()




    def evaluate():
        model.eval()
        #对测试集进行生成采样，计算评估指标
        #exclude_keys = ['protein_cross_distance']
        collate_exclude_keys = ['ligand_nbh_list']
        #采样生成
        for step in [GP.final_timesteps][:]: #不能循环了，是什么情况？出现加载配置文件时编码错误
            '''
                CUDA_VISIBLE_DEVICES=0 python scripts/sample_diffusion.py --config ./configs/sampling.yml -i 0 --guide_mode pdbbind_random \
                    --type_grad_weight 100 --pos_grad_weight 25 --result_path ./cd2020_pro_0_res
            '''
            parser = argparse.ArgumentParser()
            parser.add_argument('--config', type=str, default='./configs/sampling.yml')
            parser.add_argument('-i', '--data_id', type=int, default=0) #数据开始的位置
            parser.add_argument('--device', type=str, default='cuda')
            parser.add_argument('--batch_size', type=int, default=5)
            parser.add_argument('--guide_mode', type=str, default='pdbbind_random', choices=['joint', 'pdbbind_random', 'vina', 'valuenet', 'wo'])  
            parser.add_argument('--type_grad_weight', type=float, default=100) #注意下，这里的权重如何使用的
            parser.add_argument('--pos_grad_weight', type=float, default=25)
            parser.add_argument('--result_path', type=str, default='./new_ecdock_step15') #分子生成的路径
            parser.add_argument('--log_name', type=str, default='')
            parser.add_argument('--data_flag', type=str, default='new_test', choices=['new_test', 'old_test'], help = 'use new or old data test') 
            parser.add_argument('--data_name', type=str, default='posebustersv1', choices=['posebusters_glide', 'posebustersv1', 'posebustersV2', 'pdbbind2020', 'pdbbind2020_r10'])
            parser.add_argument('--sample_num', type=int, default=500)
            parser.add_argument('--diffusion', type=str, default='cm', choices=['cm', 'ddpm'])
            parser.add_argument('--gnn', type=str, default='equiformer', choices=['equiformer', 'egnn', 'escn'])
            parser.add_argument('--rm_outdir', action='store_false') #默认是删除的，即如果不指定这个参数
            parser.add_argument('--conf_num', type=int, default=20)
            parser.add_argument('--test_name', type=str, default='cm_equiformer_gen')
            parser.add_argument('--si', type=int, default=0)
            parser.add_argument('--ei', type=int, default=500)
            parser.add_argument('--out_dir', type=str, default='')
            parser.add_argument('--step', type=int, default=None)
            parser.add_argument('--ckpt', type=str, default=None)


            test_args = parser.parse_args([
            '--config', './configs/sampling.yml',
            '-i', '0',
            '--device', 'cuda',
            '--batch_size', '5',
            '--guide_mode', 'pdbbind_random',
            '--type_grad_weight', '100',
            '--pos_grad_weight', '25',
            '--result_path', './new_ecdock_step15',
            '--log_name', '',
            '--data_flag', 'new_test',
            '--data_name', 'posebustersv1',
            '--sample_num', '500',
            '--diffusion', 'cm',
            '--gnn', 'equiformer',
            #'--rm_outdir',
            '--conf_num', '20',
            '--test_name', 'cm_equiformer_gen',
            '--si', '0',
            '--ei', '500',
            '--out_dir', '',
            '--step', '5',
            '--ckpt', '',

            
            ]) #参数一定要放在循环内，这样保证每一步的采样的命令行参数一样


            sample_num = test_args.sample_num
            data_flag = test_args.data_flag        #posebusters数据集，则执行这个，否则设置为None或者不指定
            data_name = test_args.data_name     #注意，新的数据集要和标志位一起修改
            if data_flag == 'new_test':
                if data_name == ['posebustersv1', 'posebustersv2','posebusters_glide']:
                    GP.max_atoms = 64 #根据不同的数据集，设置配体的最大原子数量

            data_path  = f'../CrossDocked2020/data/{data_name}/' #如果要使用新的数据集，把这里的"posebusters"给替换掉即可
            data_split = f'../CrossDocked2020/data/{data_name}/{data_name}_split.pt' 


            GP.final_timesteps = step
            GP.consistency_training_steps = step
            output_dir = f'../EcDock_sample_dir/train_evaluate'
            sample_main(output_dir, data_flag, sample_num, data_path, data_split, data_name, test_args, batch_size = test_args.batch_size)

            
            #把模型复制一份
            os.makedirs(os.path.join(output_dir, 'model'), exist_ok=True)
            shutil.copyfile(test_args.config, os.path.join(output_dir, 'model', os.path.basename(test_args.config)))
            shutil.copytree('./models', os.path.join(output_dir, 'model', 'models'), dirs_exist_ok=True)  
            shutil.copytree('./KGDiff', os.path.join(output_dir, 'model', 'KGDiff'), dirs_exist_ok=True)
            shutil.copytree('./EcConf', os.path.join(output_dir, 'model', 'EcConf'), dirs_exist_ok=True)
            shutil.copytree('./configs', os.path.join(output_dir, 'model', 'configs'), dirs_exist_ok=True)
            shutil.copytree('./ocp', os.path.join(output_dir, 'model', 'ocp'), dirs_exist_ok=True)






        for step in [GP.final_timesteps][:]:
            #评估rmsd
            test_model = 'ecdock'
            data_name = test_args.data_name  #pdb2020
            gnn  = test_args.gnn  #ecdock时，采用不同的神经网络, equiformer
            diffusion = test_args.diffusion #ecdock时，采用不同的扩散模型， CM/DDPM
            mode = '' #不用赋值
            if test_model == 'ecdock':    #posebusters_ecdock_cm_equiformer_step1
                test_model_name = 'train_evaluate' #这个目录下，可以存放是sdf也可以是pickle，路径别忘了改
                file_path  = f'../EcDock_sample_dir/{test_model_name}' #记得改名字
                step = step - 1
                #file_path  = f'../EcDock_sample_dir/train_evaluate'

            name_list = []
            with open(f'../CrossDocked2020/data/{data_name}/{data_name}_name.txt') as f:
                for i in f:
                    name_list.append(i.strip())

            #效果差的，复杂的分子，为了节约时间，我们平常评估这些即可
            poor_name_list = []
            with open(f'../CrossDocked2020/data/{data_name}/{data_name}_poor_name.txt') as f:
                for i in f:
                    poor_name_list.append(i.strip())
                
            #读取配体的sdf文件,truth_mol是一维list，gen_mol是一个2维度list，存放整个测试集的结果
            truth_mol, gen_mol, data_name_list = read_file(file_path, mode, flag = 'sdf', num = 10000  + 2, step = step, model = test_model, name_list = name_list, poor_name_list = poor_name_list)  #读取所有数据，并转化成rdkit mol对象, step值别忘了改
            print('truth_mol, gen_mol:', len(truth_mol), len(gen_mol))
            assert len(truth_mol) == len(gen_mol)
            

            #计算rmsd。从生成的40个分子的中随机选择1/3/5/10/40的，拿过来看rmsd成功率
            resault_dict = {}
            boxplot_data_list = [] #保留1,5,40结果用于绘制箱线图
            for num in [1, 3, 5, 10, 25, 40, test_args.conf_num][:]:
                data_dict = rmsds(truth_mol, gen_mol, num, data_name_list) #对于每一条数据，随机挑选num个进行测试
                resault_dict[num] = ['rate, rate_min, rate_mean, rmsd_mean, rmsd_std, rsmd_mid, rmsd_max, rmsd_min:', data_dict['all']]
                rate = data_dict['all'][2]
                if num in [1, 5, 10, 25, 40, test_args.conf_num]:
                    boxplot_data_list.append(data_dict['data'])

            #exit()
            print(resault_dict)
            #保存字典为JSON文件
            #path = 'resault'
            path = file_path
            os.makedirs(path, exist_ok=True)

            file_name = f'{test_model_name}_evaluate_resault.json'
            with open(os.path.join(path, file_name), 'w') as file:
                json.dump(resault_dict, file, indent=4)
            

            #绘制箱线图
            save_path = os.path.join(path, f'{test_model_name}_boxplot.png')
            boxplot(boxplot_data_list, save_path, test_model_name)

        return rate
    
    
    
    
    
    
    '''#训练模型'''
    
    
    try:
        best_loss, best_iter = 0.0, 0
        best_rate = 0.70
        
        
        if args.ckpt:
            try:
                best_rate = ckpt['rate']
            except Exception:
                best_rate = 0.70
            try:
                best_iter = ckpt['iter']
            except Exception:
                best_iter = 0
        else:
            best_rate = 0.70
        
        
        if args.epoch:
            end_epoch = args.epoch
        else:
            end_epoch = config.train.max_iters

        
        if GP.data_type in ['BindingNetv2_High']:
            for it in range(start_it, end_epoch):
                #这里的it，我们现在将其视为epoch
                #with torch.autograd.detect_anomaly():
                #使用 torch.autograd.detect_anomaly() 可以帮助您在训练过程中及时发现梯度异常，并在发现异常时抛出异常，从而帮助您及时调试和解决问题
                data_len = 0
                with open(f'{config.data.path}/{os.path.basename(config.data.path)}_name.txt') as f:
                    for i in f:
                        data_len += 1
                
                word_size = 4000
                sep = int(data_len / word_size) + 1
                for i in range(0, sep):
                    st_i  = i * word_size
                    end_i = (i + 1) * word_size
                    subset_train = load_data(st_i, end_i)
                    for sub_id, train_set in enumerate(subset_train):
                        # 加载数据集
                        #transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
                        #train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
                        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle = True, drop_last=True, num_replicas=world_size, rank=local_rank) #打乱数据在sample控制
                        #train_sampler = torch.utils.data.distributed.DistributedSampler(train_set) #打乱数据在sample控制

                        #print('train_set[0] type:', type(train_set[0])) # <class 'KGDiff.datasets.pl_data.ProteinLigandData'>
                        #print('train_set[0]', train_set[0])
                        #print('train_set[0].keys()', train_set[0].keys())
                        
                        train_loader =DataLoader(
                            train_set,
                            batch_size=config.train.batch_size,
                            shuffle=False, #有了sampler，则数据的打乱不要在DataLoader设置了，由sampler控制。所以这里shuffle必须为False
                            drop_last=True,
                            #num_workers=config.train.num_workers,
                            num_workers=10,
                            follow_batch=FOLLOW_BATCH,  #FOLLOW_BATCH = ('protein_element', 'ligand_element', 'ligand_bond_type',) #对这些数据形成批量id，命名为数据名_batch
                            exclude_keys=collate_exclude_keys,  #表示在组合批量时候，把邻接表给排除。问题是如果排除邻接表，按怎么批量传递给GNN邻接表？这一步操作在哪？
                            sampler=train_sampler,
                            pin_memory=True,
                            prefetch_factor=10, #预加载内存，可以扩大点
                            #collate_fn=custom_collate, #这种事针对pyg.Data类的数据的，但现在的传递的是自定义的类，不起作用 <class 'KGDiff.datasets.pl_data.ProteinLigandData'>
                            #exclude_keys = exclude_keys
                        )

                        train_loader = iter(train_loader)
                        ##在PYG dataloader中collate_fn参数是被删除的，所以不起作用，而exclude_keys成了关键参数，因此如果想不连接某些数据对象，只需要提供exclude_keys即可

                        train_sampler.set_epoch(it)

                        #速度慢是因为数据加载到GPU的方法不合理，不要一下子把数据全部加载到GPU,需要什么就手动加载什么到GPU
                        #目前存在的问题在于，数据的每一个批量不是等长的，导致并行的进程的进度不一样，久而久之就卡住了
                        ckpt_path = os.path.join(ckpt_dir, 'final.pt')

                        epoch_batch_num = math.ceil(args.train_set_num / config.train.batch_size) #训练集有多少个batch
                        #assert len(train_loader) == len(train_set) / math.ceil(config.train.batch_size) #train_loader是无法通过len来获取批量数量的，这是迭代器
                        sub_batch_num   = math.ceil(len(train_set) / config.train.batch_size)
                        batch_id        = it * epoch_batch_num + sub_id * sub_batch_num
                        print('epoch_batch_num, sub_batch_num:', epoch_batch_num, sub_batch_num)
                        
                        train(local_rank, args, it, sub_id, train_loader, ckpt_path, batch_id) #保存每一个batch的损失
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
                                'rate': best_rate,
                                'iter': best_iter,
                            }, ckpt_path)

                        #del train_loader
                        #gc.collect()
                        #if len(subset_train) > 1:
                            #dist.barrier() #用于同步信息，需要加上，尤其是变长数据集的PYG格式，否则容易出问题，程序容易卡住
                        
                        dist.barrier()

                    if local_rank == 0:
                        #先保存模型
                        ckpt_path = os.path.join('test_premodel', 'final.pt')

                        #保证在评估的时候，模型当前评估的模型参数不变，所以使用copy.deepcopy()
                        save_dict = {
                            'config': config,
                            'model': copy.deepcopy(model.state_dict()),
                            'ema_model': copy.deepcopy(ema_model.state_dict()),
                            'optimizer': copy.deepcopy(optimizer.state_dict()),
                            'scheduler': copy.deepcopy(scheduler.state_dict()),
                            'iteration': it,
                            'args': args,
                            'equiformer': config.equiformer,
                            'escn': config.escn,
                        }
                        torch.save(save_dict, ckpt_path)

                    dist.barrier()
                if it > 5 and it % 1 == 0:
                    time.sleep(30)
                    rate = evaluate()
                dist.barrier() #等待数据同步，没必要
                    
                if it > 5 and it % 1 == 0:  
                    #对多卡生成的结果，进行评估  
                    if local_rank == 0:
                        file_path  = f'../EcDock_sample_dir/train_evaluate'
                        rate = train_evaluate(file_path = file_path) #不生成，只评估
                        if best_rate is None or rate > best_rate:
                            if torch.distributed.get_rank() == 0:
                                logger.info(f'[Validate] Best val rate achieved: {rate:.6f}')
                            best_rate, best_iter = rate, it
                            ckpt_path = os.path.join(ckpt_dir, 'rate_%d.pt' % it)

                            #new add rate
                            save_dict['rate'] = best_rate
                            save_dict['iter'] = best_iter
                            torch.save(save_dict, ckpt_path)
                        else:
                            if torch.distributed.get_rank() == 0:
                                logger.info(f'[Validate] Val rate is not improved. '
                                            f'Best val rate: {best_rate:.6f} at iter {best_iter}, current rate is {rate}')


                    dist.barrier() #等待数据同步，没必要
        else:
            subset_train = load_data(0, 1000000000, num = 40000000)
            for it in range(start_it, end_epoch):
                for sub_id, train_set in enumerate(subset_train):
                    # 加载数据集
                    #transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
                    #train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
                    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle = True, drop_last=True, num_replicas=world_size, rank=local_rank) #打乱数据在sample控制
                    #train_sampler = torch.utils.data.distributed.DistributedSampler(train_set) #打乱数据在sample控制

                    #print('train_set[0] type:', type(train_set[0])) # <class 'KGDiff.datasets.pl_data.ProteinLigandData'>
                    #print('train_set[0]', train_set[0])
                    #print('train_set[0].keys()', train_set[0].keys())
                    
                    train_loader =DataLoader(
                        train_set,
                        batch_size=config.train.batch_size,
                        shuffle=False, #有了sampler，则数据的打乱不要在DataLoader设置了，由sampler控制。所以这里shuffle必须为False
                        drop_last=True,
                        #num_workers=config.train.num_workers,
                        num_workers=10,
                        follow_batch=FOLLOW_BATCH,  #FOLLOW_BATCH = ('protein_element', 'ligand_element', 'ligand_bond_type',) #对这些数据形成批量id，命名为数据名_batch
                        exclude_keys=collate_exclude_keys,  #表示在组合批量时候，把邻接表给排除。问题是如果排除邻接表，按怎么批量传递给GNN邻接表？这一步操作在哪？
                        sampler=train_sampler,
                        pin_memory=True,
                        prefetch_factor=10, #预加载内存，可以扩大点
                        #collate_fn=custom_collate, #这种事针对pyg.Data类的数据的，但现在的传递的是自定义的类，不起作用 <class 'KGDiff.datasets.pl_data.ProteinLigandData'>
                        #exclude_keys = exclude_keys
                    )

                    train_loader = iter(train_loader)
                    ##在PYG dataloader中collate_fn参数是被删除的，所以不起作用，而exclude_keys成了关键参数，因此如果想不连接某些数据对象，只需要提供exclude_keys即可

                    train_sampler.set_epoch(it)

                    #速度慢是因为数据加载到GPU的方法不合理，不要一下子把数据全部加载到GPU,需要什么就手动加载什么到GPU
                    #目前存在的问题在于，数据的每一个批量不是等长的，导致并行的进程的进度不一样，久而久之就卡住了
                    ckpt_path = os.path.join(ckpt_dir, 'final.pt')

                    epoch_batch_num = math.ceil(args.train_set_num / config.train.batch_size) #训练集有多少个batch
                    #assert len(train_loader) == len(train_set) / math.ceil(config.train.batch_size) #train_loader是无法通过len来获取批量数量的，这是迭代器
                    sub_batch_num   = math.ceil(len(train_set) / config.train.batch_size)
                    batch_id        = it * epoch_batch_num + sub_id * sub_batch_num
                    print('epoch_batch_num, sub_batch_num:', epoch_batch_num, sub_batch_num)
                    
                    train(local_rank, args, it, sub_id, train_loader, ckpt_path, batch_id) #保存每一个batch的损失
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
                            'rate': best_rate,
                            'iter': best_iter,
                        }, ckpt_path)

                    #del train_loader
                    #gc.collect()
                    #if len(subset_train) > 1:
                        #dist.barrier() #用于同步信息，需要加上，尤其是变长数据集的PYG格式，否则容易出问题，程序容易卡住
                    
                    dist.barrier()

                if local_rank == 0:
                    #先保存模型
                    ckpt_path = os.path.join('test_premodel', 'final.pt')

                    #保证在评估的时候，模型当前评估的模型参数不变，所以使用copy.deepcopy()
                    save_dict = {
                        'config': config,
                        'model': copy.deepcopy(model.state_dict()),
                        'ema_model': copy.deepcopy(ema_model.state_dict()),
                        'optimizer': copy.deepcopy(optimizer.state_dict()),
                        'scheduler': copy.deepcopy(scheduler.state_dict()),
                        'iteration': it,
                        'args': args,
                        'equiformer': config.equiformer,
                        'escn': config.escn,
                    }
                    torch.save(save_dict, ckpt_path)
                dist.barrier()
                
                if it > 200 and it % 1 == 0:
                    rate = evaluate()
                dist.barrier() #等待数据同步，没必要
                if it > 200 and it % 1 == 0:
                    #对多卡生成的结果，进行评估  
                    if local_rank == 0:
                        file_path  = f'../EcDock_sample_dir/train_evaluate'
                        rate = train_evaluate(file_path = file_path) #不生成，只评估
                        if best_rate is None or rate > best_rate:
                            if torch.distributed.get_rank() == 0:
                                logger.info(f'[Validate] Best val rate achieved: {rate:.6f}')
                            best_rate, best_iter = rate, it
                            ckpt_path = os.path.join(ckpt_dir, 'rate_%d.pt' % it)

                            #new add rate
                            save_dict['rate'] = best_rate
                            save_dict['iter'] = best_iter
                            torch.save(save_dict, ckpt_path)
                        else:
                            if torch.distributed.get_rank() == 0:
                                logger.info(f'[Validate] Val rate is not improved. '
                                            f'Best val rate: {best_rate:.6f} at iter {best_iter}, current rate is {rate}')

                    if rate >= 0.82:
                        break
                    
                    dist.barrier() #等待数据同步，没必要
                    
    except KeyboardInterrupt:
        print('Terminating...')

    
    dist.destroy_process_group()
        

if __name__ == '__main__':

    #['cudagraphs', 'inductor', 'onnxrt', 'openxla', 'openxla_eval', 'tvm']
    #main = torch.compile(main, mode="max-autotune", dynamic=True, fullgraph=True, backend='inductor') #torch.compile在2.0后才能用
    #torch.compile更多是面向model，而不是其非神经网络对象，否则可能会报错不支持

    ##在PYG dataloader中collate_fn参数是被删除的，所以不起作用，而exclude_keys成了关键参数，因此如果想不连接某些数据对象，只需要提供exclude_keys即可
    #PYG dataloader要排除连接的数据对象
    #exclude_keys = ['protein_cross_distance']
    collate_exclude_keys = ['ligand_nbh_list']
    best_rate, best_iter = 0, 0
    main()
