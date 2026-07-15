import numpy as np
import torch as th
from joblib import Parallel, delayed
import pandas as pd
import argparse
import os, sys
import MDAnalysis as mda
#sys.path.append("/home/shenchao/resdocktest2/rtmscore2")
sys.path.append(os.path.abspath(__file__).replace("rtmscore.py",".."))
from torch.utils.data import DataLoader
from EcDockScore.data.data import VSDataset
from EcDockScore.model.utils import collate, run_an_eval_epoch
from EcDockScore.model.model2 import EcDockScore, DGLGraphTransformer #LigandNet, TargetNet
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

#you need to set the babel libdir first if you need to generate the pocket
#os.environ["BABEL_LIBDIR"] = "/data/fan_zg/anaconda3/envs/cu128/lib/openbabel/3.1.0"
from openbabel import openbabel as ob

import copy
from rdkit import Chem
from rdkit.Chem import AllChem
import copy
import subprocess
import time
import multiprocessing

from rtmscore import one_data, is_file_exist_and_not_empty

from tqdm import tqdm


        
    
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default = '/data/fan_zg/MDocking/VSDS_ECDock_Gen', help='Input data dir')
    
    p.add_argument('--data_name', default = '/data/fan_zg/MDocking/data_name.txt', help='Input data name')

    p.add_argument('--refine_flag', default = 'norefine', help='Glide refine')

    p.add_argument('-p','--prot', default = None,
                                    help='Input protein file (.pdb)')


    p.add_argument('-l','--lig', default = None,
                                    help='Input ligand file (.sdf/.mol2)')

    p.add_argument('-m','--model', default="../trained_models/rtmscore_model1.pth",
                                    help='trained model path (default: "../trained_models/rtmscore_model1.pth")')

    p.add_argument('-o','--outprefix', default="out",
                                    help='the prefix of output file (default: "out")，')
    p.add_argument('-gen_pocket','--gen_pocket', action="store_true", default=False,
                                    help='whether to generate the pocket，，，。，')

    p.add_argument('-c','--cutoff', default=10.0, type=float,
                                    help='the cutoff the define the pocket and interactions within the pocket (default: 10.0)，')

    p.add_argument('-rl','--reflig', default = None,
                help='the reference ligand to determine the pocket(.sdf/.mol2)， ，。，')

    p.add_argument('-pl', '--parallel', default=False, action="store_true",
                        help='whether to obtain the graphs in parallel (When the dataset is too large,\
                        it may be out of memory when conducting the parallel mode).')

    p.add_argument('-ac', '--atom_contribution', default=False, action="store_true",
                        help='whether to obtain the atom contrubution of the score.')

    p.add_argument('-rc', '--res_contribution', default=False, action="store_true",
                        help='whether to obtain the residue contrubution of the score.')
    
    
    p.add_argument('--st_i', default=0, type=int)
    p.add_argument('--en_i', default=-1, type=int)
    
    p.add_argument('--cover', default=0, type=int)

    inargs = p.parse_args()
    #if inargs.gen_pocket:
        #if inargs.reflig is None:
            #raise ValueError("if pocket is generated, the reference ligand should be provided.")

    #，，。
    if inargs.atom_contribution and inargs.res_contribution:
        raise ValueError("only one of the atom_contribution and res_contribution can be supported")
    

    dt_dir = inargs.data_dir
    
    success_count = 0
    for dt in tqdm(list(sorted(list(os.listdir(dt_dir))))[:]):
        path = os.path.join(dt_dir, dt)
        if os.path.exists(path) and os.path.isdir(path) and os.listdir(path) and 'model' not in dt: #
            if inargs.gen_pocket:
                p_file = os.path.join(path, f'{dt}_protein.pdb') #
            else:
                p_file = os.path.join(path, f'{dt}_protein_400.pdb') 
            if inargs.refine_flag == 'refine':
                l_file = os.path.join(path, f'glide_refine_gen_{dt}_ligand.sdf')  #glide_refine_gen_active0.sdf
                if not os.path.exists(l_file):
                    print('refine is ')
                    #l_file = os.path.join(path, f'gen_{dt}_ligand.sdf')
                    continue
            else: 
                '''
                l_file = os.path.join(path, f'mmff_gen_{dt}_ligand.sdf') 
                if not os.path.exists(l_file):
                    l_file = os.path.join(path, f'gen_{dt}_ligand.sdf') 
                '''
                l_file = os.path.join(path, f'gen_{dt}_ligand.sdf')
            

                    
                
            #rmt，posebuses
            origin_gen_l_file = l_file
            #print('origin_gen_l_file:', origin_gen_l_file)
            if inargs.refine_flag == 'refine':
                rtm_score_l_file   = os.path.join(path, f'refine_rtm_sort_gen_{dt}_ligand.sdf')
            else:
                rtm_score_l_file   = os.path.join(path, f'rtm_sort_gen_{dt}_ligand.sdf')
            
            ref_lig_file = l_file
            
            if inargs.refine_flag == 'refine':
                if inargs.atom_contribution:
                    out_file = os.path.join(path, f'refine_{dt}_rtmscore_ac')
                elif inargs.res_contribution:
                    out_file = os.path.join(path, f'refine_{dt}_rtmscore_rc')
                else:
                    out_file = os.path.join(path, f'refine_{dt}_rtmscore')
            else:
                if inargs.atom_contribution:
                    out_file = os.path.join(path, f'{dt}_rtmscore_ac')
                elif inargs.res_contribution:
                    out_file = os.path.join(path, f'{dt}_rtmscore_rc')
                else:
                    out_file = os.path.join(path, f'{dt}_rtmscore')
                
            
            
            
            
            try:
                print('dt_name:', dt_dir)
                ligand_num = len(list(Chem.rdmolfiles.SDMolSupplier(l_file)))
                
                if is_file_exist_and_not_empty(out_file + '.csv'):
                    nums = 0
                    with open(out_file + '.csv') as f:
                        for i in f:
                            nums += 1
                            
                    if inargs.cover:
                        print('cover, rewrite')
                        one_data(p_file, l_file, ref_lig_file, out_file, origin_gen_l_file, rtm_score_l_file, inargs)
                        
                    elif ligand_num != nums: 
                        print(f'{ligand_num} != {nums}')
                        one_data(p_file, l_file, ref_lig_file, out_file, origin_gen_l_file, rtm_score_l_file, inargs)
                    
                    else:
                        print(f" {out_file}.csv ")
                        success_count += 1  
                        continue
                else:
                    #print('，')
                    one_data(p_file, l_file, ref_lig_file, out_file, origin_gen_l_file, rtm_score_l_file, inargs)
                
                success_count += 1
            except Exception as e:
                print('error:', e) #error: The graph of pocket cannot be generated
            
            
            #one_data(p_file, l_file, ref_lig_file, out_file, origin_gen_l_file, rtm_score_l_file, inargs)
        
    print('success_count:', success_count)



