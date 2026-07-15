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




def set_seed(seed):
    torch.manual_seed(seed)  #  PyTorch 
    torch.cuda.manual_seed_all(seed)  #  GPU 
    np.random.seed(seed)  #  NumPy 
    random.seed(seed)  #  Python 
    torch.backends.cudnn.deterministic = True  #  CuDNN 
    torch.backends.cudnn.benchmark = True



'''
：sdfpdb，，。rdkit molpickle，
sdf
'''



def rmsds_rdkit(truth_mol_list, gen_mol_list, num):
    #truth_mollist，gen_mol2list，
    #，num
    #print('truth_mol_list:', len(truth_mol_list))
    #print('gen_mol_list:', gen_mol_list)
    index       = list(range(40)) #40，num
    #print('index:', index)
    data_dict = {}
    try:
        tg_index    = random.sample(index, num) #diffdock，，40，，，
        truth_mols  = truth_mol_list #ground truth，
        gen_mols    = []

        for mols in gen_mol_list:
            #sub_mol = [mols[i] for i in tg_index]
            random.shuffle(mols)
            sub_mol = mols[:num] #，num
            gen_mols.append(sub_mol)
    except Exception as e:
        truth_mols  = truth_mol_list #ground truth，
        gen_mols    = gen_mol_list


    rmsd_dict = defaultdict(list)      #
    rmsd_list = []
    rmsd_list_per = []  #，list
    for i, mols in enumerate(gen_mols):
        tmp = []
        for mol in mols:
            try:
                #rsmd, ，GetBestRMS，rmsd
                #AllChem.GetBestRMS，rmsd
                #，，rmsd，
                #https://www.rdkit.org/docs/source/rdkit.Chem.rdMolAlign.html#rdkit.Chem.rdMolAlign.GetBestRMS
                rmsd = AllChem.GetBestRMS(Chem.RemoveHs(mol), Chem.RemoveHs(truth_mols[i]))
                #rmsd = Chem.rdMolAlign.GetBestRMS(Chem.RemoveHs(mol), Chem.RemoveHs(truth_mols[i]))  

                #rmsd = Chem.rdMolAlign.AlignMol(Chem.RemoveHs(mol), Chem.RemoveHs(truth_mols[i])) #AllChem.GetBestRMS()

                rmsd = rmsd * np.sqrt(3)

                #rmsd
                #pre_pos  = Chem.RemoveHs(mol).GetConformer(0).GetPositions()
                #holo_pos = Chem.RemoveHs(truth_mols[i]).GetConformer(0).GetPositions()
                #rmsd = np.sqrt(np.sum((pre_pos - holo_pos) ** 2) / pre_pos.shape[0]) # unimolrmsd，np.sum，, 
            except Exception as e:
                print('error:', e)
                #print('mol:', mol.GetConformer(0).GetPositions())
                #print('truth_mols:', truth_mols[i].GetConformer(0).GetPositions())
                continue
            tmp.append(rmsd)
            rmsd_list.append(rmsd)
        if tmp:
            rmsd_list_per.append(tmp)

            #num
            rmsd_dict['rmsd_mean'].append(np.mean(tmp))
            rmsd_dict['rmsd_std'].append(np.std(tmp))
            rmsd_dict['rsmd_mid'].append(np.median(tmp))
            rmsd_dict['rmsd_max'].append(np.max(tmp))
            rmsd_dict['rmsd_min'].append(np.min(tmp))

            #rmsd2，
            np_rmsd = np.array(tmp)
            all_num = np_rmsd.shape[0] #，num
            indices = np_rmsd <= 2 
            sub_num = np.count_nonzero(indices)
            rmsd_dict['rmsd_rate'].append(sub_num / all_num)

            print('sub_num / all_num:', sub_num / all_num)
            #print('np.mean(np_rmsd <= 2):', np.mean(np_rmsd <= 2)) #



    print('all num:', len(truth_mol_list) * num)
    print('rmsd_list num:', len(rmsd_list))

    #num，100, 
    rmsd_mean = round(np.mean(rmsd_dict['rmsd_mean']), 4)
    rmsd_std  = round(np.mean(rmsd_dict['rmsd_std']), 4)
    rsmd_mid  = round(np.mean(rmsd_dict['rsmd_mid']),4)
    rmsd_max  = round(np.mean(rmsd_dict['rmsd_max']), 4)
    rmsd_min  = round(np.mean(rmsd_dict['rmsd_min']), 4)
    rmsd_rate      = round(np.mean(rmsd_dict['rmsd_rate']), 4)
    #print('new_rmsd_rate:', round(np.sum(np.array(rmsd_list) <= 2), 4))

    data_dict['data_per']   =  rmsd_list_per #，numpy
    data_dict['data']       =  np.array(rmsd_list)
    data_dict['all']        = [rmsd_rate, rmsd_mean, rmsd_std, rsmd_mid, rmsd_max, rmsd_min]

    return data_dict




def rmsds(truth_mol_list, gen_mol_list, num, data_name_list, model = None, random_flag = False):
    #truth_mollist，gen_mol2list，
    #，num
    #print('truth_mol_list:', len(truth_mol_list))
    #print('gen_mol_list:', gen_mol_list)
    index       = list(range(40)) #40，num
    #print('index:', index)
    data_dict = {}
    try:
        tg_index    = random.sample(index, num) #diffdock，，40，，，
        truth_mols  = truth_mol_list #ground truth，
        gen_mols    = []

        for mols in gen_mol_list:
            #sub_mol = [mols[i] for i in tg_index]
            #if model == 'ecdock_glide' or random_flag:
            if random_flag:
                random.shuffle(mols)
            sub_mol = mols[:num] #，num
            gen_mols.append(sub_mol)
    except Exception as e:
        truth_mols  = truth_mol_list #ground truth，
        gen_mols    = gen_mol_list


    rmsd_dict = defaultdict(list)      #
    rmsd_list = []
    rmsd_list_per = []  #，list

    #40rmsd2ai，True
    rmsd_mean_bool = []
    rmsd_min_bool  = []
    rmsd_cutoff = 2
    assert len(gen_mols) == len(data_name_list)
    for i, (mols, dt_name) in enumerate(zip(gen_mols, data_name_list)):
        tmp = []
        for mol in mols:
            try:
                #rsmd, ，GetBestRMS，rmsd
                #https://www.rdkit.org/docs/source/rdkit.Chem.rdMolAlign.html#rdkit.Chem.rdMolAlign.GetBestRMS
                #rmsd = AllChem.GetBestRMS(Chem.RemoveHs(mol), Chem.RemoveHs(truth_mols[i]))
                #rmsd = Chem.rdMolAlign.GetBestRMS(Chem.RemoveHs(mol), Chem.RemoveHs(truth_mols[i]))  


                #rmsd
                pre_pos  = Chem.RemoveHs(mol).GetConformer(0).GetPositions()
                holo_pos = Chem.RemoveHs(truth_mols[i]).GetConformer(0).GetPositions()

                assert pre_pos.shape == holo_pos.shape
                #rmsd = np.sqrt(np.mean(np.sum((pre_pos - holo_pos) ** 2, axis=-1)))  #rmsd，
                rmsd = np.sqrt(np.sum((pre_pos - holo_pos) ** 2) / pre_pos.shape[0]) # unimolrmsd，np.sum，, 

            
            except Exception as e:
                print('error:', e)
                #print('mol:', mol.GetConformer(0).GetPositions())
                #print('truth_mols:', truth_mols[i].GetConformer(0).GetPositions())
                continue
            tmp.append(rmsd)
            rmsd_list.append(rmsd)
        if tmp:
            rmsd_list_per.append(tmp)

            #num
            rmsd_dict['rmsd_mean'].append(np.mean(tmp))
            rmsd_dict['rmsd_std'].append(np.std(tmp))
            rmsd_dict['rsmd_mid'].append(np.median(tmp))
            rmsd_dict['rmsd_max'].append(np.max(tmp))
            rmsd_dict['rmsd_min'].append(np.min(tmp))
            rmsd_dict['data_name'].append(dt_name)
            rmsd_dict['data_mol'].append(mols) #mol, ，

            #rmsd2，
            np_rmsd = np.array(tmp)
            all_num = np_rmsd.shape[0] #，num
            indices = np_rmsd <= rmsd_cutoff 
            sub_num = np.count_nonzero(indices)
            rmsd_dict['rmsd_rate'].append(sub_num / all_num)

            #print('sub_num / all_num:', sub_num / all_num)
            #print('np.mean(np_rmsd <= 2):', np.mean(np_rmsd <= 2)) #


            #40rmsd2ai，True

            if indices.sum() >= 1:
                rmsd_min_bool.append(True)
            else:
                rmsd_min_bool.append(False)

            
            if np.mean(np_rmsd) <= rmsd_cutoff:
                rmsd_mean_bool.append(True)
            else:
                rmsd_mean_bool.append(False)




    print(' data num:', len(truth_mol_list) * num)
    print(' data num:', len(rmsd_list))

    #num，100, 
    rmsd_mean = round(np.mean(rmsd_dict['rmsd_mean']), 4)
    rmsd_std  = round(np.mean(rmsd_dict['rmsd_std']), 4)
    rsmd_mid  = round(np.mean(rmsd_dict['rsmd_mid']),4)
    rmsd_max  = round(np.mean(rmsd_dict['rmsd_max']), 4)
    rmsd_min  = round(np.mean(rmsd_dict['rmsd_min']), 4)
    rmsd_rate = round(np.mean(rmsd_dict['rmsd_rate']), 4)

    min_rate  = round(np.mean(rmsd_min_bool), 4)
    mean_rate = round(np.mean(rmsd_mean_bool), 4)

    data_dict['data_per']   =  rmsd_list_per #，numpy
    data_dict['data']       =  np.array(rmsd_list)
    data_dict['all']        = [rmsd_rate, min_rate, mean_rate, rmsd_mean, rmsd_std, rsmd_mid, rmsd_max, rmsd_min]

    data_dict['rmsd_mean']  = rmsd_dict['rmsd_mean']
    data_dict['rmsd_std']   = rmsd_dict['rmsd_std']
    data_dict['rmsd_mid']   = rmsd_dict['rsmd_mid']
    data_dict['rmsd_max']   = rmsd_dict['rsmd_max']
    data_dict['rmsd_min']   = rmsd_dict['rsmd_min']
    data_dict['data_name']  = rmsd_dict['data_name']
    data_dict['data_name_rmsd_mean']  = {k: v for k, v in zip(rmsd_dict['data_name'], rmsd_dict['rmsd_mean'])}
    data_dict['data_name_mol']        = {k: v for k, v in zip(rmsd_dict['data_name'], rmsd_dict['data_mol'])}

    data_dict['min_rate']  = np.mean(rmsd_min_bool)
    data_dict['mean_rate'] = np.mean(rmsd_mean_bool)
    

    return data_dict



def rmsdsV2(truth_mol_list, gen_mol_list, num):
    #truth_mollist，gen_mol2list，
    #，num
    #print('truth_mol_list:', len(truth_mol_list))
    #print('gen_mol_list:', gen_mol_list)
    index       = list(range(40)) #40，num
    #print('index:', index)
    data_dict = {}
    try:
        tg_index    = random.sample(index, num) #diffdock，，40，，，
        truth_mols  = truth_mol_list #ground truth，
        gen_mols    = []

        for mols in gen_mol_list:
            #sub_mol = [mols[i] for i in tg_index]
            random.shuffle(mols)
            sub_mol = mols[:num] #，num
            gen_mols.append(sub_mol)
    except Exception as e:
        truth_mols  = truth_mol_list #ground truth，
        gen_mols    = gen_mol_list


    rmsd_dict = defaultdict(list)      #
    rmsd_list = []
    rmsd_list_per = []  #，list
    for i, mols in enumerate(gen_mols):
        tmp = []
        for mol in mols:
            try:
                #rsmd, ，GetBestRMS，rmsd
                #https://www.rdkit.org/docs/source/rdkit.Chem.rdMolAlign.html#rdkit.Chem.rdMolAlign.GetBestRMS
                #rmsd = AllChem.GetBestRMS(Chem.RemoveHs(mol), Chem.RemoveHs(truth_mols[i]))
                #rmsd = Chem.rdMolAlign.GetBestRMS(Chem.RemoveHs(mol), Chem.RemoveHs(truth_mols[i]))  


                #rmsd
                pre_pos  = Chem.RemoveHs(mol).GetConformer(0).GetPositions()
                holo_pos = Chem.RemoveHs(truth_mols[i]).GetConformer(0).GetPositions()
                #rmsd = np.sqrt(np.mean(np.sum((pre_pos - holo_pos) ** 2, axis=-1)))  #rmsd，
                rmsd = np.sqrt(np.sum((pre_pos - holo_pos) ** 2) / pre_pos.shape[0]) # unimolrmsd，np.sum，, 

            
            except Exception as e:
                print('error:', e)
                #print('mol:', mol.GetConformer(0).GetPositions())
                #print('truth_mols:', truth_mols[i].GetConformer(0).GetPositions())
                continue
            tmp.append(rmsd)
            rmsd_list.append(rmsd)
        if tmp:
            rmsd_list_per.append(tmp)

            #num
            rmsd_dict['rmsd_mean'].extend(tmp)
            rmsd_dict['rmsd_std'].extend(tmp)
            rmsd_dict['rsmd_mid'].extend(tmp)
            rmsd_dict['rmsd_max'].extend(tmp)
            rmsd_dict['rmsd_min'].extend(tmp)

            #rmsd2，
            np_rmsd = np.array(tmp)
            all_num = np_rmsd.shape[0] #，num
            indices = np_rmsd <= 2 
            sub_num = np.count_nonzero(indices)
            rmsd_dict['rmsd_rate'].append(sub_num / all_num)

            #print('sub_num / all_num:', sub_num / all_num)
            #print('np.mean(np_rmsd <= 2):', np.mean(np_rmsd <= 2)) #



    print('all num:', len(truth_mol_list) * num)
    print('rmsd_list num:', len(rmsd_list))

    #num，100, 
    rmsd_mean = round(np.mean(rmsd_dict['rmsd_mean']), 4)
    rmsd_std  = round(np.std(rmsd_dict['rmsd_std']), 4)
    rsmd_mid  = round(np.median(rmsd_dict['rsmd_mid']),4)
    rmsd_max  = round(np.max(rmsd_dict['rmsd_max']), 4)
    rmsd_min  = round(np.min(rmsd_dict['rmsd_min']), 4)
    #rmsd_rate      = round(np.mean(rmsd_dict['rmsd_rate']), 4)
    rmsd_rate       = round(np.mean(np.array(rmsd_list) <= 2.0), 4)

    data_dict['data_per']   =  rmsd_list_per #，numpy
    data_dict['data']       =  np.array(rmsd_list)
    data_dict['all']        = [rmsd_rate, rmsd_mean, rmsd_std, rsmd_mid, rmsd_max, rmsd_min]

    return data_dict




def read_file(file_path, mode = None, flag = 'sdf', num = 1000, step = 24, model ='ecdock', name_list = None, poor_name_list = None):
    truth_mol_list, gen_mol_list = [], []
    failed_list = []
    data_name_list = []
    if model == 'ecdock':
        if flag == 'sdf':
            #/mnt/home/fanzhiguang/47/new_KGDiff-EcDock/ecdock_step25/result_0/step24/gen_ligand_0.sdf
            #Chem.rdmolfiles.SDMolSupplier(self.ligand_sdf)[0],[0], list，Chem.MolFromMolFile， ，sdf
            #for i in os.listdir(file_path)[:num]:
            for i in name_list[:num]:
            #for i in poor_name_list[:num]: #，ecdock
                #print(num)
                #print(i)
                path = os.path.join(file_path, i)
                if os.path.exists(path) and os.path.isdir(path) and os.listdir(path): #
                    #print('exist')
                    try:
                        base_path   = os.path.join(file_path, i)
                        
                        try:
                            org_sup     = Chem.rdmolfiles.SDMolSupplier(os.path.join(base_path, f'origin_{i}_ligand.sdf'))[0]
                            gen_sup     = Chem.rdmolfiles.SDMolSupplier(os.path.join(base_path, f'gen_{i}_ligand.sdf'))
                        except Exception as e:
                            base_path   = os.path.join(file_path, f'{i}')
                            org_sup     = Chem.rdmolfiles.SDMolSupplier(os.path.join(base_path, f'origin_{i}.sdf'))[0]
                            gen_sup     = Chem.rdmolfiles.SDMolSupplier(os.path.join(base_path, f'gen_{i}.sdf'))
                        
                        org_sup = Chem.RemoveHs(org_sup)
                        gen_sup = [Chem.RemoveHs(mol) for mol in gen_sup]
                        assert org_sup.GetNumAtoms() == gen_sup[0].GetNumAtoms()
                        truth_mol_list.append(org_sup) 
                        gen_mol_list.append(gen_sup) 
                    except Exception as e:
                        failed_list.append(i)
                        print('error:', e)
                        continue
                    data_name_list.append(i)
            #print('len(data_name_list):', len(data_name_list))
        else:
            with open(file_path + '/resault.pickle', 'rb') as file:
                data            = dill.load(file)
                truth_mol_list  = data['org']
                gen_mol_list    = data['gen']






    elif model == 'ecdock_rtm':
        if flag == 'sdf':
            #/mnt/home/fanzhiguang/47/new_KGDiff-EcDock/ecdock_step25/result_0/step24/gen_ligand_0.sdf
            #Chem.rdmolfiles.SDMolSupplier(self.ligand_sdf)[0],[0], list，Chem.MolFromMolFile， ，sdf
            #for i in os.listdir(file_path)[:num]:
            for i in name_list[:num]:
            #for i in poor_name_list[:num]: #，ecdock
                #print(num)
                #print(i)
                path = os.path.join(file_path, i)
                if os.path.exists(path) and os.path.isdir(path) and os.listdir(path): #
                    #print('exist')
                    try:
                        base_path   = os.path.join(file_path, f'{i}/step{step}')
                        if not os.path.exists(base_path):
                            base_path   = os.path.join(file_path, f'{i}')
                        org_sup     = Chem.rdmolfiles.SDMolSupplier(os.path.join(base_path, f'origin_{i}.sdf'))[0]
                        gen_sup     = Chem.rdmolfiles.SDMolSupplier(os.path.join(base_path, f'rtm_sort_gen_ligand_{i}.sdf'))
                        org_sup = Chem.RemoveHs(org_sup)
                        gen_sup = [Chem.RemoveHs(mol) for mol in gen_sup]
                        assert org_sup.GetNumAtoms() == gen_sup[0].GetNumAtoms()
                        truth_mol_list.append(org_sup) 
                        gen_mol_list.append(gen_sup) 
                    except Exception as e:
                        failed_list.append(i)
                        print('error:', e)
                        #exit()
                        continue
                    data_name_list.append(i)
            #print('len(data_name_list):', len(data_name_list))
        else:
            with open(file_path + '/resault.pickle', 'rb') as file:
                data            = dill.load(file)
                truth_mol_list  = data['org']
                gen_mol_list    = data['gen']



    
    elif model == 'ecdock_rtm_refine':
        if flag == 'sdf':
            #/mnt/home/fanzhiguang/47/new_KGDiff-EcDock/ecdock_step25/result_0/step24/gen_ligand_0.sdf
            #Chem.rdmolfiles.SDMolSupplier(self.ligand_sdf)[0],[0], list，Chem.MolFromMolFile， ，sdf
            #for i in os.listdir(file_path)[:num]:
            for i in name_list[:num]:
            #for i in poor_name_list[:num]: #，ecdock
                #print(num)
                #print(i)
                path = os.path.join(file_path, i)
                if os.path.exists(path) and os.path.isdir(path) and os.listdir(path): #
                    #print('exist')
                    try:
                        base_path   = os.path.join(file_path, f'{i}/step{step}')
                        if not os.path.exists(base_path):
                            base_path   = os.path.join(file_path, f'{i}')
                        org_sup     = Chem.rdmolfiles.SDMolSupplier(os.path.join(base_path, f'origin_{i}.sdf'))[0]
                        gen_sup     = Chem.rdmolfiles.SDMolSupplier(os.path.join(base_path, f'refine_rtm_sort_gen_ligand_{i}.sdf'))
                        org_sup = Chem.RemoveHs(org_sup)
                        gen_sup = [Chem.RemoveHs(mol) for mol in gen_sup]
                        assert org_sup.GetNumAtoms() == gen_sup[0].GetNumAtoms()
                        truth_mol_list.append(org_sup) 
                        gen_mol_list.append(gen_sup) 
                    except Exception as e:
                        failed_list.append(i)
                        print('error:', e)
                        #exit()
                        continue
                    data_name_list.append(i)
            #print('len(data_name_list):', len(data_name_list))
        else:
            with open(file_path + '/resault.pickle', 'rb') as file:
                data            = dill.load(file)
                truth_mol_list  = data['org']
                gen_mol_list    = data['gen']
                
                
    elif model == 'ecdock_score_glide':
        if flag == 'sdf':
            #/mnt/home/fanzhiguang/47/new_KGDiff-EcDock/ecdock_step25/result_0/step24/gen_ligand_0.sdf
            #Chem.rdmolfiles.SDMolSupplier(self.ligand_sdf)[0],[0], list，Chem.MolFromMolFile， ，sdf
            #for i in os.listdir(file_path)[:num]:
            for i in name_list[:num]:
            #for i in poor_name_list[:num]: #，ecdock
                #print(num)
                #print(i)
                path = os.path.join(file_path, i)
                if os.path.exists(path) and os.path.isdir(path) and os.listdir(path): #
                    #print('exist')
                    try:
                        base_path   = os.path.join(file_path, f'{i}')
                        org_sup     = Chem.rdmolfiles.SDMolSupplier(os.path.join(base_path, f'origin_ligand_{i}.sdf'))[0]
                        gen_sup     = Chem.rdmolfiles.SDMolSupplier(os.path.join(base_path, f'glide_gen_ligand_{i}.sdf'))
                        org_sup = Chem.RemoveHs(org_sup)
                        gen_sup = [Chem.RemoveHs(mol) for mol in gen_sup]
                        assert org_sup.GetNumAtoms() == gen_sup[0].GetNumAtoms()
                        truth_mol_list.append(org_sup) 
                        gen_mol_list.append(gen_sup) 
                    except Exception as e:
                        failed_list.append(i)
                        print('error:', e)
                        continue
                    data_name_list.append(i)
            #print('len(data_name_list):', len(data_name_list))
        else:
            with open(file_path + '/resault.pickle', 'rb') as file:
                data            = dill.load(file)
                truth_mol_list  = data['org']
                gen_mol_list    = data['gen']



    elif model == 'ecdock_score_glide_refine':
        if flag == 'sdf':
            #/mnt/home/fanzhiguang/47/new_KGDiff-EcDock/ecdock_step25/result_0/step24/gen_ligand_0.sdf
            #Chem.rdmolfiles.SDMolSupplier(self.ligand_sdf)[0],[0], list，Chem.MolFromMolFile， ，sdf
            #for i in os.listdir(file_path)[:num]:
            for i in name_list[:num]:
            #for i in poor_name_list[:num]: #，ecdock
                #print(num)
                #print(i)
                path = os.path.join(file_path, i)
                if os.path.exists(path) and os.path.isdir(path) and os.listdir(path): #
                    #print('exist')
                    try:
                        base_path   = os.path.join(file_path, f'{i}')
                        org_sup     = Chem.rdmolfiles.SDMolSupplier(os.path.join(base_path, f'origin_ligand_{i}.sdf'))[0]
                        gen_sup     = Chem.rdmolfiles.SDMolSupplier(os.path.join(base_path, f'redock_glide_gen_ligand_{i}.sdf'))
                        org_sup = Chem.RemoveHs(org_sup)
                        gen_sup = [Chem.RemoveHs(mol) for mol in gen_sup]
                        assert org_sup.GetNumAtoms() == gen_sup[0].GetNumAtoms()
                        truth_mol_list.append(org_sup) 
                        gen_mol_list.append(gen_sup) 
                    except Exception as e:
                        failed_list.append(i)
                        print('error:', e)
                        continue
                    data_name_list.append(i)
            #print('len(data_name_list):', len(data_name_list))
        else:
            with open(file_path + '/resault.pickle', 'rb') as file:
                data            = dill.load(file)
                truth_mol_list  = data['org']
                gen_mol_list    = data['gen']





    elif model == 'ecdock_refine':
        if flag == 'sdf':
            #/mnt/home/fanzhiguang/47/new_KGDiff-EcDock/ecdock_step25/result_0/step24/gen_ligand_0.sdf
            #Chem.rdmolfiles.SDMolSupplier(self.ligand_sdf)[0],[0], list，Chem.MolFromMolFile， ，sdf
            #for i in os.listdir(file_path)[:num]:
            for i in name_list[:num]:
            #for i in poor_name_list[:num]: #，ecdock
                #print(num)
                #print(i)
                path = os.path.join(file_path, i)
                if os.path.exists(path) and os.path.isdir(path) and os.listdir(path): #
                    #print('exist')
                    try:
                        base_path   = os.path.join(file_path, f'{i}')
                        org_sup     = Chem.rdmolfiles.SDMolSupplier(os.path.join(base_path, f'origin_ligand_{i}.sdf'))[0]
                        gen_sup     = Chem.rdmolfiles.SDMolSupplier(os.path.join(base_path, f'redock_glide_gen_ligand_{i}.sdf'))
                        org_sup = Chem.RemoveHs(org_sup)
                        gen_sup = [Chem.RemoveHs(mol) for mol in gen_sup]
                        assert org_sup.GetNumAtoms() == gen_sup[0].GetNumAtoms()
                        truth_mol_list.append(org_sup) 
                        gen_mol_list.append(gen_sup) 
                    except Exception as e:
                        failed_list.append(i)
                        print('error:', e)
                        continue
                    data_name_list.append(i)
            #print('len(data_name_list):', len(data_name_list))
        else:
            with open(file_path + '/resault.pickle', 'rb') as file:
                data            = dill.load(file)
                truth_mol_list  = data['org']
                gen_mol_list    = data['gen']






    elif model == 'glide':
        if flag == 'sdf':
            #/mnt/home/fanzhiguang/47/new_KGDiff-EcDock/ecdock_step25/result_0/step24/gen_ligand_0.sdf
            #Chem.rdmolfiles.SDMolSupplier(self.ligand_sdf)[0],[0], list，Chem.MolFromMolFile， ，sdf
            #for i in os.listdir(file_path)[:num]:
            for i in name_list[:num]:
            #for i in poor_name_list[:num]: #，ecdock
                #print(num)
                #print(i)
                path = os.path.join(file_path, i)
                if os.path.exists(path) and os.path.isdir(path) and os.listdir(path): #
                    #print('exist')
                    try:
                        base_path   = os.path.join(file_path, f'{i}')
                        org_sup     = Chem.rdmolfiles.SDMolSupplier(os.path.join(base_path, f'origin_ligand_{i}.sdf'))[0]
                        gen_sup     = Chem.rdmolfiles.SDMolSupplier(os.path.join(base_path, f'origin_ligand_{i}_config_onlyscore_lib.sdf'))
                        org_sup = Chem.RemoveHs(org_sup)
                        gen_sup = [Chem.RemoveHs(mol) for mol in gen_sup]
                        assert org_sup.GetNumAtoms() == gen_sup[0].GetNumAtoms()
                        truth_mol_list.append(org_sup) 
                        gen_mol_list.append(gen_sup) 
                    except Exception as e:
                        failed_list.append(i)
                        print('error:', e)
                        continue
                    data_name_list.append(i)
            #print('len(data_name_list):', len(data_name_list))
        else:
            with open(file_path + '/resault.pickle', 'rb') as file:
                data            = dill.load(file)
                truth_mol_list  = data['org']
                gen_mol_list    = data['gen']



    elif model == 'vina':
        if flag == 'sdf':
            #/mnt/home/fanzhiguang/47/new_KGDiff-EcDock/ecdock_step25/result_0/step24/gen_ligand_0.sdf
            #Chem.rdmolfiles.SDMolSupplier(self.ligand_sdf)[0],[0], list，Chem.MolFromMolFile， ，sdf
            #for i in os.listdir(file_path)[:num]:
            for i in name_list[:num]:
            #for i in poor_name_list[:num]: #，ecdock
                #print(num)
                #print(i)
                path = os.path.join(file_path, i)
                if os.path.exists(path) and os.path.isdir(path) and os.listdir(path): #
                    #print('exist')
                    try:
                        base_path   = os.path.join(file_path, f'{i}')
                        org_sup     = Chem.rdmolfiles.SDMolSupplier(os.path.join(base_path, f'origin_ligand_{i}.sdf'))[0]
                        gen_sup     = Chem.rdmolfiles.SDMolSupplier(os.path.join(base_path, f'vina_gen_ligand_{i}.sdf'))
                        org_sup = Chem.RemoveHs(org_sup)
                        gen_sup = [Chem.RemoveHs(mol) for mol in gen_sup]
                        assert org_sup.GetNumAtoms() == gen_sup[0].GetNumAtoms()
                        truth_mol_list.append(org_sup) 
                        gen_mol_list.append(gen_sup) 
                    except Exception as e:
                        failed_list.append(i)
                        print('error:', e)
                        continue
                    data_name_list.append(i)
            #print('len(data_name_list):', len(data_name_list))
        else:
            with open(file_path + '/resault.pickle', 'rb') as file:
                data            = dill.load(file)
                truth_mol_list  = data['org']
                gen_mol_list    = data['gen']





    elif model == 'diffdock':
        #/mnt/home/fanzhiguang/47/DiffDock-main/results/user_predictions_small/complex_0/gen.sdf
        #for i in os.listdir(file_path)[:num]:
        for i in name_list[:num]:
            path = os.path.join(file_path, i)
            if os.path.exists(path) and os.path.isdir(path) and os.listdir(path): #
                try:
                    base_path   = os.path.join(file_path, f'{i}')
                    org_sup     = Chem.rdmolfiles.SDMolSupplier(os.path.join(base_path, 'origin.sdf'))[0]  #Chem.rdmolfiles.SDMolSupplierlist
                    gen_sup     = Chem.rdmolfiles.SDMolSupplier(os.path.join(base_path, 'gen.sdf'))
                    org_sup = Chem.RemoveHs(org_sup)
                    gen_sup = [Chem.RemoveHs(mol) for mol in gen_sup]
                    assert org_sup.GetNumAtoms() == gen_sup[0].GetNumAtoms()
                    truth_mol_list.append(org_sup) 
                    gen_mol_list.append(gen_sup) 
                except Exception as e:
                    failed_list.append(i)
                    print('error:', e)
                    continue
                data_name_list.append(i)
        
    elif model == 'glide':
        #for i in os.listdir(file_path)[:num]:
        for i in name_list[:num]:
            path = os.path.join(file_path, i)
            #print('path:', path)
            if os.path.exists(path) and os.path.isdir(path) and os.listdir(path) and len(os.listdir(path)) > 1: #
                base_path   = os.path.join(file_path, f'{i}')
                try:
                    file = os.path.join(base_path, f'{i}_ligand_config_lib.sdf')

                    if  not os.path.exists(file):
                        file = os.path.join(base_path, f'{i}_ligand-rdkit-glide.sdf')

                    gen_sup     = Chem.rdmolfiles.SDMolSupplier(file)
                except OSError:
                    failed_list.append(i)
                    continue #
                org_sup     = Chem.rdmolfiles.SDMolSupplier(os.path.join(base_path, f'{i}_ligand.sdf'))[0]  #Chem.rdmolfiles.SDMolSupplierlist
                org_sup = Chem.RemoveHs(org_sup)
                gen_sup = [Chem.RemoveHs(mol) for mol in gen_sup][:] #
                try:
                    assert org_sup.GetNumAtoms() == gen_sup[0].GetNumAtoms() #
                except:
                    failed_list.append(i)
                    print(f'{org_sup.GetNumAtoms()} != {gen_sup[0].GetNumAtoms()}') #43 != 42
                    continue
                truth_mol_list.append(org_sup) 
                gen_mol_list.append(gen_sup) 
                data_name_list.append(i)
        
    elif model == 'KarmaDock':
        if mode == 'uncorrected':
            #for i in os.listdir(file_path)[:num]:
            for i in name_list[:num]:
                path = os.path.join(file_path, i)
                if os.path.exists(path) and os.path.isdir(path) and os.listdir(path): #
                    try:
                        base_path   = os.path.join(file_path, f'{i}')
                        org_sup     = Chem.rdmolfiles.SDMolSupplier(os.path.join(base_path, f'{i}_org.sdf'))[0]  #Chem.rdmolfiles.SDMolSupplierlist
                        gen_sup     = Chem.rdmolfiles.SDMolSupplier(os.path.join(base_path, f'{i}_pred_uncorrected.sdf'))
                        org_sup = Chem.RemoveHs(org_sup)
                        gen_sup = [Chem.RemoveHs(mol) for mol in gen_sup]
                        assert org_sup.GetNumAtoms() == gen_sup[0].GetNumAtoms()
                        truth_mol_list.append(org_sup) 
                        gen_mol_list.append(gen_sup) 
                    except Exception as e:
                        failed_list.append(i) #reading failed_list: ['6mla', '6erv', '6r4k', '5nw8', '5zw6', '6hp5', '6jse', '6d3x', '6bvh', '6s07', '5wyq', '6a8n', '5ol3']
                        print('error:', e)
                        continue
                    data_name_list.append(i)
        elif mode == 'ff_corrected':
            #for i in os.listdir(file_path)[:num]:
            for i in name_list[:num]:
                path = os.path.join(file_path, i)
                if os.path.exists(path) and os.path.isdir(path) and os.listdir(path): #
                    try:
                        base_path   = os.path.join(file_path, f'{i}')
                        org_sup     = Chem.rdmolfiles.SDMolSupplier(os.path.join(base_path, f'{i}_org.sdf'))[0]  #Chem.rdmolfiles.SDMolSupplierlist
                        gen_sup     = Chem.rdmolfiles.SDMolSupplier(os.path.join(base_path, f'{i}_pred_ff_corrected.sdf'))
                        org_sup = Chem.RemoveHs(org_sup)
                        gen_sup = [Chem.RemoveHs(mol) for mol in gen_sup]
                        assert org_sup.GetNumAtoms() == gen_sup[0].GetNumAtoms()
                        truth_mol_list.append(org_sup) 
                        gen_mol_list.append(gen_sup) 
                    except Exception as e:
                        failed_list.append(i) #['6mla', '6erv', '6r4k', '5nw8', '5zw6', '6hp5', '6jse', '6d3x', '6bvh', '6s07', '6a8n', '5ol3']
                        print('error:', e)
                        continue
                    data_name_list.append(i)
        elif mode == 'align_corrected':
            #for i in os.listdir(file_path)[:num]:
            for i in name_list[:num]:
                path = os.path.join(file_path, i)
                if os.path.exists(path) and os.path.isdir(path) and os.listdir(path): #
                    try:
                        base_path   = os.path.join(file_path, f'{i}')
                        org_sup     = Chem.rdmolfiles.SDMolSupplier(os.path.join(base_path, f'{i}_org.sdf'))[0]  #Chem.rdmolfiles.SDMolSupplierlist
                        gen_sup     = Chem.rdmolfiles.SDMolSupplier(os.path.join(base_path, f'{i}_pred_align_corrected.sdf'))
                        org_sup = Chem.RemoveHs(org_sup)
                        gen_sup = [Chem.RemoveHs(mol) for mol in gen_sup]
                        assert org_sup.GetNumAtoms() == gen_sup[0].GetNumAtoms()
                        truth_mol_list.append(org_sup) 
                        gen_mol_list.append(gen_sup) 
                    except Exception as e:
                        failed_list.append(i) #['6mla', '6erv', '6r4k', '5nw8', '5zw6', '6hp5', '6jse', '6d3x', '6bvh', '6s07', '6a8n', '5ol3']
                        print('error:', e)
                        continue
                    data_name_list.append(i)

    elif model == 'unimol':
        if mode == 'uncorrected':
            #for i in os.listdir(file_path)[:num]:
            #print('name_list:', name_list)
            for i in name_list[:num]:
                path = os.path.join(file_path, i)
                #print('path:', path)
                if os.path.exists(path) and os.path.isdir(path) and os.listdir(path): #
                    try:
                        base_path   = os.path.join(file_path, f'{i}')
                        org_sup     = Chem.rdmolfiles.SDMolSupplier(os.path.join(base_path, f'org_{i}.sdf'))[0]  #Chem.rdmolfiles.SDMolSupplierlist
                        gen_sup     = Chem.rdmolfiles.SDMolSupplier(os.path.join(base_path, f'gen_{i}.sdf'))
                        org_sup = Chem.RemoveHs(org_sup)
                        gen_sup = [Chem.RemoveHs(mol) for mol in gen_sup]
                        assert org_sup.GetNumAtoms() == gen_sup[0].GetNumAtoms()
                        truth_mol_list.append(org_sup) 
                        gen_mol_list.append(gen_sup) 
                    except Exception as e:
                        failed_list.append(i) #reading failed_list: ['6mla', '6erv', '6r4k', '5nw8', '5zw6', '6hp5', '6jse', '6d3x', '6bvh', '6s07', '5wyq', '6a8n', '5ol3']
                        print('error:', e)
                        continue
                    data_name_list.append(i)
        elif mode == 'corrected':
            #for i in os.listdir(file_path)[:num]:
            for i in name_list[:num]:
                path = os.path.join(file_path, i)
                if os.path.exists(path) and os.path.isdir(path) and os.listdir(path): #
                    try:
                        base_path   = os.path.join(file_path, f'{i}')
                        org_sup     = Chem.rdmolfiles.SDMolSupplier(os.path.join(base_path, f'org_{i}.sdf'))[0]  #Chem.rdmolfiles.SDMolSupplierlist
                        gen_sup     = Chem.rdmolfiles.SDMolSupplier(os.path.join(base_path, f'clash_fix_optimize_gen_{i}.sdf'))
                        org_sup = Chem.RemoveHs(org_sup)
                        gen_sup_ = []
                        for mol in gen_sup:
                            try: 
                                Chem.RemoveHs(mol) #mol，nan，，
                                gen_sup_.append(mol)
                            except Exception as e:
                                print('error:', e)
                                continue
                        #gen_sup = [Chem.RemoveHs(mol) for mol in gen_sup]
                        gen_sup = gen_sup_
                        assert org_sup.GetNumAtoms() == gen_sup[0].GetNumAtoms()
                        truth_mol_list.append(org_sup) 
                        gen_mol_list.append(gen_sup) 
                    except Exception as e:
                        failed_list.append(i) #reading failed_list: ['6mla', '6erv', '6r4k', '5nw8', '5zw6', '6hp5', '6jse', '6d3x', '6bvh', '6s07', '5wyq', '6a8n', '5ol3']
                        print('error:', e)
                        continue
                    data_name_list.append(i)
        else:
            raise Exception('need model name')

    
    print('reading failed_list:', failed_list)
    
    return truth_mol_list, gen_mol_list, data_name_list

def boxplot(boxplot_data_list, save_path, name):
    data = boxplot_data_list

    # 
    #plt.boxplot(data, vert=True, patch_artist=True)
    plt.boxplot(data,
                notch = True, 
                sym = 'b+', 			# 
                vert = True, 
                #whis = 1, 
                positions = [1,2,3,4,5,6], 
                widths = 0.4, 
                showmeans = True
    )

    # 
    plt.title(f'{name.upper()} RMSD')
    plt.xlabel('Sampling Num')
    plt.ylabel('Value')
    plt.xticks([1, 2, 3, 4, 5, 6], [1, 5, 10, 25, 40, 'MAX'])
    plt.show()
    plt.savefig(save_path)
    plt.close()




def histplot(plot_data_dict, base_save_path, name, model):
    name_list = plot_data_dict['data_name']
    for k_name in list(plot_data_dict.keys())[:-1]:
        #print('k_name:', k_name)
        data = plot_data_dict[k_name]
        data = np.array(data)

        if k_name == 'rmsd_mean':
            print('rmsd_mean <= 2 rate:', round(np.mean(data < 2.0), 4))
            print('rmsd_mean > 2 rate:', round(np.mean(data > 2.0), 4))
            print('rmsd_mean > 3 rate:', round(np.mean(data > 3.0), 4))
            print('rmsd_mean > 4 rate:', round(np.mean(data > 4.0), 4))
            print('rmsd_mean > 5 rate:', round(np.mean(data > 5.0), 4))

            #，
            if model == 'ecdock':
                with open(f'../CrossDocked2020/data/{data_name}_poor_name.txt', 'w') as f: #data_name
                    assert len(name_list) == len(data)
                    for name, dt in zip(name_list, data > 2.0):
                        if dt:
                            f.write(f'{name}\n')


        # 
        sns.histplot(data, bins=30, kde=True, color='blue')

        # 
        plt.title(f'{k_name.upper()} Histogram of {model.upper()}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')

        # 
        #plt.show()
        save_path = os.path.join(os.path.dirname(base_save_path), os.path.splitext(os.path.basename(base_save_path))[0] + f'_{k_name}.png')
        #print('save_path:', save_path)
        plt.savefig(f'{save_path}')
    plt.close()





def only_boxplot(optim_good_atom_list, optim_poor_atom_list, save_path, name1 = 'optim_good', name2 = 'optim_poor', title = 'optim atom num'):
    data = [optim_good_atom_list, optim_poor_atom_list]

    save_path = os.path.join(save_path, f'box_plot_{title}.png')
    # 
    #plt.boxplot(data, vert=True, patch_artist=True)
    plt.boxplot(data,
                notch = True, 
                sym = 'b+', 			# 
                vert = True, 
                #whis = 1, 
                positions = [1,2], 
                widths = 0.4, 
                showmeans = True
    )

    # 
    plt.title(f'{title.upper()}')
    plt.xlabel('Atom Num')
    plt.ylabel('Value')
    plt.xticks([1, 2], [name1, name2])
    plt.grid()
    plt.show()
    plt.savefig(save_path)
    plt.close()




def only_plot(atom_num_list, optim_min_rmsd_list, no_optim_min_rmsd_list, save_path, title):
    # 
    plt.scatter(optim_min_rmsd_list, atom_num_list, marker='o', s = 20, label = 'optim')
    plt.scatter(no_optim_min_rmsd_list, atom_num_list, marker='s', s = 20, label = 'no_optim')

    # 
    plt.title(f'{title.upper()}')
    plt.xlabel('RMSD')
    plt.ylabel('Atom Num')

    # 
    plt.legend()

    # 
    plt.grid()
    #plt.show()

    save_path = os.path.join(save_path, f'scatter_plot_{title}.png')
    #print('save_path:', save_path)
    plt.savefig(f'{save_path}')
    plt.close()




def only_rmsd(truth_mol, mols):
    rmsd_list = []
    for mol in mols:
        try:
            #rsmd, ，GetBestRMS，rmsd
            #https://www.rdkit.org/docs/source/rdkit.Chem.rdMolAlign.html#rdkit.Chem.rdMolAlign.GetBestRMS
            #rmsd = AllChem.GetBestRMS(Chem.RemoveHs(mol), Chem.RemoveHs(truth_mols[i]))
            #rmsd = Chem.rdMolAlign.GetBestRMS(Chem.RemoveHs(mol), Chem.RemoveHs(truth_mols[i]))  


            #rmsd
            pre_pos  = Chem.RemoveHs(mol).GetConformer(0).GetPositions()
            holo_pos = Chem.RemoveHs(truth_mol).GetConformer(0).GetPositions()

            assert pre_pos.shape == holo_pos.shape
            #rmsd = np.sqrt(np.mean(np.sum((pre_pos - holo_pos) ** 2, axis=-1)))  #rmsd，
            rmsd = np.sqrt(np.sum((pre_pos - holo_pos) ** 2) / pre_pos.shape[0]) # unimolrmsd，np.sum，, 
            rmsd_list.append(rmsd)

        
        except Exception as e:
            print('error:', e)
            #print('mol:', mol.GetConformer(0).GetPositions())
            #print('truth_mols:', truth_mols[i].GetConformer(0).GetPositions())
            continue
    
    return np.array(rmsd_list)



def force_analy(truth_mols, no_optim_gen_mols, optim_gen_mols, data_name_list, save_path):
    #rmsd，xmin_rsmdmean_rmsd，y， 
    #，，，
    #，min_rsmdmean_rmsd

    #
    optim_min_rmsd_list         = []
    optim_mean_rmsd_list        = []
    no_optim_min_rmsd_list      = []
    no_optim_mean_rmsd_list     = []
    all_atom_num_list           = []

    #
    optim_good_atom_list = []
    optim_poor_atom_list = []

    #，，
    optim_good_molname_list = []
    optim_poor_molname_list = []


    #，，rate
    min_rmsd_bool_list  = []
    mean_rmsd_bool_list = []

    optim_min_rmsd_bool_list  = []
    optim_mean_rmsd_bool_list = []

    no_optim_min_rmsd_bool_list  = []
    no_optim_mean_rmsd_bool_list = []

    count = 0

    cutoff = 2
    for truth_mol, no_optim_gen_mol, optim_gen_mol, data_name in zip(truth_mols, no_optim_gen_mols, optim_gen_mols, data_name_list):
        #print('data_name:', data_name) 
        count += 1  

        ##
        optim_rmsds     = only_rmsd(truth_mol, optim_gen_mol)
        optim_min_rmsd  = np.min(optim_rmsds)
        optim_mean_rmsd = np.mean(optim_rmsds)

        no_optim_rmsds     = only_rmsd(truth_mol, no_optim_gen_mol)
        no_optim_min_rmsd  = np.min(no_optim_rmsds)
        no_optim_mean_rmsd = np.mean(no_optim_rmsds)

        atom_num = truth_mol.GetNumAtoms()

        if no_optim_min_rmsd <= 3:
            optim_min_rmsd_list.append(optim_min_rmsd)
            optim_mean_rmsd_list.append(optim_mean_rmsd)

            no_optim_min_rmsd_list.append(no_optim_min_rmsd)
            no_optim_mean_rmsd_list.append(no_optim_mean_rmsd)

            all_atom_num_list.append(atom_num)

        #
        if optim_min_rmsd <= cutoff and no_optim_min_rmsd > cutoff:
        #if optim_min_rmsd <= cutoff and optim_min_rmsd < no_optim_min_rmsd:
            optim_good_atom_list.append(atom_num)
            optim_good_molname_list.append(data_name)
        else:
            optim_poor_atom_list.append(atom_num)
            optim_poor_molname_list.append(data_name)
        

        #，，rate
        if optim_min_rmsd <= cutoff or no_optim_min_rmsd <= cutoff:
            min_rmsd_bool_list.append(True)
        else:
            min_rmsd_bool_list.append(False)
        
        if optim_mean_rmsd <= cutoff or no_optim_mean_rmsd <= cutoff:
            mean_rmsd_bool_list.append(True)
        else:
            mean_rmsd_bool_list.append(False)

        

        #，，rate
        if optim_min_rmsd <= cutoff:
            optim_min_rmsd_bool_list.append(True)
        else:
            optim_min_rmsd_bool_list.append(False)
        
        if optim_mean_rmsd <= cutoff:
            optim_mean_rmsd_bool_list.append(True)
        else:
            optim_mean_rmsd_bool_list.append(False)


        #，，rate
        if no_optim_min_rmsd <= cutoff:
            no_optim_min_rmsd_bool_list.append(True)
        else:
            no_optim_min_rmsd_bool_list.append(False)
        
        if no_optim_mean_rmsd <= cutoff:
            no_optim_mean_rmsd_bool_list.append(True)
        else:
            no_optim_mean_rmsd_bool_list.append(False)
    




    print('count:', count)

    #
    #optim_min_rmsd_list         
    #optim_mean_rmsd_list        
    #no_optim_min_rmsd_list      
    #no_optim_mean_rmsd_list     
    #all_atom_num_list 
    
    only_plot(all_atom_num_list, optim_min_rmsd_list, no_optim_min_rmsd_list, save_path, 'min_rmsd')  
    only_plot(all_atom_num_list, optim_mean_rmsd_list, no_optim_mean_rmsd_list, save_path, 'mean_rmsd')        

    #
    #optim_good_atom_list
    #optim_poor_atom_list 
    only_boxplot(optim_good_atom_list, optim_poor_atom_list, save_path, 'optim_good', 'optim_poor', title = 'optim atom num')
    print('mean(optim_good_atom_list):', f'{np.mean(optim_good_atom_list):.2f}') #26.073578595317727
    print('median(optim_good_atom_list):', f'{np.median(optim_good_atom_list):.2f}')
    print('std(optim_good_atom_list):', f'{np.std(optim_good_atom_list):.2f}')


    print('mean(optim_poor_atom_list):', f'{np.mean(optim_poor_atom_list):.2f}') #23.26050420168067
    print('median(optim_poor_atom_list):', f'{np.median(optim_poor_atom_list):.2f}')
    print('std(optim_poor_atom_list):', f'{np.std(optim_poor_atom_list):.2f}')

    #，，
    #optim_good_molname_list 
    #optim_poor_molname_list 

    print('optim_good_molname_list num:', len(optim_good_molname_list))
    print('optim_poor_molname_list num:', len(optim_poor_molname_list))

    with open(os.path.join(save_path, 'optim_good_molname.txt'), 'w') as f:
        for i in optim_good_molname_list:
            f.write(i + '\n')
        
    with open(os.path.join(save_path, 'optim_poor_molname.txt'), 'w') as f:
        for i in optim_poor_molname_list:
            f.write(i + '\n')

    


    #，，rate
    #min_rmsd_bool_list  
    #mean_rmsd_bool_list 
    #print('min_rmsd_bool_list:', min_rmsd_bool_list)
    #print('mean_rmsd_bool_list:', mean_rmsd_bool_list)
    print('both min_rmsd rate: {:.3f}'.format(np.mean(min_rmsd_bool_list)))
    print('both mean_rmsd rate: {:.3f}'.format(np.mean(mean_rmsd_bool_list)))

    print('optim min_rmsd rate: {:.3f}'.format(np.mean(optim_min_rmsd_bool_list)))
    print('optim mean_rmsd rate: {:.3f}'.format(np.mean(optim_mean_rmsd_bool_list)))

    print('no_optim both min_rmsd rate: {:.3f}'.format(np.mean(no_optim_min_rmsd_bool_list)))
    print('no_optim both mean_rmsd rate: {:.3f}'.format(np.mean(no_optim_mean_rmsd_bool_list)))

    with open(os.path.join(save_path, 'reault.txt'), 'w') as f:
        f.write('mean(optim_good_atom_list):' +  f'{np.mean(optim_good_atom_list):.2f}\n') #26.073578595317727
        f.write('median(optim_good_atom_list):' + f'{np.median(optim_good_atom_list):.2f}\n')
        f.write('std(optim_good_atom_list):' + f'{np.std(optim_good_atom_list):.2f}\n\n')


        f.write('mean(optim_poor_atom_list):' +  f'{np.mean(optim_poor_atom_list):.2f}\n') #23.26050420168067
        f.write('median(optim_poor_atom_list):' + f'{np.median(optim_poor_atom_list):.2f}\n')
        f.write('std(optim_poor_atom_list):' +  f'{np.std(optim_poor_atom_list):.2f}\n\n')

        f.write('both min_rmsd rate: {:.3f}\n'.format(np.mean(min_rmsd_bool_list)))
        f.write('both mean_rmsd rate: {:.3f}\n\n'.format(np.mean(mean_rmsd_bool_list)))
        
        f.write('optim min_rmsd rate: {:.3f}\n'.format(np.mean(optim_min_rmsd_bool_list)))
        f.write('optim mean_rmsd rate: {:.3f}\n\n'.format(np.mean(optim_mean_rmsd_bool_list)))

        f.write('no_optim both min_rmsd rate: {:.3f}\n'.format(np.mean(no_optim_min_rmsd_bool_list)))
        f.write('no_optim both mean_rmsd rate: {:.3f}\n\n'.format(np.mean(no_optim_mean_rmsd_bool_list)))

        f.write('optim_good_molname_list num: {}\n'.format(len(optim_good_molname_list)))
        f.write('optim_poor_molname_list num: {}\n\n'.format(len(optim_poor_molname_list)))




def train_evaluate(file_path = None):
    #
    set_seed(2024) #data_path

    random_flag = False  #

    model = 'ecdock'
    data_name = 'posebustersv1'  #new_pdb2020_test, posebusters, pdbbind2020_r10, casf2016, posebustersv1, posebustersv2
    step = 5
    gnn  = 'equiformer'  #ecdock，, equiformer
    diffusion = 'cm' #ecdock，， CM/DDPM
    mode = '' #

        
    model_name = data_name + '_' + model + '_' + diffusion + '_' + gnn + f'_step{step}' #
    step = step - 1
    
        
    name_list = []
    with open(f'../CrossDocked2020/data/{data_name}/{data_name}_name.txt') as f:
        for i in f:
            name_list.append(i.strip())

    #，，，
    poor_name_list = []
    print('name_list num:', len(name_list))
        


    #sdf,truth_mollist，gen_mol2list，
    truth_mol, gen_mol, data_name_list = read_file(file_path, mode, flag = 'sdf', num = 100000, step = step, model = model, name_list = name_list, poor_name_list = poor_name_list)  #，rdkit mol, step
    print('truth_mol, gen_mol:', len(truth_mol), len(gen_mol))
    assert len(truth_mol) == len(gen_mol)


    #rmsd。401/3/5/10/40，rmsd
    resault_dict = {}
    boxplot_data_list   = [] #1,5,40
    histplot_data_dict  = {} #rmsd
    dt_name_dict = {}
    dt_name_mol_dict = {}
    for num in [1, 3, 5, 10, 25, 40, 100000][:]: #1000
        data_dict = rmsds(truth_mol, gen_mol, num, data_name_list, model, random_flag) #，num
        resault_dict[num] = ['rate, rate_min, rate_mean, rmsd_mean, rmsd_std, rsmd_mid, rmsd_max, rmsd_min:', data_dict['all']]
        rate = data_dict['all'][2]
        if num in [1, 5, 10, 25, 40, 100000]:
            boxplot_data_list.append(data_dict['data'])

        if num == 100000:
            histplot_data_dict['rmsd_mean'] = data_dict['rmsd_mean']
            histplot_data_dict['rmsd_min']  = data_dict['rmsd_min']
            histplot_data_dict['data_name'] = data_dict['data_name']
            dt_name_dict     = data_dict['data_name_rmsd_mean'] 
            dt_name_mol_dict = data_dict['data_name_mol']

    #rmsd，
    dt_name_sorted_dict = dict(sorted(dt_name_dict.items(), key=lambda item: item[1], reverse=False))
    #data_name_mol
    dt_name_mol_sorted_dict = {}
    for k in dt_name_sorted_dict:
        dt_name_mol_sorted_dict[k] = dt_name_mol_dict[k]
    print('rmsd sorted from smallest to biggest', list(dt_name_sorted_dict.keys()))

    with open(f'resault/{model}_rmsdmean_sorted.pkl', 'wb') as f:
        dill.dump(dt_name_sorted_dict, f)

    with open(f'resault/{model}_mol_sorted.pkl', 'wb') as f:
        dill.dump(dt_name_sorted_dict, f)

    #exit()
    print(json.dumps(resault_dict, indent=4))
    #JSON
    #path = 'resault'
    path = file_path
    os.makedirs(path, exist_ok=True)

    file_name = f'{model_name}_evaluate_resault.json'
    with open(os.path.join(path, file_name), 'w') as file:
        json.dump(resault_dict, file, indent=4)


    #exit()
    #print(json.dumps(resault_dict, indent=4))
    #JSON
    path = 'resault'
    os.makedirs(path, exist_ok=True)

    file_name = f'{model_name}_evaluate_resault.json'
    with open(os.path.join(path, file_name), 'w') as file:
        json.dump(resault_dict, file, indent=4)
    

    
    return rate



if __name__ == '__main__':
    #
    set_seed(2024)

    '''
    #diffdock
    name_list = []
    with open('../CrossDocked2020/data/pdb2020_test_name.txt') as f:
        for i in f:
            name_list.append(i.strip())
    print('len(name_list):', len(name_list))

    with open('../CrossDocked2020/data/protein_ligand_example_csv.csv', 'w') as f:
        f.write('complex_name,protein_path,ligand_description,protein_sequence\n') #
        for i in name_list:
            #complex_name,protein_path,ligand_description,protein_sequence
            #,data/PDBBind_processed/5l8c/5l8c_protein_processed.pdb,data/PDBBind_processed/5l8c/5l8c_ligand.sdf,
            ps = f'../CrossDocked2020/data/pdb2020_test/{i}'
            pf = os.path.join(ps, f'{i}_pocket10_400.pdb')
            lf = os.path.join(ps, f'{i}_ligand.sdf')
            line = f',{pf},{lf},\n'
            f.write(line)

    '''
    random_flag = False  #

    model = 'ecdock'
    data_name = 'posebustersv1'  #new_pdb2020_test, posebusters, pdbbind2020_r10, casf2016, posebustersv1, posebustersv2
    step = 5
    gnn  = 'equiformer'  #ecdock，, equiformer
    diffusion = 'cm' #ecdock，， CM/DDPM
    mode = '' #
    if model == 'ecdock':    #posebusters_ecdock_cm_equiformer_step1
        #file_path = f'/mnt/home/fanzhiguang/47/new_KGDiff-EcDock/{data_name}_ecdock_{diffusion}_{gnn}_step{step}' #，sdfpickle，
        #file_path = 'posebusters_glide_ecdock_cm_equiformer_step10_atom_8i_glide'
        #file_path = 'pdbbind2020_r10_ecdock_cm_equiformer_step25_interaction_limit4.5ai_only_test_leak4.5_finetune'
        #file_path = '/mnt/home/fanzhiguang/47/new_KGDiff-EcDock/posebusters_ecdock_cm_equiformer_step25_interaction_limit4.5ai_retain_fine_test_3'
        #file_path = 'posebusters_ecdock_cm_equiformer_step25_interaction_limit4.5ai_retrain_fine'
        #file_path = 'posebusters_ecdock_cm_equiformer_step25_interaction_limit4.5ai_retrain'
        #file_path = 'posebusters_ecdock_cm_equiformer_step25_interaction_limit4.5ai_retain_fine_No_'
        #file_path = '/mnt_191/fanzhiguang/47/mnt/EcDock_sample_dir/posebusters_ecdock_cm_equiformer_step15_interaction_limit4.5ai_ddpm_egnn_gen_split_3.5_best'
        #file_path = f'/data/fan_zg/MDocking/EcDock_Evaluate/evaluate_model/ecdock_refine/{data_name}/posebustersv1_257_cross0_force0_optim0_step5_conf40_glide_refine'
        file_path = '/data/fan_zg/MDocking/ECDock_sample_dir/posebustersv1_ecdock_cm_equiformer_step5_interaction_limit4.5ai_pdbbind_origin_test'

        model_name = data_name + '_' + model + '_' + diffusion + '_' + gnn + f'_step{step}' #
        step = step - 1
    
    
    #/data/fan_zg/MDocking/EcDock_Evaluate/evaluate_model/ecdock_refine/posebustersv1/posebustersv1_257_cross0_force0_optim0_step5_conf40_glide_refine

    #RTMScoreecdock
    elif model == 'ecdock_rtm':
        file_path = f'/data/fan_zg/MDocking/EcDock_Evaluate/evaluate_model/ecdock_refine/{data_name}/posebustersv1_257_cross0_force0_optim0_step5_conf40_glide_refine'
        #file_path = '/data/fan_zg/MDocking/EcDock_Evaluate/evaluate_model/ecdock_refine/posebustersv1/posebustersv1_257_cross0_force0_optim0_step5_conf40_glide_refine'
        model_name = data_name + '_' + model + '_' + diffusion + '_' + gnn + f'_step{step}' #
        step = step - 1


    elif model == 'ecdock_rtm_refine':
        file_path = f'/data/fan_zg/MDocking/EcDock_Evaluate/evaluate_model/ecdock_refine/{data_name}/posebustersv1_257_cross0_force0_optim0_step5_conf40_glide_refine'
        #file_path = '/data/fan_zg/MDocking/EcDock_Evaluate/evaluate_model/ecdock_refine/posebustersv1/posebustersv1_257_cross0_force0_optim0_step5_conf40_glide_refine'
        model_name = data_name + '_' + model + '_' + diffusion + '_' + gnn + f'_step{step}' #
        step = step - 1
    
    #Glide ，ecdock
    elif model == 'ecdock_score_glide':
        #/data/fan_zg/MDocking/Glide/glide/posebustersv1/model/ecdock_step5/glide_docking_score_sort_refine
        file_path = f'/data/fan_zg/MDocking/Glide/glide/{data_name}/model/ecdock_step5/glide_score_sort_refine'
        model_name = data_name + '_' + model + '_' + diffusion + '_' + gnn + f'_step{step}' #
        step = step - 1


    #Glide ，ecdock
    elif model == 'ecdock_score_glide_refine':
        file_path = f'/data/fan_zg/MDocking/Glide/glide/{data_name}/model/ecdock_step5/glide_docking_score_sort_refine'
        model_name = data_name + '_' + model + '_' + diffusion + '_' + gnn + f'_step{step}' #
        step = step - 1


    #Glideecdock
    elif model == 'ecdock_refine':
        file_path = f'/data/fan_zg/MDocking/Glide/glide/{data_name}/model/ecdock_step5/glide_docking_score_sort_refine'
        model_name = data_name + '_' + model + '_' + diffusion + '_' + gnn + f'_step{step}' #
        step = step - 1

    #Glide，, 
    elif model == 'glide': #，
        #file_path = '/mnt/home/fanzhiguang/47/CrossDocked2020/data/pdb2020_test_glide/docking_baseline'  #docking_baseline, 
        file_path = f'/data/fan_zg/MDocking/Glide/glide_docking/{data_name}/model/ecdock_step5/docking'
        model_name = model #


    #/data/fan_zg/MDocking/Vina/vina_docking/posebustersv1/model/ecdock_step5/docking
    #Vian，, 
    elif model == 'vina': #，
        #file_path = '/mnt/home/fanzhiguang/47/CrossDocked2020/data/pdb2020_test_glide/docking_baseline'  #docking_baseline, 
        file_path = f'/data/fan_zg/MDocking/Vina/vina_docking/{data_name}/model/ecdock_step5/docking'
        model_name = model #



    elif model == 'diffdock':
        file_path = '/mnt/home/fanzhiguang/47/DiffDock-main/results/new_user_predictions104' #user_predictions_full_protein, user_predictions_pocket400_protein
        model_name = model + '_full_protein' #




    elif model == 'KarmaDock':
        file_path = '/mnt/home/fanzhiguang/47/KarmaDock/pdbbind_result'
        mode = 'uncorrected'   #uncorrected/ff_corrected/align_corrected
        model_name = model + '_' + mode #



    elif model == 'unimol':
        box_size = 10
        #file_path = f'/mnt/home/fanzhiguang/47/unimol_docking_v2/interface/{data_name}_predict_sdf_boxsize{box_size}_fix_protein_cutoff'
        #file_path = f'/mnt/home/fanzhiguang/47/unimol_docking_v2/interface/{data_name}_predict_sdf_boxsize{box_size}'
        #file_path = '/mnt_191/fanzhiguang/47/mnt/unimol_docking_v2/interface/posebusters_predict_sdf_boxsize10_origin'
        #file_path = '/mnt_191/fanzhiguang/47/mnt/unimol_docking_v2/interface/posebusters_predict_sdf_boxsize10_random_protein_cutoff'
        file_path = '/data/fan_zg/MDocking/Docking_baseline/unimol_docking_v2/interface/casf2016_predict_sdf_boxsize10'
        #file_path = '/data/fan_zg/MDocking/unimol_docking_v2/interface/posebustersv2_predict_sdf_boxsize10'
        mode = 'uncorrected'   #uncorrected/corrected #, ，3reading failed_list: ['7RPZ', '7U0U', '7ZXV']
        model_name = data_name + '_' + model + '_' + f'box_size{box_size}_{mode}' #
        
    name_list = []
    with open(f'../CrossDocked2020/data/{data_name}/{data_name}_name.txt') as f:
        for i in f:
            name_list.append(i.strip())

    #，，，
    poor_name_list = []
    
    '''
    try:
        with open(f'../CrossDocked2020/data/{data_name}/{data_name}_poor_name.txt') as f:
            for i in f:
                poor_name_list.append(i.strip())
    except Exception as e:
        poor_name_list = None
    '''
    
    print('name_list num:', len(name_list))
        
    '''
    #
    no_optim_file_path = '/data/fan_zg/MDocking/EcDock_sample_dir/posebusters_ecdock_cm_equiformer_step15_interaction_limit4.5ai_cm_equiformer_gen_split_3.5_test_210_step15_Noforce_alldata'
    no_optim_truth_mol, no_optim_gen_mol, no_optim_data_name_list = read_file(no_optim_file_path, mode, flag = 'sdf', num = 500, step = step, model = model, name_list = name_list, poor_name_list = poor_name_list)  #，rdkit mol, step
    
    optim_file_path = '/data/fan_zg/MDocking/EcDock_sample_dir/posebusters_ecdock_cm_equiformer_step15_interaction_limit4.5ai_cm_equiformer_gen_split_3.5_test_210_step15_force_step5'
    optim_truth_mol, optim_gen_mol, optim_data_name_list = read_file(optim_file_path, mode, flag = 'sdf', num = 500, step = step, model = model, name_list = name_list, poor_name_list = poor_name_list)  #，rdkit mol, step
    optim_step = 5
    save_path  = f'optim_analysis/optim_step{optim_step}'
    print(f'{len(no_optim_data_name_list)} == {len(optim_data_name_list)}')
    #assert len(no_optim_data_name_list) == len(optim_data_name_list)
    os.makedirs(save_path, exist_ok=True)

    force_analy(no_optim_truth_mol, no_optim_gen_mol, optim_gen_mol, no_optim_data_name_list, save_path)

    exit()
    '''
    



    #sdf,truth_mollist，gen_mol2list，
    truth_mol, gen_mol, data_name_list = read_file(file_path, mode, flag = 'sdf', num = 100000, step = step, model = model, name_list = name_list, poor_name_list = poor_name_list)  #，rdkit mol, step
    print('truth_mol, gen_mol:', len(truth_mol), len(gen_mol))
    assert len(truth_mol) == len(gen_mol)


    #rmsd。401/3/5/10/40，rmsd
    resault_dict = {}
    boxplot_data_list   = [] #1,5,40
    histplot_data_dict  = {} #rmsd
    dt_name_dict = {}
    dt_name_mol_dict = {}
    for num in [1, 3, 5, 10, 25, 40, 100000][:]: #1000
        data_dict = rmsds(truth_mol, gen_mol, num, data_name_list, model, random_flag) #，num
        resault_dict[num] = ['rate, rate_min, rate_mean, rmsd_mean, rmsd_std, rsmd_mid, rmsd_max, rmsd_min:', data_dict['all']]
        if num in [1, 5, 10, 25, 40, 100000]:
            boxplot_data_list.append(data_dict['data'])

        if num == 100000:
            histplot_data_dict['rmsd_mean'] = data_dict['rmsd_mean']
            histplot_data_dict['rmsd_min']  = data_dict['rmsd_min']
            histplot_data_dict['data_name'] = data_dict['data_name']
            dt_name_dict     = data_dict['data_name_rmsd_mean'] 
            dt_name_mol_dict = data_dict['data_name_mol']

    #rmsd，
    dt_name_sorted_dict = dict(sorted(dt_name_dict.items(), key=lambda item: item[1], reverse=False))
    #data_name_mol
    dt_name_mol_sorted_dict = {}
    for k in dt_name_sorted_dict:
        dt_name_mol_sorted_dict[k] = dt_name_mol_dict[k]
    print('rmsd sorted from smallest to biggest', list(dt_name_sorted_dict.keys()))

    with open(f'resault/{model}_rmsdmean_sorted.pkl', 'wb') as f:
        dill.dump(dt_name_sorted_dict, f)

    with open(f'resault/{model}_mol_sorted.pkl', 'wb') as f:
        dill.dump(dt_name_sorted_dict, f)

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

    #rmsd
    save_path = os.path.join(path, f'{model_name}_histplot.png')
    histplot(histplot_data_dict, save_path, model_name, model)




    #exit()
    #print(json.dumps(resault_dict, indent=4))
    #JSON
    path = 'resault'
    os.makedirs(path, exist_ok=True)

    file_name = f'{model_name}_evaluate_resault.json'
    with open(os.path.join(path, file_name), 'w') as file:
        json.dump(resault_dict, file, indent=4)
    

    #
    save_path = os.path.join(path, f'{model_name}_boxplot.png')
    boxplot(boxplot_data_list, save_path, model_name)


    #rmsd
    save_path = os.path.join(path, f'{model_name}_histplot.png')
    histplot(histplot_data_dict, save_path, model_name, model)



    