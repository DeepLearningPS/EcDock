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


def compare_rmsd(base_path, file1, file2, file_tail_name):
    #rmsd，，，
    
    data_dict1 = {}
    data_dict2 = {}

    with open(os.path.join(base_path, file1 + file_tail_name), 'rb') as f:
        data_dict1 = dill.load(f)

    with open(os.path.join(base_path, file2 + file_tail_name), 'rb') as f:
        data_dict2 = dill.load(f)

    #
    common_keys = list(set(data_dict1.keys()) & set(data_dict2.keys()))

    rmsd_less_than_2    = []
    rmsd_greater_than_2 = []
    rmsd_greater_than_3 = []
    rmsd_greater_than_4 = []

    #rmsd
    for key in common_keys:
        if data_dict1[key] <= 2:
            rmsd_less_than_2.append(key)
        if data_dict1[key] > 2:
            rmsd_greater_than_2.append(key)
        if data_dict1[key] > 3:
            rmsd_greater_than_3.append(key)
        if data_dict1[key] > 4:
            rmsd_greater_than_4.append(key)

    with open(os.path.join(base_path, 'rmsd_less_than_2.txt'), 'w') as f:
        for item in rmsd_less_than_2:
            f.write(item + '\n')


    with open(os.path.join(base_path, 'rmsd_greater_than_2.txt'), 'w') as f:
        for item in rmsd_greater_than_2:
            f.write(item + '\n')


    with open(os.path.join(base_path, 'rmsd_greater_than_3.txt'), 'w') as f:
        for item in rmsd_greater_than_3:
            f.write(item + '\n')


    with open(os.path.join(base_path, 'rmsd_greater_than_4.txt'), 'w') as f:
        for item in rmsd_greater_than_4:
            f.write(item + '\n')

    print('---------------------------------------------------------')
    print('rmsd_less_than_2 num:', len(rmsd_less_than_2))
    #print('rmsd_less_than_2:', rmsd_less_than_2)
    print('---------------------------------------------------------')
    print('rmsd_greater_than_2 num:', len(rmsd_greater_than_2))
    #print('rmsd_greater_than_2:', rmsd_greater_than_2)
    print('---------------------------------------------------------')
    print('rmsd_greater_than_3 num:', len(rmsd_greater_than_3))
    #print('rmsd_greater_than_3:', rmsd_greater_than_3)
    print('---------------------------------------------------------')
    print('rmsd_greater_than_4 num:', len(rmsd_greater_than_4))
    #print('rmsd_greater_than_4:', rmsd_greater_than_4)


def file_copy(ecdock_path, unimol_path, base_path, file_list, model_list):
    #

    for file_name in file_list:
        data_name = []
        with open(os.path.join(base_path, file_name + '.txt'), 'r') as f:
            for line in f:
                data_name.append(line.strip())
        
        for path, model in zip([ecdock_path, unimol_path], model_list):
            for name in data_name:
                if model == 'ecdock':
                    s_path = os.path.join(path, name, 'step24/')
                else:
                    s_path = os.path.join(path, name)
                t_path = os.path.join(base_path, 'compare_rmsd', file_name, model, name) #，'/', 
                
                # ，
                if os.path.exists(t_path):
                    shutil.rmtree(t_path)
                    #print('')

                os.makedirs(t_path, exist_ok = True)
                #print('s_path:', s_path)
                #print('t_path:', t_path)
                shutil.copytree(s_path, t_path, dirs_exist_ok=True)



def rmsds(truth_mol_list, gen_mol_list, num, data_name_list):
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
            rmsd_dict['atom_num'].append(mol.GetNumAtoms())

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
    rmsd_std  = round(np.mean(rmsd_dict['rmsd_std']), 4)
    rsmd_mid  = round(np.mean(rmsd_dict['rsmd_mid']),4)
    rmsd_max  = round(np.mean(rmsd_dict['rmsd_max']), 4)
    rmsd_min  = round(np.mean(rmsd_dict['rmsd_min']), 4)
    rmsd_rate      = round(np.mean(rmsd_dict['rmsd_rate']), 4)

    data_dict['data_per']   =  rmsd_list_per #，numpy
    data_dict['data']       =  np.array(rmsd_list)
    data_dict['all']        = [rmsd_rate, rmsd_mean, rmsd_std, rsmd_mid, rmsd_max, rmsd_min]

    data_dict['rmsd_mean']  = rmsd_dict['rmsd_mean']
    data_dict['rmsd_std']   = rmsd_dict['rmsd_std']
    data_dict['rmsd_mid']   = rmsd_dict['rsmd_mid']
    data_dict['rmsd_max']   = rmsd_dict['rsmd_max']
    data_dict['rmsd_min']   = rmsd_dict['rsmd_min']
    data_dict['data_name']  = rmsd_dict['data_name']
    data_dict['atom_num']   = rmsd_dict['atom_num']

    data_dict['data_name_rmsd_mean']  = {k: v for k, v in zip(rmsd_dict['data_name'], rmsd_dict['rmsd_mean'])}
    data_dict['data_name_mol']        = {k: v for k, v in zip(rmsd_dict['data_name'], rmsd_dict['data_mol'])}
    data_dict['data_name_atom_num']   = {k: v for k, v in zip(rmsd_dict['data_name'], rmsd_dict['atom_num'])}
    

    return data_dict


def read_file(file_path, mode = None, flag = 'sdf', num = 1000, step = 24, model ='ecdock', name_list = None, poor_name_list = None):
    truth_mol_list, gen_mol_list = [], []
    failed_list = []
    data_name_list = []
    
    #/mnt/home/fanzhiguang/47/new_KGDiff-EcDock/ecdock_step25/result_0/step24/gen_ligand_0.sdf
    #Chem.rdmolfiles.SDMolSupplier(self.ligand_sdf)[0],[0], list，Chem.MolFromMolFile， ，sdf
    for i in os.listdir(file_path)[:num]:
    #for i in name_list[:num]:
    #for i in poor_name_list[:num]: #，ecdock
        path = os.path.join(file_path, i)
        if os.path.exists(path) and os.path.isdir(path) and os.listdir(path): #
            try:
                base_path   = path
                if model == 'ecdock':
                    org_sup     = Chem.rdmolfiles.SDMolSupplier(os.path.join(base_path, f'origin_ligand_{i}.sdf'))[0]
                    gen_sup     = Chem.rdmolfiles.SDMolSupplier(os.path.join(base_path, f'gen_ligand_{i}.sdf'))
                elif model == 'unimol':
                    org_sup     = Chem.rdmolfiles.SDMolSupplier(os.path.join(base_path, f'org_{i}.sdf'))[0]
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
                positions = [1,2,3,4,5], 
                widths = 0.4, 
                showmeans = True
    )

    # 
    plt.title(f'{name.upper()} RMSD')
    plt.xlabel('Sampling Num')
    plt.ylabel('Value')
    plt.xticks([1, 2, 3, 4, 5], [1, 5, 10, 25, 40])
    plt.show()
    plt.savefig(save_path)
    plt.close()




def histplot(plot_data_dict, base_save_path, name, model):
    name_list = plot_data_dict['data_name']
    for k_name in list(plot_data_dict.keys())[:]:

        if k_name in ['rmsd_min', 'rmsd_mean']:
            #print('k_name:', k_name)
            data = plot_data_dict[k_name]
            data = np.array(data)

            if k_name == 'rmsd_mean':
                print('rmsd_mean <= 2 rate:', round(np.mean(data <= 2.0), 4))
                print('rmsd_mean > 2 rate:', round(np.mean(data > 2.0), 4))
                print('rmsd_mean > 3 rate:', round(np.mean(data > 3.0), 4))
                print('rmsd_mean > 4 rate:', round(np.mean(data > 4.0), 4))
                print('rmsd_mean > 5 rate:', round(np.mean(data > 5.0), 4))

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



def histplot_atom(plot_data_dict, base_save_path, name, model):
    name_list = plot_data_dict['data_name']
    for k_name in list(plot_data_dict.keys())[:]:
        if k_name == 'atom_num':

            #print('k_name:', k_name)
            data = plot_data_dict[k_name]
            data = np.array(data)
            atom_num_dict = {}
            print('atom <= 10 rate:', round(np.mean(data <= 10), 4))
            print('atom > 10 rate:', round(np.mean(data > 10), 4))
            print('atom > 20 rate:', round(np.mean(data > 20), 4))
            print('atom > 30 rate:', round(np.mean(data > 20), 4))
            print('atom > 40 rate:', round(np.mean(data > 20), 4))

            atom_num_dict['atom <= 10 num'] = int(np.sum(data <= 10))
            atom_num_dict['atom > 10 num']  = int(np.sum(data > 10))
            atom_num_dict['atom > 20 num']  = int(np.sum(data > 20))
            atom_num_dict['atom > 30 num']  = int(np.sum(data > 30))
            atom_num_dict['atom > 40 num']  = int(np.sum(data > 40))

            file_name = f'{model}_atom_num_resault.json'
            with open(os.path.join(os.path.dirname(base_save_path), file_name), 'w') as file:
                json.dump(atom_num_dict, file, indent=4)
            
            print('atom_num_resault:', sorted(data))
            atom_dict = defaultdict(list)
            for nm in sorted(data):
                atom_dict[nm].append(nm)


            file_name = f'{model}_atom_num_resault.txt'
            with open(os.path.join(os.path.dirname(base_save_path), file_name), 'w') as file:
                file.write(f'atom num: num\n')
                for n in atom_dict:
                    file.write(f'{n}: {len(atom_dict[n])}\n')


            # 
            sns.histplot(data, bins=30, kde=True, color='red')
            #sns.histplot(data, kde=True, color='blue') #bins

            # 
            plt.title(f'{k_name.upper()} Histogram of {model.upper()}')
            plt.xlabel('Value')
            plt.ylabel('Frequency')

            # 
            #plt.show()
            save_path = os.path.join(os.path.dirname(base_save_path), os.path.splitext(os.path.basename(base_save_path))[0] + f'_{k_name}.png')
            #print('save_path:', save_path)
            plt.savefig(f'{save_path}')
            plt.close() #


def painting_plot(base_path, file_list, model_list):
    '''
    base_path   = 'resault/compare_rmsd'
    file_list   = ['rmsd_less_than_2', 'rmsd_greater_than_2', 'rmsd_greater_than_3', 'rmsd_greater_than_4']
    model_list  = ['unimol', 'ecdock']

    '''
    #./resault/compare_rmsd/rmsd_greater_than_4/unimol/8DW5/org_8DW5.sdf
    #./resault/compare_rmsd/rmsd_greater_than_4/ecdock/8DW5/org_ligand_8DW5.sdf
    for model in model_list:
        for file in file_list:
            file_path = os.path.join(base_path, file, model)
            truth_mol, gen_mol, data_name_list = read_file(file_path, model = model)  #，rdkit mol, step



            #rmsd。401/3/5/10/40，rmsd
            resault_dict = {}
            boxplot_data_list   = [] #1,5,40
            histplot_data_dict  = {} #rmsd
            dt_name_dict = {}
            dt_name_mol_dict = {}
            dt_name_atom_dict = {}
            for num in [1, 3, 5, 10, 25, 40][:]:
                data_dict = rmsds(truth_mol, gen_mol, num, data_name_list) #，num
                resault_dict[num] = ['rate, rmsd_mean, rmsd_std, rsmd_mid, rmsd_max, rmsd_min:', data_dict['all']]
                if num in [1, 5, 10, 25, 40]:
                    boxplot_data_list.append(data_dict['data'])

                if num == 40:
                    histplot_data_dict['rmsd_mean'] = data_dict['rmsd_mean']
                    histplot_data_dict['rmsd_min']  = data_dict['rmsd_min']
                    histplot_data_dict['data_name'] = data_dict['data_name']
                    histplot_data_dict['atom_num']  = data_dict['atom_num']
                    dt_name_dict     = data_dict['data_name_rmsd_mean'] 
                    dt_name_mol_dict = data_dict['data_name_mol']

            #rmsd，
            dt_name_sorted_dict = dict(sorted(dt_name_dict.items(), key=lambda item: item[1], reverse=False))
            #data_name_mol
            dt_name_mol_sorted_dict = {}
            for k in dt_name_sorted_dict:
                dt_name_mol_sorted_dict[k] = dt_name_mol_dict[k]
            #print('rmsd sorted from smallest to biggest', list(dt_name_sorted_dict.keys()))


            #exit()
            print(resault_dict)
            #JSON
            #path = 'resault'
            path = file_path
            os.makedirs(path, exist_ok=True)

            file_name = f'{model}_evaluate_resault.json'
            with open(os.path.join(path, file_name), 'w') as file:
                json.dump(resault_dict, file, indent=4)
            

            #
            save_path = os.path.join(path, f'{model}_boxplot.png')
            boxplot(boxplot_data_list, save_path, model)

            #rmsd
            save_path = os.path.join(path, f'{model}_histplot.png')
            histplot(histplot_data_dict, save_path, model, model)


            #atom
            save_path = os.path.join(path, f'{model}_histplot_atom.png')
            histplot_atom(histplot_data_dict, save_path, model, model)



if __name__ == '__main__':
    '''
    #ecdock
    base_path = 'resault'
    file1 = 'ecdock_rmsdmean_sorted'
    file2 = 'unimol_rmsdmean_sorted'
    file_tail_name = '.pkl'
    compare_rmsd(base_path, file1, file2, file_tail_name)

    #rmsd_greater_than_3.txt，ecdockunimol
    ecdock_path = '/mnt/home/fanzhiguang/47/new_KGDiff-EcDock/posebusters_ecdock_cm_equiformer_step25_interaction_limit4.5ai_retrain_fine'
    unimol_path = '/mnt/home/fanzhiguang/47/unimol_docking_v2/interface/posebusters_predict_sdf_boxsize10_origin'
    file_list = ['rmsd_less_than_2', 'rmsd_greater_than_2', 'rmsd_greater_than_3', 'rmsd_greater_than_4']
    model_list = ['ecdock', 'unimol']
    file_copy(ecdock_path, unimol_path, base_path, file_list[:], model_list)
    '''


    #rmsd, (sdf)，<2>2，？
    #./resault/compare_rmsd/rmsd_greater_than_4/unimol/8DW5/org_8DW5.sdf
    #./resault/compare_rmsd/rmsd_greater_than_4/ecdock/8DW5/org_ligand_8DW5.sdf

    base_path   = 'resault/compare_rmsd'
    file_list   = ['rmsd_less_than_2', 'rmsd_greater_than_2', 'rmsd_greater_than_3', 'rmsd_greater_than_4']
    model_list  = ['ecdock', 'unimol']

    #rmsd
    painting_plot(base_path, file_list, model_list)
