# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from .processor import Processor
import shutil
from ordered_set import OrderedSet
import pathlib


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


class UnimolPredictor:
    def __init__(self, model_dir, mode='single', nthreads=4, conf_size=1, cluster=False, use_current_ligand_conf=False, steric_clash_fix=False):
        self.model_dir = model_dir
        self.mode = mode
        self.nthreads = nthreads
        self.use_current_ligand_conf = use_current_ligand_conf
        self.cluster = cluster
        self.steric_clash_fix = steric_clash_fix
        self.conf_size = conf_size
        #print('self.use_current_ligand_conf:', self.use_current_ligand_conf) #False
        if self.use_current_ligand_conf:
            self.conf_size = 1
        else:
            self.conf_size = conf_size
        #print('self.conf_size:', self.conf_size) #40

    def preprocess(self, input_protein, input_ligand, input_docking_grid, output_ligand_name, output_ligand_dir):
        # process the input pocket.pdb and ligand.sdf, store in LMDB.
        preprocessor = Processor.build_processors(
            self.mode, self.nthreads, conf_size=self.conf_size, cluster=self.cluster,
            use_current_ligand_conf=self.use_current_ligand_conf)
        processed_data = preprocessor.preprocess(input_protein, input_ligand, input_docking_grid, output_ligand_name, output_ligand_dir)

        # return lmdb path
        return processed_data

    def is_file_exist_and_not_empty(self, filepath):
        # 
        if not os.path.exists(filepath):
            return False
        
        # （）
        if not os.path.isfile(filepath):
            return False
        
        # 0
        if os.path.getsize(filepath) > 1000:
            return True
        else:
            return False

    def predict(self, input_protein:str, 
                input_ligand:str, 
                input_docking_grid:str, 
                output_ligand_name:str, 
                output_ligand_dir:str, 
                batch_size:int,start_idx, end_idx, new_batch_data_name, gpu = 0):
        
        
        
        #，，
        #print('os.path.abspath(output_ligand_dir):', os.path.abspath(output_ligand_dir))
        #if os.path.isdir(os.path.abspath(output_ligand_dir)):
        
        
        
        os.makedirs(os.path.abspath(output_ligand_dir), exist_ok=True)
        #print('output_ligand_dir:', output_ligand_dir)
        #print('directory_path:', os.path.abspath(output_ligand_dir))
        if os.path.exists(os.path.join(os.path.abspath(output_ligand_dir), 'batch_data.lmdb')):
            # 
            #shutil.rmtree(os.path.abspath(output_ligand_dir))
            print(f"：{os.path.abspath(output_ligand_dir)}")
            #os.makedirs(os.path.abspath(output_ligand_dir), exist_ok=True)
            pass
        else:
            # #, lmdb，，，fail，。smiles，，
            lmdb_name = self.preprocess(input_protein, input_ligand, input_docking_grid, output_ligand_name, output_ligand_dir) #
            #exit()
        
        
        #lmdb_name = 'batch_data'

        
        lmdb_name = 'batch_data'
        #raise Exception('fist test') #，
        
        pkt_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "example_data", "dict_pkt.txt")
        mol_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "example_data", "dict_mol.txt")
        script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "ecmol", "infer.py")
        user_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "ecmol")
        ##print('pkt_data_path:', pkt_data_path)
        ##print('mol_data_path:', mol_data_path)
        # inference
        
        
        #pkt_data_pathmol_data_path，，，，？？dataloader，
        #dataloader，，.pkl，，，，，
        cmd = f' cp {pkt_data_path} {os.path.abspath(output_ligand_dir)} \n\
                cp {mol_data_path} {os.path.abspath(output_ligand_dir)} \n\
            CUDA_VISIBLE_DEVICES={gpu} python {script_path} --user-dir {user_dir} {os.path.abspath(output_ligand_dir)} --valid-subset {lmdb_name} \
            --results-path {os.path.abspath(output_ligand_dir)} \
            --num-workers 1 --ddp-backend=c10d --batch-size {batch_size} \
            --task docking_pose_v2 --loss docking_pose_v2 --arch docking_pose_v2 \
            --conf-size {self.conf_size} \
            --dist-threshold 8.0 --recycling 4 \
            --path {self.model_dir}  \
            --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
            --log-interval 50 --log-format simple --required-batch-size-multiple 1 \
            --start_idx {start_idx} \
            --end_idx {end_idx}'

        #print('cmd:\n', cmd)

        os.system(cmd)
        
        #exit()
        
        
        
        '''
        cp /mnt/home/fanzhiguang/47/unimol_docking_v2/example_data/dict_pkt.txt /mnt/home/fanzhiguang/47/unimol_docking_v2/interface/posebusters_predict_sdf 
        cp /mnt/home/fanzhiguang/47/unimol_docking_v2/example_data/dict_mol.txt /mnt/home/fanzhiguang/47/unimol_docking_v2/interface/posebusters_predict_sdf 
            CUDA_VISIBLE_DEVICES="0" python /mnt/home/fanzhiguang/47/unimol_docking_v2/unimol/infer.py --user-dir \
                /mnt/home/fanzhiguang/47/unimol_docking_v2/unimol /mnt/home/fanzhiguang/47/unimol_docking_v2/interface/posebusters_predict_sdf \
                --valid-subset batch_data \
                --results-path /mnt/home/fanzhiguang/47/unimol_docking_v2/interface/posebusters_predict_sdf \
                --num-workers 8 --ddp-backend=c10d --batch-size 1 \
                --task docking_pose_v2 --loss docking_pose_v2 --arch docking_pose_v2 \
                --conf-size 1 \
                --dist-threshold 8.0 --recycling 4 \
                --path ../premodel/unimol_docking_v2_240517.pt \
                --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
                --log-interval 50 --log-format simple \
                --required-batch-size-multiple 1
        '''
        
        
        
        '''
        lmdbpkl，，，
        .pkllmdb，，
        '''
        
        
        #exit()
        #
        lmdb_name = 'batch_data'
        pkl_file  = os.path.join(os.path.abspath(output_ligand_dir), lmdb_name + '.pkl')
        new_pkl_file  = os.path.join(os.path.abspath(output_ligand_dir), 'new_' + lmdb_name + '.pkl')
        lmdb_file = os.path.join(os.path.abspath(output_ligand_dir), lmdb_name+'.lmdb')
        new_lmdb_file = os.path.join(os.path.abspath(output_ligand_dir), 'new_' + lmdb_name + '.lmdb')
        
        
        '''
        ###lmdb，lmdb
        '''
        
        
        #，，
        #pkl_data = pd.read_pickle(pkl_file)

        with open(pkl_file, 'rb') as f:
            pkl_data = dill.load(f)


    
        pkl_data_dict = {}
        for dt in pkl_data:
            ##print('dt.keys:', dt.keys())
            name = dt['pocket_name'][0].split('/')[-2]
            #[/mnt_191/fanzhiguang/47/mnt/CrossDocked2020/data/pdbbind2020_r10/pdbbind2020_r10/3d7g/3d7g_protein.pdb',...]
            pkl_data_dict[name] = dt
        
        #print('len(pkl_data):', len(pkl_data)) #307
        #print('len(pkl_data_dict):', len(pkl_data_dict)) #307
        



        #，pkllmb，lmb，。
        #lmdb，lmdb
        error_protein = OrderedSet()
        '''
        with open('protein_fail.txt') as f:
            for i in f:
                tg = i.split('/')[-2]    #'[mnt/home/fanzhiguang/47/CrossDocked2020/data/pdbbind2020_r10/pdbbind2020_r10/3gqo/3gqo_protein.pdb]'
                error_protein.add(tg)
        '''
        #print('error_protein num:', len(error_protein))


        #,lmdbpkl
        data_dict = self.load_lmdb_data(lmdb_file, "mol_list", start_idx, end_idx)
        new_data_list = []
        new_pkl_list  = []
        name_list     = []
        
        #print('len(data_list):', len(data_list))
    
        current_name_list = list(pkl_data_dict.keys())       
        # pkl_data_dictdata_list，
        for name in current_name_list:
            try:
                assert pkl_data_dict[name] #None
                new_pkl_list.append(pkl_data_dict[name])
            except (KeyError, AssertionError) as e:
                print('KeyError:', name, e)
                continue                
            new_data_list.append(data_dict[name])
            name_list.append(name)
        
        
        #print('len(name_list):', len(name_list))
        

        
        ###lmdb，lmdb
        
        #print('len(data_list):', len(data_list))
        #print('len(new_data_list):', len(new_data_list))

        #，，
        if os.path.isfile(new_lmdb_file):
            os.remove(new_lmdb_file)

        #
        self.write_lmdb(new_lmdb_file, new_data_list)

        if os.path.isfile(new_pkl_file):
            os.remove(new_pkl_file)
            
        #pickle
        
        with open(new_pkl_file, 'wb') as f:
            pkl_data = dill.dump(new_pkl_list, f)
        

        #print('change success')
        
        
        #：input_protein,  input_ligand, input_docking_grid, output_ligand_name
        new_input_protein,  new_input_ligand, new_input_docking_grid, new_output_ligand_name = [], [], [], []

        assert len(input_protein) == len(input_ligand) and len(input_docking_grid) == len(output_ligand_name)

        for name in name_list:
            for i,j,k,l in zip(input_protein, input_ligand, input_docking_grid, output_ligand_name):
                nm = i.split('/')[-2]
                if nm == name and name in j and name in k and name in l:
                    new_input_protein.append(i)
                    new_input_ligand.append(j)
                    new_input_docking_grid.append(k)
                    new_output_ligand_name.append(l)
                    break
        
        #exit()
        
        
        return new_pkl_file, new_lmdb_file, new_input_protein,  new_input_ligand, new_input_docking_grid, new_output_ligand_name, output_ligand_dir, name_list
        
        #return pkl_file, lmdb_file, input_protein,  input_ligand, input_docking_grid, output_ligand_name, output_ligand_dir, name_list


    def write_lmdb(self, outputfilename, mol_list, seed=42, result_dir="./results"):
        env_new = lmdb.open(
            outputfilename,
            subdir=False,
            readonly=False,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1,
            map_size=int(10e9),
        )
        txn_write = env_new.begin(write=True)

        ii = 0
        for inner_output in mol_list: #content_list
            txn_write.put(f"{ii}".encode("ascii"), pickle.dumps(inner_output))
            ii+=1

        txn_write.commit()
        env_new.close()




    def load_lmdb_data(self, lmdb_path, key, start_idx = 0, end_idx = None):
        env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
        )
        
        #  keys
        with env.begin() as txn:
            cursor = txn.cursor()
            all_keys = [key for key, _ in cursor]
        total_len = len(all_keys)

        if end_idx is None or end_idx > total_len:
            end_idx = total_len

        _keys = all_keys[start_idx:end_idx]
        
        txn = env.begin()
        #_keys = list(txn.cursor().iternext(values=False))
        collects = []
        collects_dict = {}
        #print('_keys:', _keys)
        
        #print('start_idx:end_idx:', start_idx, end_idx)
        #print('list(_keys):', list(_keys))
        for idx in _keys:
            #print('idx:', idx)
            datapoint_pickled = txn.get(idx)
            #print('datapoint_pickled:', datapoint_pickled)
            data = pickle.loads(datapoint_pickled)
            #collects.append(data[key])
            #collects.append(data)
            name = data["pocket"].split('/')[-2]
            #print('name:', name)
            collects_dict[name] = data
            
            
        
        '''
        #_keys = list(txn.cursor().iternext(values=False))
        collects = []
        collects_dict = {}
        #print('_keys:', _keys)
        for idx in list(range(len(_keys)))[start_idx:end_idx]:
            datapoint_pickled = txn.get(f"{idx}".encode("ascii"))
            data = pickle.loads(datapoint_pickled)
            #collects.append(data[key])
            #collects.append(data)
            name = data["pocket"].split('/')[-2]
            collects_dict[name] = data
        '''
        return collects_dict
    
    #
    def postprocess(self, output_pkl, output_lmdb, output_ligand_name, output_ligand_dir, output_ligand_dir2, input_ligand, input_protein):
        # output the inference results to SDF file
        postprocessor = Processor.build_processors(self.mode, conf_size=self.conf_size)
        #print('output_pkl, output_lmdb num:', len(output_pkl), len(output_lmdb)) #，
        #lmdbpkl()
        mol_list, smi_list, coords_predict_list, holo_coords_list, holo_center_coords_list, prmsd_score_list, fail_index, pocket_coords_list, cross_distance_list, holo_pocket_coords_list, ligand_emb_list, pocket_emb_list = postprocessor.postprocess_data_pre(output_pkl, output_lmdb)
        
        new_output_ligand_name = []
        for j in output_ligand_name:
            new_output_ligand_name.extend([j] * self.conf_size)
        
        #print('mol_list, smi_list, output_ligand_name, fail_index1, holo_coords_list, pocket_coords_list, cross_distance_list:', len(mol_list), len(smi_list), len(new_output_ligand_name), len(fail_index), len(holo_coords_list), len(pocket_coords_list), len(cross_distance_list)) #3 4 3 2
        #mol_list0，，

        mol_list = [n for i, n in enumerate(mol_list) if i not in fail_index]
        smi_list = [n for i, n in enumerate(smi_list) if i not in fail_index]
        coords_predict_list = [n for i, n in enumerate(coords_predict_list) if i not in fail_index]
        holo_center_coords_list = [n for i, n in enumerate(holo_center_coords_list) if i not in fail_index]
        prmsd_score_list = [n for i, n in enumerate(prmsd_score_list) if i not in fail_index]
        new_output_ligand_name = [n for i, n in enumerate(new_output_ligand_name) if i not in fail_index]
        output_ligand_name = list(OrderedSet(new_output_ligand_name))

        #print('mol_list, smi_list, output_ligand_name, fail_index2, holo_coords_list, pocket_coords_list, cross_distance_list:', len(mol_list), len(smi_list), len(output_ligand_name), len(fail_index), len(holo_coords_list), len(pocket_coords_list), len(cross_distance_list)) #1 2 2 2
        #mol_list, smi_list, output_ligand_name, fail_index1, holo_coords_list, pocket_coords_list, cross_distance_list: 80 80 80 0 80 80 80
        #mol_list, smi_list, output_ligand_name, fail_index2, holo_coords_list, pocket_coords_list, cross_distance_list: 80 80 2 0 80 80 80
        #raise Exception('test')
        #coords_predict_list, holo_coords_list, pocket_coords_list, cross_distance_listpostprocessor.get_sdf，，rmsd，？
        #rmsd, ecdock，
        #，，

        if not mol_list: #1，，
            #print('gen failed, skip')
            return None
        else:
            output_ligand_sdf = postprocessor.get_sdf(mol_list, smi_list, coords_predict_list, holo_center_coords_list, prmsd_score_list, output_ligand_name, output_ligand_dir, output_ligand_dir2, holo_coords_list, pocket_coords_list, holo_pocket_coords_list, cross_distance_list, ligand_emb_list, pocket_emb_list, tta_times=self.conf_size)
            #print('output_ligand_sdf num:', len(output_ligand_sdf))

            #print('output_ligand_sdf, input_protein, input_ligand:', len(output_ligand_sdf), len(input_protein), len(input_ligand))
            #
            for i, j, k in zip(output_ligand_sdf, input_protein, input_ligand):
                tg_path = os.path.dirname(i)
                ##print('i:', i)
                ##print('j:', j)
                ##print('k:', k)
                ##print('tg_path:', tg_path)
                shutil.copy2(j, tg_path)
                shutil.copy2(k, tg_path)

                #，，，
        
            
            
            
            #if self.steric_clash_fix and 'pdbbind2020' not in input_protein[0]: #，，
                ##print('')
                #try:
                    #output_ligand_sdf = postprocessor.clash_fix(output_ligand_sdf, input_protein, input_ligand)
                #except Exception as e:
                    ##print(':', e)
                    #return None
            
                

            return output_ligand_sdf

    def predict_sdf(self, input_protein:str, 
                    input_ligand:str, input_docking_grid:str, 
                    output_ligand_name:str, output_ligand_dir:str, output_ligand_dir2,
                    batch_size:int = 4, start_idx = 0, end_idx = None, new_batch_data_name = None, gpu = 0):
        
        #st_time = time.time()

        output_pkl, output_lmdb, input_protein,  input_ligand, input_docking_grid, output_ligand_name, output_ligand_dir, name_list = self.predict(input_protein, 
                                            input_ligand, 
                                            input_docking_grid, 
                                            output_ligand_name, 
                                            output_ligand_dir, 
                                            batch_size,
                                            start_idx, end_idx, new_batch_data_name, gpu = gpu)
        


        #end_time = time.time()
        #print('len(name_list):', len(name_list))
        #print('sample mean time: {}s'.format((end_time - st_time) / len(name_list))) #

        #raise Exception('stop time test')



        new_output_ligand_dir2 = []
        for name in name_list:
            for i in output_ligand_dir2:
                if name in i:
                    new_output_ligand_dir2.append(i)
                    break


        output_sdf = self.postprocess(output_pkl, 
                                    output_lmdb, 
                                    output_ligand_name, 
                                    output_ligand_dir,
                                    new_output_ligand_dir2,
                                    input_ligand,
                                    input_protein)
        
        #

        # return sdf path
        return input_protein, input_ligand, input_docking_grid, output_sdf

    #cls，
    @classmethod
    def build_predictors(cls, model_dir, mode = 'batch_one2one', 
                         nthreads = 4, conf_size =1, 
                         cluster=False, use_current_ligand_conf=False, steric_clash_fix=False):
        return cls(model_dir, mode, nthreads, conf_size, 
                   cluster, use_current_ligand_conf=use_current_ligand_conf, steric_clash_fix=steric_clash_fix)    
