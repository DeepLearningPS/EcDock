import os
import sys
import traceback
sys.path.append(os.path.abspath('./'))

import pickle
import lmdb
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from KGDiff.utils.data import PDBProtein, parse_sdf_file
from KGDiff.datasets.pl_data import ProteinLigandData, torchify_dict
from KGDiff.scripts.data_preparation.clean_crossdocked import TYPES_FILENAME
import torch
import numpy as np
from collections import defaultdict
from collections import defaultdict
from ordered_set import OrderedSet
import dill

import gzip
import shutil

from Bio.PDB import PDBParser, PDBIO, Select
from Bio import PDB

from rdkit import Chem
from rdkit.Chem.rdchem import BondType
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
import copy
from rdkit.Chem import AllChem

from EcDock.comparm import *
from multiprocessing import Pool

import torch.multiprocessing as tmp
#tmp.set_start_method('spawn', force=True)  # 
import math
from itertools import islice
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from functools import partial

import os
import lmdb
import pickle
import shutil
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from rdkit import Chem
from functools import partial
from multiprocessing import Pool, Lock, Manager
from filelock import FileLock  # 


import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed


# （）， multiprocessing  pickle 
import os
import shutil
import pickle
import traceback
import numpy as np
from rdkit import Chem


import tempfile

import torch.multiprocessing as tmp
tmp.set_sharing_strategy('file_system')


    
class PocketLigandPairDataset(Dataset):

    def __init__(self, raw_path, transform=None, version='final'):
        super().__init__()
        self.raw_path = raw_path.rstrip('/')
        self.index_path = os.path.join(self.raw_path, 'index.pkl')
        self.processed_path = os.path.join(os.path.dirname(self.raw_path),
                                           os.path.basename(self.raw_path) + f'_processed_{version}.lmdb')
        self.raw_affinity_path = os.path.join('/data/qianhao', TYPES_FILENAME)
        self.affinity_path = os.path.join('data', 'affinity_info_complete.pkl')
        self.transform = transform
        self.db = None
        self.keys = None
        self.affinity_info = None
        
        if not os.path.exists(self.processed_path):
            print(f'{self.processed_path} does not exist, begin processing data')
            self._process()
        
        # len
        if self.db is None:
            self._connect_db()
        self.lengths =  len(self.keys)

        #dataset,
        self.datas_list = []
        for idx in range(self.lengths):
            data = self.get_ori_data(idx)
            if self.transform is not None:
                data = self.transform(data)
            
            self.datas_list.append(data)

            
    def _load_affinity_info(self):
        if self.affinity_info is not None:
            return
        if os.path.exists(self.affinity_path):
            with open(self.affinity_path, 'rb') as f:
                affinity_info = pickle.load(f)
        else:
            affinity_info = {}
            with open(self.raw_affinity_path, 'r') as f:
                for ln in tqdm(f.readlines()):
                    # <label> <pK> <RMSD to crystal> <Receptor> <Ligand> # <Autodock Vina score>
                    label, pk, rmsd, protein_fn, ligand_fn, vina = ln.split()
                    ligand_raw_fn = ligand_fn[:ligand_fn.rfind('.')]
                    affinity_info[ligand_raw_fn] = {
                        'label': float(label),
                        'rmsd': float(rmsd),
                        'pk': float(pk),
                        'vina': float(vina[1:])
                    }
            # save affinity info
            with open(self.affinity_path, 'wb') as f:
                pickle.dump(affinity_info, f)
        
        self.affinity_info = affinity_info
        
    def _connect_db(self):
        """
            Establish read-only database connection
        """
        assert self.db is None, 'A connection has already been opened.'
        self.db = lmdb.open(
            self.processed_path,
            map_size=100*(1024*1024*1024),   # 10GB
            
            readonly=True, lock=False, readahead=False, meminit=False, subdir=False,
            
            #create=False,
            #subdir=False,
            #readonly=False,
            #lock=False,
            #readahead=False,
            #meminit=False,
        )
        with self.db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))

    def _close_db(self):
        self.db.close()
        self.db = None
        self.keys = None
        
    def _process(self):
        db = lmdb.open(
            self.processed_path,
            map_size=300*(1024*1024*1024),   # 10GB
            create=True,
            subdir=False,
            readonly=False,  # Writable
        )
        with open(self.index_path, 'rb') as f:
            index = pickle.load(f)

        num_skipped = 0
        with db.begin(write=True, buffers=True) as txn:
            for i, (pocket_fn, ligand_fn, *_) in enumerate(tqdm(index)):
                if pocket_fn is None: continue
                try:
                    data_prefix = self.raw_path
                    pocket_dict = PDBProtein(os.path.join(data_prefix, pocket_fn)).to_dict_atom()
                    ligand_dict = parse_sdf_file(os.path.join(data_prefix, ligand_fn))
                    data = ProteinLigandData.from_protein_ligand_dicts(
                        protein_dict=torchify_dict(pocket_dict),
                        ligand_dict=torchify_dict(ligand_dict),
                    )
                    data.protein_filename = pocket_fn
                    data.ligand_filename = ligand_fn
                    data = data.to_dict()  # avoid torch_geometric version issue
                    txn.put(
                        key=str(i).encode(),
                        value=pickle.dumps(data)
                    )
                except:
                    num_skipped += 1
                    print('Skipping (%d) %s' % (num_skipped, ligand_fn, ))
                    continue
        db.close()
    
    def __len__(self):
        return self.lengths

    def __getitem__(self, idx):
        return self.datas_list[idx]
    
    def _update(self, sid, affinity):
        if self.db is None:
            self._connect_db()
        txn = self.db.begin(write=True)
        data = pickle.loads(txn.get(sid))
        data.update({
            'affinity': affinity['vina'],
            'rmsd': affinity['rmsd'],
            'pk': affinity['pk'],
            'rmsd<2': affinity['label']
        })
        txn.put(
            key=sid,
            value=pickle.dumps(data)
        )
        txn.commit()

    def _inject_affinity(self, sid, ligand_path):
        if ligand_path[:-4] in self.affinity_info:
            affinity = self.affinity_info[ligand_path[:-4]]
            self._update(sid, affinity)
        else:
            raise AttributeError(f'affinity_info has no {ligand_path[:-4]}')
            
    def get_ori_data(self, idx):
        if self.db is None:
            self._connect_db()
        key = self.keys[idx]
        data = pickle.loads(self.db.begin().get(key))
        if 'affinity' not in data:
            self._load_affinity_info()
            self._inject_affinity(key, data['ligand_filename'])
            data = pickle.loads(self.db.begin().get(key))
        
        data = ProteinLigandData(**data)
        data.id = idx
        assert data.protein_pos.size(0) > 0
        return data




def is_standard_amino_acid(residue):
    standard_aa = {
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
        'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
    }
    return residue.get_resname() in standard_aa

def is_metal_ion(residue):
    # Common metal ions in PDB files
    metal_ions = {'NA', 'K', 'MG', 'CA', 'MN', 'FE', 'CO', 'NI', 'CU', 'ZN'}
    return residue.get_resname() in metal_ions

def is_water(residue):
    return residue.get_resname() == 'HOH'




def pdb2020_filter_pdb(path, data_name):
    #base_path = data_name  #'posebusters'
    #data_name: posebusters/5SAK/5SAK_protein.pdb

        pdb_file = os.path.join(path, data_name)
        output_pdb_file = pdb_file


        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure('protein', pdb_file)
        
        io = PDB.PDBIO()
        io.set_structure(structure)
        
        class NonStandardResidueSelect(PDB.Select):
            def accept_residue(self, residue):
                return is_standard_amino_acid(residue) and not is_metal_ion(residue) and not is_water(residue)
        
        io.save(output_pdb_file, NonStandardResidueSelect())

class SkipItemException(Exception):
    pass

class PDBPairDataset(Dataset): #from torch.utils.data import Dataset PyG

    def __init__(self, raw_path, data_flag='new_test', single_test = False, protein = None, ligand = None, data_name = 'tmpdata', cross_distance_num = None, \
        st_i = None, end_i = None, transform=None, version='final'):
        super().__init__()
        self.raw_path = os.getcwd()
        self.data_flag = data_flag
        self.data_name = data_name
        self.index_path = os.path.join(self.raw_path, self.data_name, 'index.pkl')
        self.cross_distance_num = cross_distance_num
        self.st_i = st_i 
        self.end_i = end_i
        #v3，v43d，v3220

        self.processed_path = os.path.join(self.raw_path, self.data_name, 
                                    f'{self.data_name}.lmdb')

        print('self.processed_path:', self.processed_path)

        self.transform = transform
        self.db = None
        self.keys = None
        self.exclude = {} #
        self.exclude_name = OrderedSet() #
        self.nameid2id_dict = {} #name_id:id

        if not os.path.exists(self.processed_path):
            print(f'{self.processed_path} does not exist, begin processing data')
            self._process_mutiple()
            #self._process_single()
        else:
            # （）
            size_in_bytes = os.path.getsize(self.processed_path)
            #  MB
            size_in_mb = size_in_bytes / (1024 * 1024)

            self._process_mutiple()
            pass
            
            '''
            if size_in_mb < 2:
                print('<10MB, begin processing data')
                self._process_mutiple()
                #self._process_single()
            #else:
                #self._process()
                #exit()
            '''
        
        
        #zmats，index，，zmats，：zmats，，
        #de,

        name2id_dict_file = os.path.join(self.raw_path, self.data_name, f'{self.data_name}_name2id_dict.txt')


        with open(name2id_dict_file) as f:
            for i in f:
                tg = i.strip('\n').split('\t')
                if tg[1] == '0':
                    print("tg[1] == '0':", i)
                self.nameid2id_dict[tg[0]] = tg[1]

        self.name2nameid = defaultdict(list) #name，keyname_id, ，name:nameid_list 
        for i in self.nameid2id_dict:
            tg = i.rsplit('_', 1) #name, id
            assert len(tg) == 2
            if tg[1] == '0':
                print("tg[1] == '0':", i)
            self.name2nameid[tg[0]].append(i)
            

        
        assert '0' not in list(self.name2nameid.keys()) # '0'？

        print('self.nameid2id_dict num:', len(self.nameid2id_dict))
        print('self.name2nameid num:', len(self.name2nameid))

        print('self.processed_path:', self.processed_path)

        '''
        # len
        if self.db is None: #，
            self._connect_db()
        self.lengths =  len(self.keys)

        print('self.keys:', self.keys)
        print('self.keys:', type(self.keys)) #type: list
        print('self.keys:', len(self.keys)) #self.keys: 16251

        #dataset,
        self.datas_list = []
        for idx in range(self.lengths): #idrange(self.lengths)，self.keys
            data = self.get_ori_data(idx)
            if self.transform is not None:
                data = self.transform(data) #，zmats
            
            self.datas_list.append(data)
        '''

        #exit()
    def _connect_db(self):
        """
            Establish read-only database connection
        """
        assert self.db is None, 'A connection has already been opened.'
        print('self.processed_path:', self.processed_path)
        self.db = lmdb.open(
            self.processed_path,
            map_size=30*(1024*1024*1024),   # 100GB #，.。，
            
            readonly=True, lock=False, readahead=False, meminit=False, subdir=False,
            #create=False,
            #subdir=False,
            #readonly=False,
            #lock=False,
            #readahead=False,
            #meminit=False,
        )
        #with self.db.begin() as txn:
            #self.keys = list(txn.cursor().iternext(values=False)) #，values=False，。，，？
            #self.keys = list(txn.cursor())
        
        with open(os.path.join(os.path.dirname(self.processed_path), 'exclude_index.txt'))as f:
            for i in f:
                self.exclude[int(i.strip('\n')) - 2] = int(i.strip('\n')) - 2

    def _close_db(self):
        self.db.close()
        self.db = None
        self.keys = None


    def _process_single(self):
        #
        db = lmdb.open(
            self.processed_path,
            map_size=300*(1024*1024*1024),   # 500GB
            create=True,
            subdir=False,
            readonly=False,  # Writable
        )
        with open(self.index_path, 'rb') as f:
            index = pickle.load(f)

        error_file = open(os.path.join(os.path.dirname(self.processed_path), 'error_file.txt'), 'w')
        num_skipped = 0
        num_pocket  = 0
        num_zmats   = 0
        num_success = 0
        
        count = 0
        nameid2id_dict = {}


        with db.begin(write=True, buffers=True) as txn:
            error_list = [
                'posebusters/5SAK/5SAK_protein.pdb',
                'posebusters/6YDY/6YDY_protein.pdb',
                'posebusters/6YQV/6YQV_protein.pdb',
                'posebusters/6YRV/6YRV_protein.pdb',
                'posebusters/7BA0/7BA0_protein.pdb',
                'posebusters/7LMO/7LMO_protein.pdb',
                'posebusters/7POM/7POM_protein.pdb',
                'posebusters/7PRM/7PRM_protein.pdb',
                'posebusters/7U0U/7U0U_protein.pdb',
                'posebusters/7ZXV/7ZXV_protein.pdb',
                'posebusters/8D5D/8D5D_protein.pdb',
                'posebusters/8DP2/8DP2_protein.pdb',
                'posebusters/7S9H/7S9H_protein.pdb'

            ]
            print('len(index):', len(index))
            for i, (pocket_fn, protein_fn, (pka, year, resl), ligand_fn, pdbid) in enumerate(tqdm(index[:])): #100
                
                
                #pocket_fn = protein_fn
                #if not os.path.isfile(os.path.join(self.raw_path, pocket_fn)):
                    #pocket_fn = protein_fn
            
                #if self.data_flag == 'new_test':
                    #pocket_fn = protein_fn #，，。，
                
                #，，，：
                #print('os.path.exists(os.path.join(self.raw_path, pocket_fn):', os.path.join(self.raw_path, pocket_fn))
                if not os.path.exists(os.path.join(self.raw_path, pocket_fn)):
                    pocket_fn = protein_fn

                print('pocket_fn:', pocket_fn) #data/O00329-6PYR/O00329-6PYR/active0/active0_protein.pdb
                print('ligand_fn:', ligand_fn) #data/O00329-6PYR/O00329-6PYR/active0/active0_ligand.sdf
                #exit()

                #if pocket_fn is None or pocket_fn in ['posebusters/5SAK/5SAK_protein.pdb', 'posebusters/6YDY/6YDY_protein.pdb', 'posebusters/6YQV/6YQV_protein.pdb']: 
                if pocket_fn is None:
                #if pocket_fn not in error_list:
                    continue
                

                '''    
                #，
                if self.data_name == 'pdbbind2020_r10' and not os.path.isfile(os.path.join(self.raw_path, pocket_fn)):
                    s_file = os.path.join('../CrossDocked2020/data/pdbbind2020/pdbbind2020',  '/'.join(pocket_fn.split('/')[1:]))
                    t_file = os.path.join(self.raw_path, pocket_fn)
                    try:
                        shutil.copy(s_file, t_file)
                    except Exception as e:
                        print(e)
                        exit()
                '''

                


                #
                #pdb2020_filter_pdb(self.raw_path, pocket_fn)



            #try:
                data_prefix = self.raw_path
                ligand_dict = parse_sdf_file(os.path.join(data_prefix, ligand_fn), self.data_flag)
                try:
                    ligand_centor = np.mean(ligand_dict['pos'], axis=0)
                except TypeError as e:
                    print(e)
                    raise SystemExit

                #n = os.path.basename(pocket_fn).rsplit("_protein.pdb", 1)[0]
                #try:
                #unimol_pcoords = unimol_pcoords_dict[n] #unimol
                #except Exception as e:
                    #print(e)
                    #print('unimol_pcoords_dict.keys:', list(unimol_pcoords_dict.keys()))
                    #print('error key:', n)
                    #exit()
                    #continue
                #pocket_dict = PDBProtein(os.path.join(data_prefix, pocket_fn), ligand_centor, ligand_dict, unimol_pcoords).to_dict_atom_unimol() # 
                
                #pocket_dict = PDBProtein(os.path.join(data_prefix, pocket_fn), ligand_centor, ligand_dict, self.data_flag, unimol_pcoords = 
                                        #None).to_dict_atom_interaction() #
                
                #
                #pocket_dict = PDBProtein(os.path.join(data_prefix, pocket_fn), ligand_centor, ligand_dict, self.data_flag, unimol_pcoords = 
                                        #None).to_dict_atom_interaction_org() #



                #
                #pocket_dict = PDBProtein(os.path.join(data_prefix, pocket_fn), ligand_centor, ligand_dict, self.data_flag, unimol_pcoords = 
                                        #None).to_dict_atom_interaction_v2() #, ，-，
                
                #，，knn，id。，40，

                #
                #pocket_dict = PDBProtein(os.path.join(data_prefix, pocket_fn), ligand_centor, ligand_dict, self.data_flag, unimol_pcoords = 
                                        #None).to_dict_atom_interaction_v2_org()


                #，<3.53.5~4.5
                #<3.53.5~4.5，，4，，，，20，2，
                #pocket_dict = PDBProtein(os.path.join(data_prefix, pocket_fn), ligand_centor, ligand_dict, self.data_flag, self.cross_distance_num, unimol_pcoords = 
                                        #None).to_dict_atom_interaction_gen_split3_5()
                
                
                
                
                #，3.5~4.5，，2ai，，，unimol，
                #
                
                pocket_dict = PDBProtein(os.path.join(data_prefix, pocket_fn), ligand_centor, ligand_dict, self.data_flag, self.cross_distance_num, unimol_pcoords = 
                                        None).to_dict_atom_interaction_gen_split3_5_extend()
                
                
                
                #-, Bindingnet v2
                '''
                pocket_dict = PDBProtein(os.path.join(data_prefix, pocket_fn), ligand_centor, ligand_dict, self.data_flag, self.cross_distance_num, unimol_pcoords = 
                                        None).to_dict_atom_not_interaction_gen_split3_5_extend()
                '''
                
                
                
                #，，3D, 
                #pocket_dict = PDBProtein(os.path.join(data_prefix, pocket_fn), ligand_centor, ligand_dict, self.data_flag, self.cross_distance_num, unimol_pcoords = 
                                        #None).to_dict_atom_interaction_gen_split3_5_coords_connection()
                

                data = ProteinLigandData.from_protein_ligand_dicts(
                    protein_dict=torchify_dict(pocket_dict),
                    ligand_dict=torchify_dict(ligand_dict),
                )
                protein_file = os.path.join(data_prefix, pocket_fn)
                sub_protein_file = os.path.join(os.path.dirname(protein_file), os.path.splitext(os.path.basename(protein_file))[0] + '.pdb') 
                l_mol = ligand_dict['mol']
                p_mol = Chem.MolFromPDBFile(sub_protein_file,removeHs=True,sanitize=False)
                complex_mol = Chem.CombineMols(p_mol, l_mol)
                data.complex_mol = complex_mol

                output_file = os.path.join(os.path.dirname(protein_file), os.path.splitext(os.path.basename(protein_file))[0] + '_complex.pdb')
                with Chem.PDBWriter(output_file) as writer:
                    writer.write(complex_mol)

                data.protein_filename = pocket_fn
                data.ligand_filename = ligand_fn
                data.affinity = pka
                
                n = os.path.basename(pocket_fn).rsplit("_protein", 1)[0] #os.path.basename(self.protein_file).rsplit("_protein.pdb", 1)[0]
                data.name = n #
                data = data.to_dict()  # avoid torch_geometric version issue
                assert data['protein_pos'].size(0) > 0 #zmats，


                txn.put(
                    key=(n + '_' + str(i)).encode(), #lmdbint. ，+id，id
                    value=pickle.dumps(data)
                )
                num_success += 1
                nameid2id_dict[n + '_' + str(i)] = str(i)
            
            
            #except (Exception, AssertionError, ValueError, TypeError, OSError, SystemExit) as e:
            #except Exception as e: #SystemExitException
            
            #except (FileNotFoundError, Exception, SystemExit) as e:
            #except (FileNotFoundError, SystemExit) as e:
                #print(':', e)
                #raise e
                #continue
                
                self.exclude[count] = count
                self.exclude_name.add(protein_fn)
                data = None
                n = os.path.basename(pocket_fn).rsplit("_protein", 1)[0]
                txn.put(
                key=(n + '_' + str(i)).encode(), #lmdbint
                value=pickle.dumps(data)
                )
                count += 1
                nameid2id_dict[n + '_' + str(i)] = str(i)


                self.exclude_name.add(protein_fn)
                error_file.write(f'error: {e}\n')
                error_file.write(f'type(e): {type(e)}\n')   
                print('error:', e)
                print(f": {type(e)}")
                num_skipped += 1
                error_file.write('Skipping ligand_fn (%d) %s \n' % (num_skipped, ligand_fn))
                error_file.write('Skipping pocket_fn (%d) %s \n' % (num_skipped, pocket_fn))
                print('Skipping (%d) %s' % (num_skipped, ligand_fn, ))
                print('complex name:', n)
                #exit()
                #continue
                    
                
                    
                    
                        
            
                
            
                    
                
                
                
                
                
                
                
            
                

        error_file.write(f'num_skipped: {num_skipped}\n') 
        error_file.write(f'num_pocket: {num_pocket}\n')
        error_file.write(f'num_zmats: {num_zmats}\n')
        error_file.write(f'num_success: {num_success}\n')
        error_file.write(f'num_error: {count}\n')
        error_file.close()
        print('num_skipped:', num_skipped)
        print(f'num_pocket: {num_pocket}')
        print(f'num_zmats: {num_zmats}')
        print(f'num_success: {num_success}')
        print(f'num_error: {count}')

        self.nameid2id_dict = nameid2id_dict


        file_name = os.path.join(os.path.dirname(self.processed_path), f'{self.data_name}_name2id_dict.txt')


        with open(file_name, 'w')as f:
            for k, v in nameid2id_dict.items():
                f.write(k + '\t' + v + '\n')

        db.close()

        with open(os.path.join(os.path.dirname(self.processed_path), 'exclude_index.txt'), 'w') as f:
            for i in self.exclude.keys():
                f.write(str(i) + '\n')

        with open(os.path.join(os.path.dirname(self.processed_path), 'exclude_name.txt'), 'w') as f:
            for i in self.exclude_name:
                f.write(str(i) + '\n')
    
    '''
    def __len__(self):
        return self.lengths

    def __getitem__(self, idx):
        return self.datas_list[idx]
    #，，,init()，
    '''


    def data_parser(self, data):
        i, pocket_fn, protein_fn, pka, year, resl, ligand_fn, pdbid = data
        
        copy_pocket_fn = copy.deepcopy(pocket_fn)
        

        try:
            data_prefix = self.raw_path
            #print('ok100?')
            pocket_pos = PDBProtein(os.path.join(data_prefix, pocket_fn)).pos
            
            pocket_centor = np.mean(pocket_pos, axis=0)
            
            ligand_dict = parse_sdf_file(os.path.join(data_prefix, ligand_fn), self.data_flag, pocket_centor)
            #print('ok10?')
            try:
                ligand_centor = np.mean(ligand_dict['pos'], axis=0)
            except TypeError as e:
                raise SystemExit


            pocket_fn = protein_fn
        
            
            #，3.5~4.5，，2ai，，，unimol，
            #
            
            pocket_dict = PDBProtein(os.path.join(data_prefix, pocket_fn), ligand_centor, ligand_dict, self.data_flag, self.cross_distance_num, unimol_pcoords = 
                                    None).to_dict_atom_interaction_gen_split3_5_extend()
            
            
            #-, Bindingnet v2
            '''
            pocket_dict = PDBProtein(os.path.join(data_prefix, pocket_fn), ligand_centor, ligand_dict, self.data_flag, self.cross_distance_num, unimol_pcoords = 
                                    None).to_dict_atom_not_interaction_gen_split3_5_extend()
            '''


            data = ProteinLigandData.from_protein_ligand_dicts(
                protein_dict=torchify_dict(pocket_dict),
                ligand_dict=torchify_dict(ligand_dict),
            )
            protein_file = os.path.join(data_prefix, copy_pocket_fn)
            sub_protein_file = os.path.join(os.path.dirname(protein_file), os.path.splitext(os.path.basename(protein_file))[0] + '.pdb') 
            l_mol = ligand_dict['mol']
            p_mol = Chem.MolFromPDBFile(sub_protein_file,removeHs=True,sanitize=False)
            complex_mol = Chem.CombineMols(p_mol, l_mol)
            data.complex_mol = complex_mol

            output_file = os.path.join(os.path.dirname(protein_file), os.path.splitext(os.path.basename(protein_file))[0] + '_complex.pdb')
            with Chem.PDBWriter(output_file) as writer:
                writer.write(complex_mol)

            data.protein_filename = pocket_fn
            data.ligand_filename = ligand_fn
            data.affinity = pka
            
            n = os.path.basename(pocket_fn).rsplit("_protein", 1)[0] #os.path.basename(self.protein_file).rsplit("_protein.pdb", 1)[0]
            data.name = n #
            data = data.to_dict()  # avoid torch_geometric version issue
            assert data['protein_pos'].size(0) > 0 #zmats，
            
            return n, i, data
            
        except (FileNotFoundError, RuntimeError, Exception, SystemExit) as e: 
            #raise e
            print('error:', e)
            n = os.path.basename(pocket_fn).rsplit("_protein", 1)[0]
            return n, i, None


    def _process_mutiple(self):
        #
        db = lmdb.open(
            self.processed_path,
            map_size=30*(1024*1024*1024),   # 500GB
            create=True,
            subdir=False,
            readonly=False,  # Writable
        )
        with open(self.index_path, 'rb') as f:
            index = pickle.load(f)

        error_file = open(os.path.join(os.path.dirname(self.processed_path), 'error_file.txt'), 'w')
        num_skipped = 0
        num_pocket  = 0
        num_zmats   = 0
        num_success = 0
        
        count = 0
        nameid2id_dict = {}


        txn = db.begin(write=True, buffers=True)

        i_list = []
        pocket_fn_list = []
        protein_fn_list = []
        
        pka_list, year_list, resl_list = [], [], []
        ligand_fn_list = []
        pdbid_list = []
        for i, (pocket_fn, protein_fn, (pka, year, resl), ligand_fn, pdbid) in enumerate(tqdm(index[:])): #100
            
            
            if not os.path.exists(os.path.join(self.raw_path, pocket_fn)):
                pocket_fn = protein_fn
                
            if pocket_fn is None:
                continue
            else:
                i_list.append(i)
                pocket_fn_list.append(pocket_fn) #
                protein_fn_list.append(protein_fn)
                pka_list.append(pka)
                year_list.append(year)
                resl_list.append(resl)
                ligand_fn_list.append(ligand_fn)
                pdbid_list.append(pdbid)
                
                



        def run_parallel(i_list, pocket_fn_list, protein_fn_list, pka_list, year_list, resl_list, ligand_fn_list, pdbid_list):
            num_success, num_skipped, count = 0, 0, 0
            nameid2id_dict = {}

            # 
            tmp_dir = tempfile.mkdtemp(prefix="dock_tmp_")

            for j in range(0, len(i_list), 1000):
                si, di = j, j + 1000
                sub_content_list = list(zip(
                    range(si, di),
                    pocket_fn_list[si:di],
                    protein_fn_list[si:di],
                    pka_list[si:di],
                    year_list[si:di],
                    resl_list[si:di],
                    ligand_fn_list[si:di],
                    pdbid_list[si:di]
                ))

                # 
                with Pool(100) as pool:
                    results = list(tqdm(pool.imap(self.process_worker, sub_content_list), total=len(sub_content_list)))

                # 
                tmp_file = os.path.join(tmp_dir, f"batch_{j}.pkl")
                with open(tmp_file, "wb") as f:
                    pickle.dump(results, f)

            print("， LMDB...")

            #  LMDB
            count = 0
            for j in range(0, len(i_list), 1000):
                tmp_file = os.path.join(tmp_dir, f"batch_{j}.pkl")
                with open(tmp_file, "rb") as f:
                    results = pickle.load(f)

                for n, pocket_fn, protein_fn, ligand_fn, data in results:
                    key = n + '_' + str(count)
                    key = key.encode()

                    if data is not None:
                        txn.put(key, pickle.dumps(data))
                        num_success += 1
                    else:
                        txn.put(key, pickle.dumps(None))
                        num_skipped += 1
                        print(f"Skipping ({num_skipped}) {ligand_fn}")
                        self.exclude_name.add(ligand_fn)
                    
                    nameid2id_dict[key.decode()] = str(count)
                    count += 1

            print("，")
            txn.commit()
            db.close()
            return nameid2id_dict


        nameid2id_dict = run_parallel(i_list, pocket_fn_list, protein_fn_list, pka_list, year_list, resl_list, ligand_fn_list, pdbid_list)
        
        
        
        

        error_file.write(f'num_skipped: {num_skipped}\n') 
        error_file.write(f'num_pocket: {num_pocket}\n')
        error_file.write(f'num_zmats: {num_zmats}\n')
        error_file.write(f'num_success: {num_success}\n')
        error_file.write(f'num_error: {count}\n')
        error_file.close()
        print('num_skipped:', num_skipped)
        print(f'num_pocket: {num_pocket}')
        print(f'num_zmats: {num_zmats}')
        print(f'num_success: {num_success}')
        print(f'num_error: {count}')

        self.nameid2id_dict = nameid2id_dict


        file_name = os.path.join(os.path.dirname(self.processed_path), f'{self.data_name}_name2id_dict.txt')


        with open(file_name, 'w')as f:
            for k, v in nameid2id_dict.items():
                f.write(k + '\t' + v + '\n')

        

        with open(os.path.join(os.path.dirname(self.processed_path), 'exclude_index.txt'), 'w') as f:
            for i in self.exclude.keys():
                f.write(str(i) + '\n')

        with open(os.path.join(os.path.dirname(self.processed_path), 'exclude_name.txt'), 'w') as f:
            for i in self.exclude_name:
                f.write(str(i) + '\n')


    def process_worker(self, args):
        """，， LMDB"""
        n, pocket_fn, protein_fn, pka, year, resl, ligand_fn, pdbid = args
        n, i, data = self.data_parser((n, pocket_fn, protein_fn, pka, year, resl, ligand_fn, pdbid))
        return (n, pocket_fn, protein_fn, ligand_fn, data)
        














    
    def __len__(self):
        if self.db is None:
            self._connect_db()
        #return len(self.keys)
        return len(self.name2nameid)

    def odl__getitem__(self, idx):
        data = self.get_ori_data(idx)
        if self.transform is not None and data is not None:
            try:
                data = self.transform(data)
            except Exception as e:
                print('data:', data) #[None]
                print('transform error:', e)
                data = data
        return data

    def __getitem__(self, idx):
        data_list = self.get_ori_data(idx)
        new_data_list = []
        for data in data_list: #listNone
            if self.transform is not None and data is not None:
                new_data = self.transform(data)
                new_data_list.append(new_data)
            else:
                new_data_list.append(data)

        return new_data_list



    def get_ori_data(self, idx):
        if self.db is None:
            self._connect_db()

        #print('self.name2nameid num 1:', len(self.name2nameid)) #self.name2nameid num: 16251
        key_list = self.name2nameid[idx] #，key，data，crossdock，data_list,
        #print('idx:', idx) # 0 ?
        #print('self.nameid2id_dict num:', len(self.nameid2id_dict)) #self.nameid2id_dict num: 16251
        #print('self.name2nameid num:', len(self.name2nameid)) #self.name2nameid num: 16251，

        #print('self.nameid2id_dict:', self.nameid2id_dict[list(self.nameid2id_dict.keys())[-1]])
        #print('self.name2nameid key:', list(self.name2nameid.keys())[-1])
        #print('self.name2nameid value:', self.name2nameid[list(self.name2nameid.keys())[-1]])
        #print('key_list:', key_list)  #[]

        #data_list = [pickle.loads(self.db.begin().get(key.encode())) for key in key_list if int(key.rsplit('_', 1)[1]) >= self.st_i and int(key.rsplit('_', 1)[1]) < self.end_i]
        data_list = [pickle.loads(self.db.begin().get(key.encode())) for key in key_list]
        #print('data_list:', data_list)
        
        new_data_list = []

        for data in data_list:
            try:
                data = ProteinLigandData(**data)
                data.id = idx
                assert data.protein_pos.size(0) > 0
            except Exception as e:
                print(e)
                #print('error id:', idx) #error id: 734
                print('error indices[idx]:', idx) #error indices[idx]: 14147 #，torch data
                data = None
                new_data_list.append(data)
                continue
            
            #，
            '''
            if max(data.protein_element) > 17 or max(data.ligand_element) > 17: 
                #print('batch.protein_element:', batch.protein_element)
                #print('batch.ligand_element:', batch.ligand_element)
                print('max(batch.protein_element), max(batch.ligand_element) > 17 ?:', max(data.protein_element), max(data.ligand_element))
                #raise Exception(f'>17')
                data = None
                new_data_list.append(data)
                continue
            '''
        
            

            '''
            'atom_isring': new_atom_isring[nonzero_indices],
            'atom_isO': new_atom_isO[nonzero_indices],
            'atom_isN': new_atom_isN[nonzero_indices],
            '''


            if len(data.protein_atom_isring) == 0 and len(data.protein_atom_isO) == 0 and len(data.protein_atom_isN) == 0:  #，
                data = None
                new_data_list.append(data)
                continue

        
            if len(data.ligand_atom_isring) == 0 and len(data.ligand_atom_isO) == 0 and len(data.ligand_atom_isN) == 0:  #，
                data = None
                new_data_list.append(data)
                continue
            
            new_data_list.append(data)
        
        return new_data_list 

    def get_ori_data_old(self, idx):
        if self.db is None:
            self._connect_db()
        key = self.keys[idx] #idxself.keys？idxself.keys？，
        data = pickle.loads(self.db.begin().get(key))

        try:
            data = ProteinLigandData(**data)
            data.id = idx
            assert data.protein_pos.size(0) > 0
        except Exception as e:
            #print('error id:', idx) #error id: 734
            print('error indices[idx]:', idx) #error indices[idx]: 14147 #，torch data
            return None
    

        if max(data.protein_element) > 17 or max(data.ligand_element) > 17: 
            #print('batch.protein_element:', batch.protein_element)
            #print('batch.ligand_element:', batch.ligand_element)
            print('max(batch.protein_element), max(batch.ligand_element) > 17 ?:', max(data.protein_element), max(data.ligand_element))
            #raise Exception(f'>17')
            return None
        

        '''
        'atom_isring': new_atom_isring[nonzero_indices],
        'atom_isO': new_atom_isO[nonzero_indices],
        'atom_isN': new_atom_isN[nonzero_indices],
        '''
        if len(data.protein_atom_isring) == 0 and len(data.protein_atom_isO) == 0 and len(data.protein_atom_isN) == 0:  #，
            return None
    
        if len(data.ligand_atom_isring) == 0 and len(data.ligand_atom_isO) == 0 and len(data.ligand_atom_isN) == 0:  #，
            return None
        
        return data 
    


if __name__ == '__main__':

    dataset = PDBPairDataset('./data/pdbbind2020/')
    print(len(dataset), dataset[0])
