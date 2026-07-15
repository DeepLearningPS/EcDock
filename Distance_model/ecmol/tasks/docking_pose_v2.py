# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import contextlib
from typing import Optional
from collections.abc import Iterable

from collections import OrderedDict

import numpy as np

from eccore.data import (
    Dictionary,
    NestedDictionaryDataset,
    AppendTokenDataset,
    PrependTokenDataset,
    RightPadDataset,
    TokenizeDataset,
    RightPadDataset2D,
    RawArrayDataset,
    FromNumpyDataset,
    EpochShuffleDataset,
)

from ecmol.data import (
    
    KeyDataset,
    LMDBDataset,
    ConformerSampleDockingPoseDataset,
    RightPadDatasetCoord,
    DistanceDataset,
    EdgeTypeDataset,
    CrossDistanceDataset,
    NormalizeDataset,
    NormalizeDockingPoseDataset,
    TTADockingPoseDataset,
    RightPadDatasetCross2D,
    CroppingPocketDataset,
    PrependAndAppend2DDataset,
    RemoveHydrogenPocketDataset,
    ReAlignLigandDataset,
    
)





from eccore import checkpoint_utils
from eccore.tasks import UnicoreTask, register_task


logger = logging.getLogger(__name__)


def _flatten_old(dico, prefix=None):
    """Flatten a nested dictionary."""
    new_dico = OrderedDict()
    if isinstance(dico, dict):
        prefix = prefix + "." if prefix is not None else ""
        for k, v in dico.items():
            if v is None:
                continue
            new_dico.update(_flatten(v, prefix + k))
    elif isinstance(dico, list):
        for i, v in enumerate(dico):
            new_dico.update(_flatten(v, prefix + ".[" + str(i) + "]"))
    else:
        new_dico = OrderedDict({prefix: dico})
    ##print('prefix:', prefix)
    return new_dico



def _flatten(dico, prefix=None):
    """Flatten a nested dictionary (non-recursive version using stack)."""
    new_dico = OrderedDict()
    stack = [(dico, prefix)]  #  (value, current_prefix) 
    
    while stack:
        current, current_prefix = stack.pop()
        
        if isinstance(current, dict):
            # ，（）
            current_prefix = (current_prefix + ".") if current_prefix is not None else ""
            for k in reversed(list(current.keys())):
                v = current[k]
                if v is not None:
                    stack.append((v, current_prefix + k))
        
        elif isinstance(current, list):
            # ，
            for i in reversed(range(len(current))):
                v = current[i]
                if v is not None:
                    stack.append((v, f"{current_prefix}.[{i}]"))
        
        else:
            # （dict/list），
            if current_prefix is not None:
                new_dico[current_prefix] = current
    
    return new_dico


@register_task("docking_pose_v2")
class DockingPoseV2(UnicoreTask):
    """Task for training transformer auto-encoder models."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument(
            "data",
            help="downstream data path",
        )
        parser.add_argument(
            "--finetune-mol-model",
            default=None,
            type=str,
            help="pretrained molecular model path",
        )
        parser.add_argument(
            "--finetune-pocket-model",
            default=None,
            type=str,
            help="pretrained pocket model path",
        )
        parser.add_argument(
            "--conf-size",
            default=10,
            type=int,
            help='number of conformers generated with each molecule'
        )
        parser.add_argument(
            "--dist-threshold",
            type=float,
            default=8.0,
            help="threshold for the distance between the molecule and the pocket",
        )
        parser.add_argument(
            "--max-pocket-atoms",
            type=int,
            default=256,
            help="selected maximum number of atoms in a pocket",
        )  
        
    def __init__(self, args, dictionary, pocket_dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.pocket_dictionary = pocket_dictionary
        self.seed = args.seed
        # add mask token
        self.mask_idx = dictionary.add_symbol("[MASK]", is_special=True)
        self.pocket_mask_idx = pocket_dictionary.add_symbol("[MASK]", is_special=True)
        
    @classmethod
    def setup_task(cls, args, **kwargs):
        mol_dictionary = Dictionary.load(os.path.join(args.data, "dict_mol.txt"))
        pocket_dictionary = Dictionary.load(os.path.join(args.data, "dict_pkt.txt"))
        logger.info("ligand dictionary: {} types".format(len(mol_dictionary)))
        logger.info("pocket dictionary: {} types".format(len(pocket_dictionary)))
        return cls(args, mol_dictionary, pocket_dictionary)

    def pre_data_iter(self, defn):
        self.defn = _flatten(defn)
        datasets = list(self.defn.values())
        first = next((v for v in datasets if len(v) > 0), None)
        assert first is not None, "No data available"
        assert all(len(v) == len(first) for v in datasets), "dataset lengths must match"
        lens = len(first)

        for idx in range(lens):
            try:
                record = OrderedDict()
                skip = False
                for k, ds in self.defn.items():
                    val = ds[idx]
                    if val is None:
                        skip = True
                        break
                    record[k] = val
                if skip:
                    continue
                if record['net_input.mol_src_distance'].shape[-1] != record['net_input.mol_src_edge_type'].shape[-1]:
                    continue
                yield record
            except Exception:
                continue




    def pre_data(self, defn):
        #，，，
        #
        self.defn = _flatten(defn)
        first = None
        for v in self.defn.values():
            first = first or v
            if len(v) > 0:
                ##print('v:', v) #<eccore.data.raw_dataset.RawArrayDataset object at 0x152eca806bf0>
                ##print('first:', first) #<eccore.data.raw_dataset.RawArrayDataset object at 0x152eca806bf0>
                ##print('len(v) == len(first):', f'{len(v)} == {len(first)}')
                assert len(v) == len(first), "dataset lengths must match" 

        lens = len(first)
        
        tmp_dict_list = []
        for index in range(lens):
            try:
                #print('index:'. index) #
                tmp_dict = OrderedDict()
                #print('---------------------------------k------------------------------')
                flags = []
                for k, ds in self.defn.items():
                    #print('k:', k)
                    ##print('ds[index]:', ds[index])
                    
                    tmp_dict[k] = ds[index]
                    if tmp_dict[k] is None:
                        flags.append(False)
                
                if flags:
                    continue
                elif tmp_dict['net_input.mol_src_distance'].shape[-1] != tmp_dict['net_input.mol_src_edge_type'].shape[-1]:
                    continue
                else:
                    tmp_dict_list.append(tmp_dict)
            except Exception as e:
                continue
        return tmp_dict_list

        
        
    def load_dataset(self, split, start_idx=0, end_idx=None, **kwargs):
        """Load a given dataset split.
        'smi','pocket','atoms','coordinates','pocket_atoms','pocket_coordinates','holo_coordinates','holo_pocket_coordinates','scaffold'
        Args:
            split (str): name of the data scoure (e.g., bppp)
        """
        #print('load_dataset:', split) # batch_data
        data_path = os.path.join(self.args.data, split + '.lmdb')
        dataset = LMDBDataset(data_path, start_idx=start_idx, end_idx=end_idx)
        
        #，。split.startswith('test') or split.startswith('valid')
        #if split.startswith('train') or split.startswith('test') or split.startswith('valid'):
        if split.startswith('train'):
            #tgt_dataset = KeyDataset(dataset, 'target')
            #smi_dataset = KeyDataset(dataset, 'smi')
            #poc_dataset = KeyDataset(dataset, 'pocket')
            dataset = ConformerSampleDockingPoseDataset(dataset, self.args.seed, 'atoms', 'coordinates', 'pocket_atoms', 'pocket_coordinates', 'holo_coordinates', 'holo_pocket_coordinates', True)
            #dataset = TTADockingPoseDataset(dataset, 'atoms', 'coordinates', 'pocket_atoms', 'pocket_coordinates', 'holo_coordinates', 'holo_pocket_coordinates', True, self.args.conf_size)
        elif split.startswith('test') or split.startswith('valid'):
            #print('self.args.conf_size test/valid:', self.args.conf_size) #10,
            #self.args.conf_size = 1 #,1，，，，，
            dataset = TTADockingPoseDataset(dataset, 'atoms', 'coordinates', 'pocket_atoms', 'pocket_coordinates', 'holo_coordinates', 'holo_pocket_coordinates', True, self.args.conf_size)
            '''
            return {
            "atoms": atoms,
            "coordinates": coordinates.astype(np.float32),
            "pocket_atoms": pocket_atoms,
            "pocket_coordinates": pocket_coordinates.astype(np.float32),
            "holo_coordinates": holo_coordinates.astype(np.float32),
            "holo_pocket_coordinates": holo_pocket_coordinates.astype(np.float32),
            "smi": smi,
            "pocket": pocket,
            # 'id': id,
            }
            '''
            
            #tgt_dataset = KeyDataset(dataset, 'target')
            #smi_dataset = KeyDataset(dataset, 'smi')
            #poc_dataset = KeyDataset(dataset, 'pocket') #pdbband
        else:
            #print('self.args.conf_size batch_data:', self.args.conf_size) #10
            #self.args.conf_size = 2
            dataset = TTADockingPoseDataset(dataset, 'atoms', 'coordinates', 'pocket_atoms', 'pocket_coordinates', 'holo_coordinates', 'holo_pocket_coordinates', True, self.args.conf_size)
            '''
            return {
            "atoms": atoms,
            "coordinates": coordinates.astype(np.float32),
            "pocket_atoms": pocket_atoms,
            "pocket_coordinates": pocket_coordinates.astype(np.float32),
            "holo_coordinates": holo_coordinates.astype(np.float32),
            "holo_pocket_coordinates": holo_pocket_coordinates.astype(np.float32),
            "smi": smi,
            "pocket": pocket,
            # 'id': id,
            }
            '''
            
            #tgt_dataset = KeyDataset(dataset, 'target')
            #smi_dataset = KeyDataset(dataset, 'smi')
            #poc_dataset = KeyDataset(dataset, 'pocket') #pdbband
            ##print('len(poc_dataset):', len(poc_dataset)) #40,40

            #for i in poc_dataset:
            #for i in range(len(poc_dataset)):
                ##print(f'poc_dataset_{i}:', poc_dataset[i])

            '''
            #，
            pocket_coordinates      = KeyDataset(dataset, 'pocket_coordinates')
            holo_pocket_coordinates = KeyDataset(dataset, 'holo_pocket_coordinates')
            #print('len(pocket_coordinates):', len(pocket_coordinates)) #40
            #print('len(holo_pocket_coordinates):', len(holo_pocket_coordinates))#40

            for i in list(range(len(pocket_coordinates)))[1:]:
                assert np.allclose(pocket_coordinates[i-1], pocket_coordinates[i], rtol=0.01, atol=0.02)

            for i in list(range(len(holo_pocket_coordinates)))[1:]:
                assert np.allclose(holo_pocket_coordinates[i-1], holo_pocket_coordinates[i], rtol=0.01, atol=0.02)

            for i in list(range(len(holo_pocket_coordinates)))[:]:
                assert np.allclose(pocket_coordinates[i], holo_pocket_coordinates[i], rtol=0.01, atol=0.02)
            '''
        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)
        
        #dataset = list(dataset)
        # None
        

        '''
        new_dataset = []
        for i in range(len(dataset)):
            if dataset[i] is not None:
                new_dataset.append(dataset[i])
            else:
                #print('None')
        dataset = new_dataset
        '''
        
        #raise Exception('test')
        
        #
        #dataset = RemoveHydrogenPocketDataset(dataset, 'pocket_atoms', 'pocket_coordinates', 'holo_pocket_coordinates', True, True) #，，，

        #dataset = list(dataset)
        # None
        
        '''
        #print('len(len(dataset)):', len(dataset))
        new_dataset = []
        for i in range(len(dataset)):
            if dataset[i] is not None:
                new_dataset.append(dataset[i])
        dataset = new_dataset
        '''
        
        
        '''
        #
        holo_coord_pocket_dataset = KeyDataset(dataset, 'holo_pocket_coordinates')
        #print('holo_coord_pocket_dataset num:', len(holo_coord_pocket_dataset)) #17280
        for i in list(range(len(holo_coord_pocket_dataset)))[1:40]:
            #1，，，
            if len(holo_coord_pocket_dataset[i-1]) ==  len(holo_coord_pocket_dataset[i]):
                assert np.allclose(holo_coord_pocket_dataset[i-1], holo_coord_pocket_dataset[i], rtol=0.01, atol=0.02)
        '''


        print('self.args.max_pocket_atoms:', self.args.max_pocket_atoms)
        #，，，，max_pocket_atoms == 256，，unicore
        dataset = CroppingPocketDataset(dataset, self.seed, 'pocket_atoms', 'pocket_coordinates', 'holo_pocket_coordinates', self.args.max_pocket_atoms)

        #raise Exception('test')
        '''
        #，，
        holo_coord_pocket_dataset = KeyDataset(dataset, 'holo_pocket_coordinates') 
        for i in list(range(len(holo_coord_pocket_dataset)))[1:40]:
            #1，，，
            if len(holo_coord_pocket_dataset[i-1]) ==  len(holo_coord_pocket_dataset[i]):
                assert np.allclose(holo_coord_pocket_dataset[i-1], holo_coord_pocket_dataset[i], rtol=0.01, atol=0.02)
        '''

        #dataset = RemoveHydrogenPocketDataset(dataset, 'atoms', 'coordinates', 'holo_coordinates', True, True) #，，，
        
        #，，，？
        
        '''
        #None
        new_dataset = []
        for dt_i in dataset:
            new_dataset.append(dt_i)
        dataset = new_dataset
        '''
        
        
        
        
        #None，dataset[i]
        #unicore/data/nested_dictionary_dataset.py，None，
        '''
        new_dataset = []
        for i in range(len(dataset)):
            if dataset[i] is not None:
                new_dataset.append(dataset[i])
        dataset = new_dataset
        '''
        

        #raise Exception('test')
        
        
        #，
        tgt_dataset = KeyDataset(dataset, 'target')
        smi_dataset = KeyDataset(dataset, 'smi')
        poc_dataset = KeyDataset(dataset, 'pocket') #pdbband
            
        
        '''
        #，
        holo_coord_pocket_dataset = KeyDataset(dataset, 'holo_pocket_coordinates')
        for i in list(range(len(holo_coord_pocket_dataset)))[1:]:
            #1，，，
            if len(holo_coord_pocket_dataset[i-1]) ==  len(holo_coord_pocket_dataset[i]):
                assert np.allclose(holo_coord_pocket_dataset[i-1], holo_coord_pocket_dataset[i], rtol=0.01, atol=0.02)
        '''
        

        apo_dataset = NormalizeDataset(dataset, 'coordinates')
        
        '''
        new_apo_dataset = []
        for i in range(len(apo_dataset)):
            if apo_dataset[i] is not None:
                new_apo_dataset.append(apo_dataset[i])
        apo_dataset = new_apo_dataset
        '''
        
        apo_dataset = NormalizeDataset(apo_dataset, 'pocket_coordinates')
        apo_dataset = ReAlignLigandDataset(dataset,'coordinates','pocket_coordinates')

        src_dataset = KeyDataset(apo_dataset, 'atoms') #
        src_dataset = TokenizeDataset(src_dataset, self.dictionary, max_seq_len=self.args.max_seq_len)
        coord_dataset = KeyDataset(apo_dataset, 'coordinates')
        src_dataset = PrependAndAppend(src_dataset, self.dictionary.bos(), self.dictionary.eos())
        edge_type = EdgeTypeDataset(src_dataset, len(self.dictionary))
        coord_dataset = FromNumpyDataset(coord_dataset)
        distance_dataset = DistanceDataset(coord_dataset)
        coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
        distance_dataset = PrependAndAppend2DDataset(distance_dataset, 0.0)

        src_pocket_dataset = KeyDataset(apo_dataset, 'pocket_atoms')
        src_pocket_dataset = TokenizeDataset(src_pocket_dataset, self.pocket_dictionary, max_seq_len=self.args.max_seq_len)
        coord_pocket_dataset = KeyDataset(apo_dataset, 'pocket_coordinates')
        src_pocket_dataset = PrependAndAppend(src_pocket_dataset, self.pocket_dictionary.bos(), self.pocket_dictionary.eos())
        pocket_edge_type = EdgeTypeDataset(src_pocket_dataset, len(self.pocket_dictionary))
        coord_pocket_dataset = FromNumpyDataset(coord_pocket_dataset)
        distance_pocket_dataset = DistanceDataset(coord_pocket_dataset)
        coord_pocket_dataset = PrependAndAppend(coord_pocket_dataset, 0.0, 0.0)
        distance_pocket_dataset = PrependAndAppend2DDataset(distance_pocket_dataset, 0.0)

        holo_dataset = NormalizeDockingPoseDataset(dataset, 'holo_coordinates', 'holo_pocket_coordinates', 'holo_center_coordinates')
        holo_coord_dataset = KeyDataset(holo_dataset, 'holo_coordinates')
        holo_coord_dataset = FromNumpyDataset(holo_coord_dataset)
        holo_coord_pocket_dataset = KeyDataset(holo_dataset, 'holo_pocket_coordinates')

        '''
        #，
        for i in list(range(len(holo_coord_pocket_dataset)))[1:]:
            #1，，，
            if len(holo_coord_pocket_dataset[i-1]) ==  len(holo_coord_pocket_dataset[i]):
                assert np.allclose(holo_coord_pocket_dataset[i-1], holo_coord_pocket_dataset[i], rtol=0.01, atol=0.02)
        '''


        holo_coord_pocket_dataset = FromNumpyDataset(holo_coord_pocket_dataset)
        ##print('holo_coord_pocket_dataset:', holo_coord_pocket_dataset)
        ##print('holo_coord_pocket_dataset:', len(holo_coord_pocket_dataset)) #40






        holo_cross_distance_dataset = CrossDistanceDataset(holo_coord_dataset, holo_coord_pocket_dataset)

        holo_distance_dataset = DistanceDataset(holo_coord_dataset)
        holo_coord_dataset = PrependAndAppend(holo_coord_dataset, 0.0, 0.0)
        holo_distance_dataset = PrependAndAppend2DDataset(holo_distance_dataset, 0.0)
        holo_coord_pocket_dataset = PrependAndAppend(holo_coord_pocket_dataset, 0.0, 0.0)
        holo_cross_distance_dataset = PrependAndAppend2DDataset(holo_cross_distance_dataset, 0.0)

        holo_center_coordinates = KeyDataset(holo_dataset, 'holo_center_coordinates')
        holo_center_coordinates = FromNumpyDataset(holo_center_coordinates)

        ##print('holo_coord_pocket_dataset:', holo_coord_pocket_dataset)
        ##print('holo_coord_pocket_dataset:', len(holo_coord_pocket_dataset)) #40
        
  

        data_dict = {
                    "net_input": {
                        "mol_src_tokens": RightPadDataset(
                            src_dataset,
                            pad_idx=self.dictionary.pad(),
                        ),
                        "mol_src_coord": RightPadDatasetCoord(
                            coord_dataset,
                            pad_idx=0,
                        ),
                        "mol_src_distance": RightPadDataset2D(
                            distance_dataset,
                            pad_idx=0,
                        ),
                        "mol_src_edge_type": RightPadDataset2D(
                            edge_type,
                            pad_idx=0,
                        ),
                        "pocket_src_tokens": RightPadDataset(
                            src_pocket_dataset,
                            pad_idx=self.pocket_dictionary.pad(),
                        ),
                        "pocket_src_coord": RightPadDatasetCoord(
                            coord_pocket_dataset,
                            pad_idx=0,
                        ),
                        "pocket_src_distance": RightPadDataset2D(
                            distance_pocket_dataset,
                            pad_idx=0,
                        ),
                        "pocket_src_edge_type": RightPadDataset2D(
                            pocket_edge_type,
                            pad_idx=0,
                        ),
                    },
                    "target": {
                        "distance_target": RightPadDatasetCross2D(holo_cross_distance_dataset, pad_idx=0),
                        "holo_coord": RightPadDatasetCoord(holo_coord_dataset, pad_idx=0),
                        "holo_distance_target": RightPadDataset2D(holo_distance_dataset, pad_idx=0),
                        "holo_coord_pocket": RightPadDatasetCoord(holo_coord_pocket_dataset, pad_idx=0),
                    },
                    "smi_name": RawArrayDataset(
                        smi_dataset
                    ),
                    "pocket_name": RawArrayDataset(
                        poc_dataset
                    ),
                    "holo_coord": RightPadDatasetCoord(
                        holo_coord_dataset,
                        pad_idx=0,
                    ),
                    "holo_coord_pocket": RightPadDatasetCoord(
                        holo_coord_pocket_dataset,
                        pad_idx=0,
                    ),
                    "holo_center_coordinates": RightPadDataset(
                        holo_center_coordinates,
                        pad_idx=0,
                    ),
                }

        print('')
        #data_list_iter = self.pre_data_iter(data_dict) #，，list
        data_list = self.pre_data(data_dict) #，，list
        #length = sum(1 for _ in data_list_iter)
        #del data_list_iter
        #print('len(data_list):', len(data_list)) #1990
        print('')
        
        
        '''
        new_data_list = []
        for i in range(len(data_list)):
            if data_list[i] is not None:
                new_data_list.append(data_list[i])
        data_list = new_data_list
        #print('len(datafter):', len(data_list)) #1990

        #exit()
        '''
        
        nest_dataset = NestedDictionaryDataset(data_dict, data_list
                
            )
        
        
        #print('split:', split) #valid/batch_data, valid/test,valid/test
        if split.startswith('train'):
            nest_dataset = EpochShuffleDataset(nest_dataset, len(nest_dataset), self.args.seed)
        self.datasets[split] = nest_dataset
        #print('nest_dataset:', nest_dataset)
        count = 0
        #print('len(nest_dataset):', len(nest_dataset))
        
        #print('')
        ##print(nest_dataset[0]) #list
        #print('len(nest_dataset):', len(nest_dataset))
        
        '''
        tmp = []
        for dt in nest_dataset:
            tmp.append(dt)
        
        nest_dataset = tmp
        '''
        
        new_nest_dataset = []
        None_num  = 0
        
        '''
            ddt.keys: odict_keys(['net_input.mol_src_tokens', 'net_input.mol_src_coord', 'net_input.mol_src_distance', 'net_input.mol_src_edge_type', 
            'net_input.pocket_src_tokens', 'net_input.pocket_src_coord', 'net_input.pocket_src_distance', 'net_input.pocket_src_edge_type', 'target.distance_target', 
            'target.holo_coord', 'target.holo_distance_target', 'target.holo_coord_pocket', 'smi_name', 'pocket_name', 'holo_coord', 'holo_coord_pocket', 
            'holo_center_coordinates'])
        '''
        
        '''
        for idx, i in enumerate(list(nest_dataset)): #40
        #for idx in range(len(nest_dataset)): #40
            ##print('nest_dataset[idx]:', nest_dataset[idx])
            #i = nest_dataset[idx]
            #print('idx?:', idx) 
            if i is None:
                None_num += 1
                #raise Exception(None)
                continue
            new_nest_dataset.append(i)
            #print('pocket_name:', i['pocket_name'])
            
            #print('i[target.distance_target]:', i['target.distance_target'].shape) #0，2
            #print('target.holo_coord_pocket:', i['target.holo_coord_pocket'].shape)
            #print('target.holo_coord_pocket:', i['target.holo_coord_pocket'])
            #print('i.key:', list(i.keys()))
            ##print(i['holo_coord_pocket'])
            ##print("len(i['holo_coord_pocket']:", len(i['holo_coord_pocket']))
            #print("i['holo_coord_pocket'].shape:", i['holo_coord_pocket'].shape) #？
            #count += 1
            #if count == 2:
                #break
        
        #print('None_num :', None_num)
        #raise Exception('test')
        '''
    
        ##print(self.datasets[split]["target"]["holo_coord_pocket"])
        
    def build_model(self, args):
        from eccore import models
        model = models.build_model(args, self)
        if args.finetune_mol_model is not None:
            #print("load pretrain model weight from...", args.finetune_mol_model)
            state = checkpoint_utils.load_checkpoint_to_cpu(
                args.finetune_mol_model, 
            )
            model.mol_model.load_state_dict(state["model"], strict=False)
        if args.finetune_pocket_model is not None:
            #print("load pretrain model weight from...", args.finetune_pocket_model)
            state = checkpoint_utils.load_checkpoint_to_cpu(
                args.finetune_pocket_model, 
            )
            model.pocket_model.load_state_dict(state["model"], strict=False)
        return model
