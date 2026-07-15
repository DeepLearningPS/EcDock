import torch
from torch.utils.data import Subset
from .pl_pair_dataset import PocketLigandPairDataset, PDBPairDataset
from collections import defaultdict
import os

def get_dataset(config, data_flag = 'old_test', single_test = False, protein = None, ligand = None, data_name = 'pdbbind2020_r10', *args, **kwargs):
    name = config.name
    root = config.path
    data_flag = data_flag
    #print('name:', name) #name: pdbbind
    #raise Exception('stop01')
    print('------------------------------------------------')
    if name == 'pl':
        dataset = PocketLigandPairDataset(root, *args, **kwargs) 
    
    elif name == 'pdbbind':
        #raise Exception('stop1')
        dataset = PDBPairDataset(root, data_flag, single_test, protein, ligand, data_name, *args, **kwargs)
    else:
        raise NotImplementedError('Unknown dataset: %s' % name)
    print('a example data 6un3 ----------------------------------------------------------------------------------------')
    print('a example dataset[6un3]:', dataset['6un3'])
    print('len(dataset):', len(dataset)) #len(dataset): 16251, len(dataset): 16259, zmats8，index，，，
    
    print('---------------------------------------------------------------------------------------------')
    if 'split' in config: #
        print('config.split:', config.split)
        split = torch.load(config.split) # ，#index.pklid，、、
        exclude_index = dataset.exclude
        print('exclude_index:', exclude_index)
        #max_index = len(dataset.keys) -1 #2: 。2，17，，

        for i in split:
            print('i:', i)
            #train: 13750
            #valid: 1240
            #test: 104
            try:
                print(f'max {i}: {max(split[i])}')
            except Exception:
                pass
            #max train: 9icd
            #max valid:   6v1c
            #max test:  6un3

            '''
            for j in split[i]:
                if exclude_index.get(j) != None: #2，，dataset.keys，None，None
                #if j > max_index: #2，
                    print('rm:', j)
                    split[i].remove(j)
            '''
            assert '0' not in split[i]
            print(f'{i}: {len(split[i])}')
            print(f'{i}: {type(split[i])}') #<class 'list'>
        #exit()
        #subsets = {k: Subset(dataset, indices=v) for k, v in split.items()} #、、,

        #exit()
        print('----------------------------------------------------------------------------------------')
        subsets = defaultdict(list)
        for k, v in split.items():
            for i in v:
                dt = dataset[i]
                #print('dt:', dt) # []
                subsets[k].extend(dt)
                #exit()
        

        #，datasetsplit，，，
        return dataset, subsets
    else:  #
        return dataset
