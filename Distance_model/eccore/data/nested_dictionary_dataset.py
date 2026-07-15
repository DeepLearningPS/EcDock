# Copyright (c) DP Technology.
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict

import torch
from torch.utils.data.dataloader import default_collate

from . import UnicoreDataset


from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

import numpy as np





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


def _unflatten(dico):
    """Unflatten a flattened dictionary into a nested dictionary."""
    new_dico = OrderedDict()
    for full_k, v in dico.items():
        full_k = full_k.split(".")
        node = new_dico
        for k in full_k[:-1]:
            if k.startswith("[") and k.endswith("]"):
                k = int(k[1:-1])
            if k not in node:
                node[k] = OrderedDict()
            node = node[k]
        node[full_k[-1]] = v
    return new_dico




class BuildDataset(Dataset):
#class BuildDataset(UnicoreDataset):#class NestedDictionaryDataset():
    def __init__(self, data_list = None):
        #super().__init__()
        ##print('type(defn):', type(defn)) #dicts

        self.data_list = data_list
        self._len = len(data_list)


                



    def __getitem__(self, index):
        return self.data_list[index]
        



    def __len__(self):
        return self._len
    

    def collater(self, samples):
        if len(samples) == 0:
            #return {}
            return None
        sample = OrderedDict()
        
        for id_k in samples[0].keys():
            tmp_list = []
            for dt in samples:
                tmp_list.append(dt[id_k])
            sample[id_k] = default_collate(tmp_list) #tmp_list
            
        return _unflatten(sample)
    
    
    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        return np.arange(len(self), dtype=np.int64)



    @property
    def supports_prefetch(self):
        """Whether this dataset supports prefetching."""
        return False

    def attr(self, attr: str, index: int):
        return getattr(self, attr, None)
        #return getattr(self, attr, None)
        #return getattr(self, attr, 'None?')


    def batch_by_size(
        self,
        indices,
        batch_size=None,
        required_batch_size_multiple=1,
    ):
        """
        Given an ordered set of indices
        """
        from eccore.data import data_utils
        return data_utils.batch_by_size(
            indices,
            batch_size=batch_size,
            required_batch_size_multiple=required_batch_size_multiple,
        )


    def set_epoch(self, epoch):
        self.epoch = epoch #




class NestedDictionaryDataset(UnicoreDataset):
#class NestedDictionaryDataset():
    def __init__(self, defn = None, data_list = None, length = None):
        #super().__init__()
        ##print('type(defn):', type(defn)) #dicts
            
        self.data_list = data_list
        self.defn = _flatten(defn)
        first = None
        for v in self.defn.values():
            '''
            if not isinstance(d
                v,
                (
                    UnicoreDataset,
                    torch.utils.data.Dataset,
                ),
            ):
                raise ValueError("Expected Dataset but found: {}".format(v.__class__))
            '''
            first = first or v
            if len(v) > 0:
                ##print('v:', v) #<eccore.data.raw_dataset.RawArrayDataset object at 0x152eca806bf0>
                ##print('first:', first) #<eccore.data.raw_dataset.RawArrayDataset object at 0x152eca806bf0>
                ##print('len(v) == len(first):', f'{len(v)} == {len(first)}')
                assert len(v) == len(first), "dataset lengths must match" 

        if length is not None:
            self._len = length
        else:   
            self._len = len(data_list)


    def __getitem__old(self, index):
        #print('---------------------------------k------------------------------')
        #print('index:', index)
        
        tmp_dict = OrderedDict()
        for k, ds in tqdm(self.defn.items()):
            #print('k:', k)
            #print('ds:', ds)
            
            try:
                value = ds[index]  # 
                if value is None:  #  None
                    #print('Dataset:', None)
                    #raise Exception('error1') #，return
                    return None
                tmp_dict[k] = value
            except Exception as e:
                ##print(f'Error accessing index {index} for dataset {k}: {e}')
                return None
                raise Exception('error2:', e)
                
        return tmp_dict
                
    




    def __getitem__(self, index):
        return self.data_list[index]
        


    def __getitem__old2(self, index):
        
        #print('index:'. index)
        try:
            tmp_dict = OrderedDict()
            #print('---------------------------------k------------------------------')
            for k, ds in self.defn.items():
                
                #print('k:', k)
                if ds[index] is None:
                    print('NestedDictionaryDataset:', None)
                    #raise Exception(None)
                    #return None
                tmp_dict[k] = ds[index]
            return tmp_dict
            
        except Exception as e:
            ##print('ds[index]:', ds[index])
            #print('nest error:', e)
            #print('ds[index]:', ds[index]) 
            #，，，
            #，，
            #s#print('ds[index]:', ds[index])
            
            #exit()
            return None

        
    
        
        '''
        try:
            tmp_dict = OrderedDict((k, ds[index]) for k, ds in self.defn.items())
            return OrderedDict((k, ds[index]) for k, ds in self.defn.items())
        except Exception as e:
            #print('nest error:', e)
            return None
        '''

    def __len__(self):
        return self._len

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch suitable for forwarding with a Model
        """
        if len(samples) == 0:
            return {}
            #return None
        sample = OrderedDict()
        for k, ds in self.defn.items():
            
            #ds.collater，，
            
            try:
                sample[k] = ds.collater([s[k] for s in samples])
            except NotImplementedError:
                sample[k] = default_collate([s[k] for s in samples])
            
            
            
            #sample[k] = default_collate([s[k] for s in samples])
            
        return _unflatten(sample)

    @property
    def supports_prefetch(self):
        """Whether this dataset supports prefetching."""
        return any(ds.supports_prefetch for ds in self.defn.values())

    def prefetch(self, indices):
        """Prefetch the data required for this epoch."""
        for ds in self.defn.values():
            if getattr(ds, "supports_prefetch", False):
                ds.prefetch(indices)

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return all(ds.can_reuse_epoch_itr_across_epochs for ds in self.defn.values())

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        for ds in self.defn.values():
            ds.set_epoch(epoch)



class NestedDictionaryDataset_old(Dataset):
#class NestedDictionaryDataset():
    def __init__(self, defn):
        #super().__init__()
        ##print('type(defn):', type(defn)) #dicts
        
        self.defn = _flatten(defn)
        first = None
        for v in self.defn.values():
            '''
            if not isinstance(
                v,
                (
                    UnicoreDataset,
                    torch.utils.data.Dataset,
                ),
            ):
                raise ValueError("Expected Dataset but found: {}".format(v.__class__))
            '''
            first = first or v
            if len(v) > 0:
                ##print('v:', v) #<eccore.data.raw_dataset.RawArrayDataset object at 0x152eca806bf0>
                ##print('first:', first) #<eccore.data.raw_dataset.RawArrayDataset object at 0x152eca806bf0>
                ##print('len(v) == len(first):', f'{len(v)} == {len(first)}')
                assert len(v) == len(first), "dataset lengths must match" 

        self._len = len(first)


    def __getitem__old(self, index):
        #print('---------------------------------k------------------------------')
        #print('index:', index)
        
        tmp_dict = OrderedDict()
        for k, ds in tqdm(self.defn.items()):
            #print('k:', k)
            #print('ds:', ds)
            
            try:
                value = ds[index]  # 
                if value is None:  #  None
                    #print('Dataset:', None)
                    #raise Exception('error1') #，return
                    return None
                tmp_dict[k] = value
            except Exception as e:
                ##print(f'Error accessing index {index} for dataset {k}: {e}')
                return None
                raise Exception('error2:', e)
                
        return tmp_dict
                
    




    def __getitem__(self, index):
        
        try:
            #print('index:'. index) #，
            tmp_dict = OrderedDict()
            #print('---------------------------------k------------------------------')
            flags = []
            for k, ds in self.defn.items():
                tmp_dict[k] = ds[index]
                if tmp_dict[k] is None:
                    flags.append(False)
            
            if flags:
                return None
            return tmp_dict
        except Exception as e:
            return None       
        


    def __getitem__old2(self, index):
        
        #print('index:'. index)
        try:
            tmp_dict = OrderedDict()
            #print('---------------------------------k------------------------------')
            for k, ds in self.defn.items():
                
                #print('k:', k)
                if ds[index] is None:
                    print('NestedDictionaryDataset:', None)
                    #raise Exception(None)
                    #return None
                tmp_dict[k] = ds[index]
            return tmp_dict
            
        except Exception as e:
            ##print('ds[index]:', ds[index])
            #print('nest error:', e)
            #print('ds[index]:', ds[index]) 
            #，，，
            #，，
            #s#print('ds[index]:', ds[index])
            
            #exit()
            return None

        
    
        
        '''
        try:
            tmp_dict = OrderedDict((k, ds[index]) for k, ds in self.defn.items())
            return OrderedDict((k, ds[index]) for k, ds in self.defn.items())
        except Exception as e:
            #print('nest error:', e)
            return None
        '''

    def __len__(self):
        return self._len

    def collater_old(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch suitable for forwarding with a Model
        """
        if len(samples) == 0:
            #return {}
            return None
        sample = OrderedDict()
        for k, ds in self.defn.items():
            try:
                sample[k] = ds.collater([s[k] for s in samples])
            except NotImplementedError:
                sample[k] = default_collate([s[k] for s in samples])
        return _unflatten(sample)

    @property
    def supports_prefetch(self):
        """Whether this dataset supports prefetching."""
        return any(ds.supports_prefetch for ds in self.defn.values())

    def prefetch(self, indices):
        """Prefetch the data required for this epoch."""
        for ds in self.defn.values():
            if getattr(ds, "supports_prefetch", False):
                ds.prefetch(indices)

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return all(ds.can_reuse_epoch_itr_across_epochs for ds in self.defn.values())

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        for ds in self.defn.values():
            ds.set_epoch(epoch)
