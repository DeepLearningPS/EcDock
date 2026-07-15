# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import lru_cache
from eccore.data import BaseWrapperDataset


class KeyDataset(BaseWrapperDataset):
    def __init__(self, dataset, key):
        self.dataset = dataset
        self.key = key

    def __len__(self):
        return len(self.dataset)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        ##print('len(self.dataset):', len(self.dataset))
        #return self.dataset[idx][self.key]
        
        try:
            return self.dataset[idx][self.key]
        except Exception as e:
            #print('KeyDataset idx:', idx) #30
            #print('self.key:', self.key) #atoms
            #print('self.dataset[idx]:', self.dataset[idx]) #，
            #raise Exception(e)
            return None
        
        
        
