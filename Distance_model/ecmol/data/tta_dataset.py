# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from functools import lru_cache
from eccore.data import BaseWrapperDataset


class TTADockingPoseDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        atoms,
        coordinates,
        pocket_atoms,
        pocket_coordinates,
        holo_coordinates,
        holo_pocket_coordinates,
        is_train=True,
        conf_size=10,
    ):
        self.dataset = dataset
        self.atoms = atoms
        self.coordinates = coordinates
        self.pocket_atoms = pocket_atoms
        self.pocket_coordinates = pocket_coordinates
        self.holo_coordinates = holo_coordinates
        self.holo_pocket_coordinates = holo_pocket_coordinates
        self.is_train = is_train
        self.conf_size = conf_size
        self.set_epoch(None)

    
    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch
    
    
    def __len__(self):
        return len(self.dataset) * self.conf_size

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int = None):
    #def __cached_item__(self, index: int):
        try:
            smi_idx = index // self.conf_size
            coord_idx = index % self.conf_size
            
            ##print('self.dataset[smi_idx]:', self.dataset[smi_idx])
            #if self.dataset[smi_idx] is None:
                #raise Exception('self.dataset[smi_idx]:', self.dataset[smi_idx])
                #return None
            #print('self.dataset[smi_idx][self.atoms]:', self.dataset[smi_idx][self.atoms])
            atoms = np.array(self.dataset[smi_idx][self.atoms])
            coordinates = np.array(self.dataset[smi_idx][self.coordinates][coord_idx])

        #try:
            pocket_atoms = np.array(
                [item[0] for item in self.dataset[smi_idx][self.pocket_atoms]]
            )
        #except TypeError: #TypeError: 'NoneType' object is not subscriptable
            #pocket_atoms = None

        #try:
            pocket_coordinates = np.array(self.dataset[smi_idx][self.pocket_coordinates][0])
            if self.is_train:
                holo_coordinates = np.array(self.dataset[smi_idx][self.holo_coordinates][0])
                holo_pocket_coordinates = np.array(
                    self.dataset[smi_idx][self.holo_pocket_coordinates][0]
                )
            else:
                holo_coordinates = coordinates
                holo_pocket_coordinates = pocket_coordinates
        #except Exception as e:
            ##print(e)
            #return None

            smi = self.dataset[smi_idx]["smi"]
            pocket = self.dataset[smi_idx]["pocket"]
            # id = self.dataset[smi_idx]['id']

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
        except Exception as e:
            #raise Exception(e)
            #print(e)
            return None

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)
        
        try:
            return self.__cached_item__(index, self.epoch)
            #return self.__cached_item__(index) #self.epoch None，Noneself.epoch
        except Exception as e:
            return None
            #raise Exception(e)
        
        
