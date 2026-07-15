# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import lmdb
import os
import pickle
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)



import lmdb
import pickle
import os
from functools import lru_cache
from collections.abc import Sequence

class LMDBDataset(Sequence):
    def __init__(self, db_path, start_idx=0, end_idx=None):
        assert os.path.isfile(db_path), f"{db_path} not found"
        self.db_path = db_path
        self.env = lmdb.open(
            db_path, subdir=False, readonly=True,
            lock=False, readahead=False, meminit=False, max_readers=256
        )

        #  keys
        with self.env.begin() as txn:
            cursor = txn.cursor()
            all_keys = [key for key, _ in cursor]
        total_len = len(all_keys)

        if end_idx is None or end_idx > total_len:
            end_idx = total_len

        self._keys = all_keys[start_idx:end_idx]
        self.txn = None

    def __len__(self):
        return len(self._keys)

    @lru_cache(maxsize=128)  # 
    def _get_by_index(self, idx):
        if self.txn is None:
            self.txn = self.env.begin(buffers=True)
        key = self._keys[idx]
        data = self.txn.get(key)
        return pickle.loads(data) if data is not None else None

    def __getitem__(self, idx):
        return self._get_by_index(idx)










class LMDBDataset_old:
    def __init__(self, db_path):
        self.db_path = db_path
        assert os.path.isfile(self.db_path), "{} not found".format(self.db_path)
        env = self.connect_db(self.db_path)
        with env.begin() as txn:
            self._keys = list(txn.cursor().iternext(values=False))

    def connect_db(self, lmdb_path, save_to_self=False):
        env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
        )
        if not save_to_self:
            return env
        else:
            self.env = env

    def __len__(self):
        return len(self._keys)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        if not hasattr(self, "env"):
            self.connect_db(self.db_path, save_to_self=True)
        datapoint_pickled = self.env.begin().get(f"{idx}".encode("ascii"))
        data = pickle.loads(datapoint_pickled)
        return data
