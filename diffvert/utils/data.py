# data file, currently compatible with ATLAS samples

import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured as s2u
from typing import NamedTuple

import multiprocessing as mp
import jax.numpy as jnp

import time

from torch.utils.data import Sampler

class Sample(NamedTuple):
    x: np.ndarray
    y: np.ndarray

class RandomBatchSampler(Sampler):
    def __init__(
        self,
        dataset_len: int,
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
    ):
        self.batch_size = batch_size
        self.dataset_length = dataset_len
        self.n_batches = self.dataset_length / self.batch_size
        self.nonzero_last_batch = int(self.n_batches) < self.n_batches
        self.drop_last = drop_last
        self.shuffle = shuffle

    def __len__(self):
        return int(self.n_batches) + int(not self.drop_last and self.nonzero_last_batch)

    def __iter__(self):
        if self.shuffle:
            self.batch_ids = torch.randperm(int(self.n_batches))
        else:
            self.batch_ids = torch.arange(int(self.n_batches))
        # yield full batches from the dataset
        for batch_id in self.batch_ids:
            start, stop = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
            yield np.s_[int(start) : int(stop)]

        # in case the batch size is not a perfect multiple of the number of samples,
        # yield the remaining samples
        if not self.drop_last and self.nonzero_last_batch:
            start, stop = int(self.n_batches) * self.batch_size, self.dataset_length
            yield np.s_[int(start) : int(stop)]


class HDF5Dataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.file = None  # delay opening

    def _ensure_file_open(self):
        if self.file is None:
            self.file = h5py.File(self.file_path, 'r', locking=False)  # safer for parallel reads

    def __len__(self):
        self._ensure_file_open()
        return self.file["x"].shape[0]

    def __getitem__(self, object_idx):
        self._ensure_file_open()
        x = self.file['x'][object_idx]
        y = self.file['y'][object_idx]
        return Sample(x=np.array(x), y=np.array(y[..., :2]))

    def close(self):
        if self.file is not None:
            self.file.close()

    def __getstate__(self):
        # Remove the h5py file before pickling
        state = self.__dict__.copy()
        state['file'] = None
        return state
    
    def __setstate__(self, state):
        # Restore the state after unpickling
        self.__dict__.update(state)
        self.file = None  # ensure it's reopened when accessed



from torch.utils.data import Subset

def no_collate(batch):
    return batch[0]  # batch is a list of one item (Sample)

def load_data(file_path, batch_size=64, shuffle=True, num_workers=0, drop_last=False):
    dataset = HDF5Dataset(file_path)
    
    kwargs = {
        'dataset': dataset,
        'batch_size': None,
        #'shuffle': shuffle, # the shuffling is taken care of by RandomBatchSampler
        'sampler': RandomBatchSampler(len(dataset), batch_size, shuffle, drop_last),
        'num_workers': num_workers,
        'collate_fn': None,
        'pin_memory': True
    }

    if num_workers > 0:
        ctx = mp.get_context("forkserver")
        kwargs['multiprocessing_context'] = ctx

    loader = DataLoader(**kwargs)

    '''
    # Use a subset of indices for debugging (ensure it's a list of ints) # works for FTAG data, test for Zenodo first
    start = 8164800+470
    end = start+9
    selected_indices = list(range(start, end))
    subset = Subset(dataset, selected_indices)
                    
    loader = DataLoader(subset, batch_size=batch_size, collate_fn=no_collate)
    '''

    return loader


