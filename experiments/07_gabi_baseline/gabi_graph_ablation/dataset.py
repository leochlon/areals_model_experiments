import sys

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from common import ROOT, ORIGINAL, UPSTREAM, records

sys.path.insert(0, str(UPSTREAM))
from datatools import HelmholtzCarDataset

class GraphDataset(Dataset):
    def __init__(self, split, variant):
        self.rows = records(split)
        self.original = HelmholtzCarDataset(str(ORIGINAL / 'bench/gabi' / (split + '.pkl')))
        assert len(self.rows) == len(self.original)
        self.graphs = {}
        for row in self.rows:
            grid = row['grid_id']
            if grid not in self.graphs:
                with np.load(ROOT / 'graphs' / (grid + '.npz')) as cache:
                    self.graphs[grid] = torch.as_tensor(cache[variant].copy())
                    points = cache['points'].copy()
                assert np.array_equal(self.original.get(self.rows.index(row)).pos.numpy(), points)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        data = self.original.get(index)
        data.edge_index = self.graphs[self.rows[index]['grid_id']]
        return data

    def get_data_dims_dict(self):
        return self.original.get_data_dims_dict()

class EpochSampler(Sampler):
    def __init__(self, count, seed):
        self.count, self.seed, self.epoch = count, seed, 0

    def order(self):
        generator = torch.Generator().manual_seed(2026102800 + self.seed * 10000 + self.epoch)
        return torch.randperm(self.count, generator=generator)

    def __iter__(self):
        return iter(self.order().tolist())

    def __len__(self):
        return self.count
