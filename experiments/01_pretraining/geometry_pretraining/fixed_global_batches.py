"""Shard each fixed global batch across ranks without padding or dropping shapes."""

import math

import torch
from torch.utils.data import Sampler


class FixedGlobalBatchSampler(Sampler):
    def __init__(self, size, global_batch, world, rank, seed, epoch):
        if not 0 <= rank < world or global_batch < world or size < world:
            raise ValueError("Each rank needs at least one example")
        if size % global_batch and size % global_batch < world:
            raise ValueError("The final global batch must contain at least one example per rank")
        self.size, self.global_batch, self.world, self.rank = size, global_batch, world, rank
        self.order = torch.randperm(size, generator=torch.Generator().manual_seed(seed + epoch)).tolist()

    def __len__(self):
        return math.ceil(self.size / self.global_batch)

    def __iter__(self):
        for start in range(0, self.size, self.global_batch):
            yield self.order[start:start + self.global_batch][self.rank::self.world]

    def global_size(self, index):
        return min(self.global_batch, self.size - index * self.global_batch)
