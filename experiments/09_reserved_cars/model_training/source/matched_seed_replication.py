import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, '/workspace/runs/field_benchmark_4gpu_20261023')
import field_benchmark as fb


if __name__ == '__main__':
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--seed', type=int, default=0)
    args, remaining = p.parse_known_args()
    NativeField = fb.FieldEncoder

    class Field(NativeField):
        def __init__(self, base):
            super().__init__(base)
            # The native scratch path consumes another model initialisation before the readout.
            # A separate stream makes the new scalar head identical in each matched pair.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(57300 + args.seed)
                self.head.reset_parameters()

    if remaining[0] == 'train-areal':
        out = Path(remaining[remaining.index('--out') + 1])
        assert not (out / 'field_best.pt').exists(), f'{out} already holds trained weights'
        fb.FieldEncoder = Field
        remaining += ['--seed', str(args.seed)]
    fb.main(remaining)
