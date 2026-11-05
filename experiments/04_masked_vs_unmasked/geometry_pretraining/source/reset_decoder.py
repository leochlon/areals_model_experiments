"""Decoder-reset checkpoints: each trained native encoder with the decoder put back to its initial weights.

Two arms of 05_gaussian_fields/field_adaptation start from these.
"""
import torch

from areal_tokens.io import save_checkpoint
from study import ROOT, SEEDS


def main():
    (ROOT / 'derived').mkdir(exist_ok=True)
    for seed in SEEDS:
        for arm in ['masked_native', 'unmasked_native']:
            out = ROOT / 'pretraining' / f'{arm}_seed{seed}'
            initial = torch.load(out / 'initial.pt', map_location='cpu', weights_only=False)
            final = torch.load(out / 'last.pt', map_location='cpu', weights_only=False)
            assert final['epoch'] == 11 and final['global_step'] == 10626
            derived = {k: v for k, v in final.items() if k not in ['optimizer', 'history']}
            derived['model'] = {k: (initial['model'][k] if k.startswith('decoder.') else v)
                                for k, v in final['model'].items()}
            derived['intervention'] = dict(type='decoder_only_reset', parent=str(out / 'last.pt'),
                                           reset_source=str(out / 'initial.pt'))
            save_checkpoint(ROOT / 'derived' / f'{arm}_reset_decoder_seed{seed}.pt', derived)
    print('wrote', ROOT / 'derived', flush=True)


if __name__ == '__main__':
    main()
