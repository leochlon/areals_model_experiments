import argparse
import json
import math

import torch

import field_study as study
import field_benchmark as fb
from util import stable, write

NativeCompletion, native_records, native_seed = study.Completion, fb.records_of, fb.stable_seed
FULL_COUNT, SELECTED = None, None


class BudgetCompletion(NativeCompletion):
    def __len__(self):
        return FULL_COUNT

    def __getitem__(self, index):
        # A loader worker evaluates one item at a time. Repeated objects get fresh masks and queries.
        repeat, item = divmod(index, len(self.cars))
        fb.stable_seed = lambda number, identity, epoch=0: native_seed(number, identity + ':repeat:' + str(repeat), epoch)
        try:
            return super().__getitem__(item)
        finally:
            fb.stable_seed = native_seed


class ScaleAreal(study.Areal):
    def __init__(self, name, model, experiment, meta, *args, **kwargs):
        # Same 40 rounds, batches and updates as the full-data fit; repeated objects get fresh observation masks.
        meta.update(scale_training_families=400, full_data_records_per_reference_round=FULL_COUNT)
        super().__init__(name, model, experiment, meta, *args, **kwargs)


def main():
    global FULL_COUNT, SELECTED
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['drivaernet'], required=True)
    a, remainder = p.parse_known_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.enable_cudnn_sdp(False)
    study.configure(a.dataset, 'measurements')
    config = fb.bench_config(study.DATA / 'bench')
    train = native_records(config, 'train')
    FULL_COUNT = len(train)
    families = sorted({r['family'] for r in train}, key=lambda f: stable('nested-small:' + f))[:400]
    assert len(families) == 400 and FULL_COUNT >= 800
    SELECTED = {r['id'] for r in train if r['family'] in set(families)}
    assert len(SELECTED) == 400
    def records(config, split=None):
        rows = native_records(config, split)
        return [r for r in rows if r['id'] in SELECTED] if split == 'train' else rows
    fb.records_of, fb.Completion, fb.Areal = records, BudgetCompletion, ScaleAreal
    if remainder == ['check-budget']:
        for seed in range(3):
            out = study.DATA / 'models' / f'small_seed{seed}'
            history = json.loads((out / 'history.json').read_text())
            state = torch.load(out / 'field_best.pt', map_location='cpu', weights_only=True)
            assert state['benchmark']['train_cars'] == 400
            assert history[-1]['update'] == 40 * math.ceil(FULL_COUNT / 8)
            write(out / 'complete.json', dict(passed=True, unique_training_families=400,
                  full_reference_records=FULL_COUNT, updates=history[-1]['update'], seed=seed,
                  selected_epoch=state['epoch']))
        return
    fb.main(remainder)


if __name__ == '__main__':
    main()
