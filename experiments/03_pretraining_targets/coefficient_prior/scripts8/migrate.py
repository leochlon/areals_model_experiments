import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts32'))
from migrate import equal, save


def main():
    # The run keeps the directory name operator32 at eight GPUs.
    out = ROOT / 'pretraining/operator32'
    receipt_path = ROOT / 'checks/migration8.json'
    if receipt_path.exists():
        state = torch.load(out / 'last.pt', map_location='cpu', weights_only=False)
        assert state['recipe']['world_size'] == 8 and len(state['rng']) == 8
        print('Eight-GPU migration already complete; resuming.', flush=True)
        return
    archive = out / 'before_resize8'
    archive.mkdir(exist_ok=False)
    for name in ('last.pt', 'best.pt', 'native_best.pt', 'run.json', 'train.jsonl', 'validation.jsonl'):
        if (out / name).exists():
            shutil.copyfile(out / name, archive / name)
    state = torch.load(archive / 'last.pt', map_location='cpu', weights_only=False)
    assert state['step'] >= 4000
    assert state['recipe']['world_size'] == 32 and len(state['rng']) == 32
    assert state['recipe']['batch_size'] == 64 and state['recipe']['steps'] == 10626
    assert state['recipe']['config']['model']['dropout'] == 0
    converted = dict(state)
    converted['recipe'] = dict(state['recipe'], world_size=8)
    converted['rng'] = [copy.deepcopy(state['rng'][rank * 4]) for rank in range(8)]
    temporary = out / 'resume8.pt.tmp'
    torch.save(converted, temporary)
    saved = torch.load(temporary, map_location='cpu', weights_only=False)
    equal(state['model'], saved['model'])
    equal(state['optimizer'], saved['optimizer'])
    assert saved['step'] == state['step'] and saved['best_validation'] == state['best_validation']
    assert dict(saved['recipe'], world_size=32) == state['recipe']
    temporary.replace(out / 'last.pt')
    receipt = dict(created_at=datetime.now(timezone.utc).isoformat(), source=str(archive / 'last.pt'),
                   resume_step=state['step'], old_world_size=32, new_world_size=8,
                   global_batch=64, local_batch=8, steps=10626,
                   rng_mapping='New rank r inherits old rank 4*r.')
    save(receipt_path, receipt)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
