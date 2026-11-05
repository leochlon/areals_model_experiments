import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import torch

ROOT = Path(__file__).resolve().parents[1]


def save(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)


def equal(a, b):
    if torch.is_tensor(a):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


def main():
    source = ROOT / 'pretraining/operator'
    out = ROOT / 'pretraining/operator32'
    out.mkdir(exist_ok=False)
    archive = out / 'source_4gpu.pt'
    shutil.copyfile(source / 'last.pt', archive)
    state = torch.load(archive, map_location='cpu', weights_only=False)
    assert state['recipe']['world_size'] == 4 and len(state['rng']) == 4
    assert state['recipe']['batch_size'] == 64 and state['recipe']['steps'] == 10626
    assert state['recipe']['mode'] == 'operator'
    assert state['recipe']['config']['model']['dropout'] == 0
    assert state['recipe']['config']['model']['hint_noise'] == 0
    converted = dict(state)
    converted['recipe'] = dict(state['recipe'], world_size=32)
    converted['rng'] = [copy.deepcopy(state['rng'][rank // 8]) for rank in range(32)]
    torch.save(converted, out / 'last.pt')
    saved = torch.load(out / 'last.pt', map_location='cpu', weights_only=False)
    equal(state['model'], saved['model'])
    equal(state['optimizer'], saved['optimizer'])
    assert saved['step'] == state['step']
    assert saved['best_validation'] == state['best_validation']
    assert dict(saved['recipe'], world_size=4) == state['recipe']
    for name in ('train.jsonl', 'validation.jsonl'):
        lines = [line for line in (source / name).read_text().splitlines()
                 if line.strip() and json.loads(line)['step'] <= state['step']]
        (out / name).write_text('\n'.join(lines) + '\n')
    save(out / 'run.json', saved['recipe'])
    receipt = dict(created_at=datetime.now(timezone.utc).isoformat(), source=str(archive),
                   resume_step=state['step'], old_world_size=4, new_world_size=32,
                   global_batch=64, local_batch=2, steps=10626,
                   rng_mapping='new rank r inherits old rank floor(r / 8)')
    save(ROOT / 'checks/migration32.json', receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
