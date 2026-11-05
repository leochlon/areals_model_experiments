import json
import shutil
from pathlib import Path

import torch

ROOT=Path('/workspace/runs/areals_mask_pressure_transfer_20261025')
EXP=Path('/workspace/runs/areals_existing_data_expansion_20261025')
MVM=Path('/workspace/runs/areals_production_mechanism_20261025')

models={}
for seed in range(3):
    for label,arm in [('masked','masked_native'),('unmasked','unmasked_native')]:
        path=MVM/'pretraining'/f'{arm}_seed{17+seed}'/'epoch_11.pt'
        state=torch.load(path,map_location='cpu',weights_only=True)
        assert state['epoch']==11 and state['global_step']==10626
        assert state['arm']==arm and state['seed']==17+seed
        models[f'{label}_seed{seed}']=dict(path=str(path),geometry_seed=17+seed,updates=state['global_step'],
            geometry_manifest=state['data']['manifest'],config=state['config'])
for seed in range(3):
    a,b=[models[f'{label}_seed{seed}'] for label in ['masked','unmasked']]
    assert a['config']['model']==b['config']['model']
    assert a['geometry_manifest']==b['geometry_manifest']
datasets={}
for name in ['doubledelta','drivaerml']:
    source=EXP/'datasets'/name
    bench=ROOT/'datasets'/name/'bench'
    bench.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source/'bench/config.json',bench/'config.json')
    shutil.copytree(source/'bench/observations',bench/'observations',dirs_exist_ok=True)
    checks=json.loads((source/'data_checks.json').read_text())
    assert checks['passed']
    config=json.loads((bench/'config.json').read_text())
    assert config['manifest']==str(source/'manifest.json') and config['fields']==str(source/'fields')
    datasets[name]=dict(manifest=str(source/'manifest.json'),official_test_counts=checks['counts'])
(ROOT/'inputs.json').write_text(json.dumps(dict(models=models,datasets=datasets),indent=2)+'\n')
print(json.dumps({k:v['path'] for k,v in models.items()},indent=2))
