from pathlib import Path
import json
import os
import sys
import torch

ROOT=Path(os.environ['TASK_ROOT'])
sys.path.insert(0,str(ROOT/'kit'))
import field_benchmark as fb


def initial(meta):
    torch.manual_seed(meta['seed'])
    base,experiment,_=fb.load_model(meta['base_checkpoint'],'cpu')
    if meta['scratch']:
        base=fb.GeometryAutoencoder(experiment.model)
    torch.manual_seed(meta['field_head_seed'])
    return fb.FieldEncoder(base)


def check(path):
    model,_,state=fb.load_field_encoder(path,'cpu')
    meta=state['benchmark']
    before=initial(meta)
    changes=[]
    for name,now in model.state_dict().items():
        if name.startswith('head.'):
            continue
        old=before.state_dict()[name]
        if name=='base.encoder.embed.0.weight':
            now,old=now[:,:-2],old[:,:-2]
        if not torch.equal(now,old):
            changes.append(name)
    if meta.get('freeze_trunk'):
        assert not changes,changes
    else:
        assert changes,'full-training trunk did not change'
    assert not torch.equal(model.head.weight,before.head.weight),'field head did not train'
    assert not torch.equal(model.base.encoder.embed[0].weight[:,-2:],before.base.encoder.embed[0].weight[:,-2:])
    assert all(torch.isfinite(p).all() for p in model.parameters())
    return dict(path=str(path),passed=True,epoch=state['epoch'],freeze_trunk=meta.get('freeze_trunk'),
                changed_trunk_tensors=changes)


if __name__=='__main__':
    torch.set_num_threads(2)
    rows=[check(Path(p)) for p in sys.argv[1:]]
    path=Path(sys.argv[1]).parent/'weight_checks.json'
    path.write_text(json.dumps(dict(passed=True,checks=rows),indent=2)+'\n')
    print('WEIGHT_CHECKS',json.dumps(rows),flush=True)
