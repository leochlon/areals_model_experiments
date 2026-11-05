import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from registry import ROOT,write

ORIGINAL=Path('/workspace/home/areal_aerojepa_20260928')
sys.path.insert(0,str(ORIGINAL))
import run_field as author
from drivaerml import DrivAerMLDataset


def load(model):
    state=torch.load(model['checkpoint'],map_location='cpu',weights_only=False)
    config=author.OmegaConf.create(state['config'])
    assert config.comparison.seed==model['seed']
    net=author.hydra.utils.instantiate(config.model)
    if model['arm']=='areal':
        from areal_context import ArealContextEncoder
        net.trunk.context_encoder=ArealContextEncoder(config.comparison.areal_checkpoint,token_dim=128,latent_count=256)
    net.load_state_dict(state['model'],strict=True)
    return net.cuda().eval()


class Dataset(DrivAerMLDataset):
    def __init__(self,root,split,seed,normals):
        super().__init__(root,split,seed=seed);self.normals=normals
    def __getitem__(self,index):
        item=super().__getitem__(index)
        if self.normals:item['context_feat']=torch.cat([item['context_feat'],item['context_feat'][...,:3]],-1)
        return item


def call(net,sample):
    with author.get_autocast_context(torch.device('cuda'),'bf16'):
        return net(context_pos=sample['context_pos'],context_feat=sample['context_feat'],
                   gen_params=sample['gen_params'],query_pos=sample['query_pos'],query_sdf=sample['query_sdf'])


@torch.inference_mode()
def evaluate(model,parity=False):
    """Test inference; with parity=True, replay one stored validation call of the trained checkpoint instead."""
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    author.set_seed(model['seed'])
    net=load(model)
    dataset=Dataset(ORIGINAL if parity else ROOT/'frozen','val' if parity else 'test',model['seed'],model['arm']=='native_normals')
    loader=DataLoader(dataset,batch_size=1,shuffle=False,num_workers=0,collate_fn=author.superwing_collate)
    arrays,rows,checks={},[],[]
    old=None
    if parity:old=np.load(Path(model['checkpoint']).parent/'raw_validation/call021.npz')
    for number,batch in enumerate(loader):
        sample=author._slice_batch_sample(author.move_batch_to_device(batch,torch.device('cuda')),0)
        prediction=call(net,sample)
        raw=(prediction.float()*dataset.std+dataset.mean).cpu().numpy().astype(np.float64).reshape(-1)
        restored=(sample['query_target'].float()*dataset.std+dataset.mean).cpu().numpy().astype(np.float64).reshape(-1)
        identity=str(sample['case_id'])
        assert raw.shape==restored.shape==(2048,) and np.isfinite(raw).all()
        if parity:
            difference=float(np.max(np.abs(raw-old[identity+'_prediction'])))
            assert difference<1e-5,('old-checkpoint inference parity',model['name'],difference)
        else:
            with np.load(ROOT/'frozen/data'/f'{identity}.npz') as z:
                truth=z['fixed_target'].astype(np.float64).reshape(-1)
            np.testing.assert_allclose(truth,restored,rtol=1e-6,atol=5e-7)
            restored=truth
        if number<4:
            corrupt=dict(sample)
            corrupt['query_target']=torch.randn_like(sample['query_target'])*1000
            corrupt['target_surface_main_feat']=torch.randn_like(sample['target_surface_main_feat'])*1000
            change=float((call(net,corrupt)-prediction).abs().max())
            assert change==0
            checks.append(dict(id=identity,target_pressure_perturbation_maximum_change=change))
        score=float(np.sqrt(np.sum((raw-restored)**2)/np.sum(restored**2)))
        rows.append(dict(car=identity,rel_l2=score))
        arrays[identity+'_prediction']=raw;arrays[identity+'_truth']=restored
        if parity:break
    result=dict(passed=True,parity=parity,model=model['name'],arm=model['arm'],seed=model['seed'],
          cases=len(rows),mean_rel_l2=float(np.mean([r['rel_l2'] for r in rows])),rows=rows,
          pressure_perturbation_checks=checks,checkpoint=model['checkpoint'],
          train_normalization=dict(mean=dataset.mean,std=dataset.std))
    if not parity:
        assert len(rows)==48
        out=ROOT/'predictions/frozen'/model['name'];out.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(out/'test.npz',**arrays);write(out/'test.json',result)
    else:
        result['old_prediction_maximum_difference']=difference
        write(ROOT/'results/parity'/f'{model["name"]}.json',result)
    print('FROZEN_EVAL',json.dumps({k:v for k,v in result.items() if k not in ['rows','pressure_perturbation_checks']}),flush=True)
    return result


if __name__=='__main__':
    model=json.loads((ROOT/'models.json').read_text())[int(sys.argv[1])]
    evaluate(model,'--parity' in sys.argv)
