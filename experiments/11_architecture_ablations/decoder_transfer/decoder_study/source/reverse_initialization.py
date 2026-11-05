import json
import sys
import time

import torch

from setup_confirmation import ROOT,OLD,write
sys.path.insert(0,str(OLD/'source'))
import architecture
from areal_tokens.data import GeometryDataset,collate_clouds,read_manifest
from areal_tokens.losses import reconstruction_loss


def shared_state(model):
    return {name:value.detach().clone() for name,value in model.state_dict().items() if not name.startswith('normal_head.')}


def unchanged(model,reference):
    return all(torch.equal(value,reference[name]) for name,value in shared_state(model).items())


def main():
    started=time.monotonic();torch.set_num_threads(2)
    model,config,_=architecture.build('reference',0);model.train()
    records,_=read_manifest(OLD/'architecture_manifest.json');record=next(r for r in records if r['split']=='train')
    data=GeometryDataset([record],config.model,config.data,0,training=True)
    batch=collate_clouds([data[0]])
    original={name:value.detach().clone() for name,value in model.normal_head.state_dict().items()}
    reference=shared_state(model);parameters=list(model.encoder.parameters())+list(model.normal_head.parameters())
    rows=[];vectors=[];distances=[]
    for phase in ['original','isotropic','restored_original']:
        with torch.no_grad():
            if phase=='isotropic':
                model.normal_head.weight.zero_();model.normal_head.bias.zero_();model.normal_head.bias[:3].fill_(1/3)
            else:model.normal_head.load_state_dict(original)
        prediction=model(batch);loss=reconstruction_loss(prediction,batch,config.train)
        gradients=torch.autograd.grad(loss['normal'],parameters,allow_unused=True)
        encoder=torch.cat([g.flatten() for g in gradients[:-2] if g is not None])
        head=torch.cat([g.flatten() for g in gradients[-2:] if g is not None])
        assert torch.isfinite(encoder).all() and torch.isfinite(head).all()
        vectors.append(encoder);distances.append(prediction['normalized_udf'].detach())
        rows.append(dict(phase=phase,encoder_normal_gradient_norm=float(encoder.norm()),
            normal_head_gradient_norm=float(head.norm()),normal_loss=float(loss['normal'].detach()),
            shared_parameters_unchanged=unchanged(model,reference)))
        assert rows[-1]['shared_parameters_unchanged']
    assert rows[0]['encoder_normal_gradient_norm']>0 and rows[1]['encoder_normal_gradient_norm']==0
    assert rows[1]['normal_head_gradient_norm']>0
    torch.testing.assert_close(vectors[0],vectors[2],rtol=1e-6,atol=1e-7)
    for value in distances[1:]:torch.testing.assert_close(distances[0],value,rtol=0,atol=0)
    result=dict(passed=True,sequence=rows,same_model_and_same_batch=True,
        restored_gradient_max_difference=float((vectors[0]-vectors[2]).abs().max()),
        initial_distance_predictions_exactly_identical=True,seconds=time.monotonic()-started)
    write(ROOT/'results/initialization_reversal.json',result)
    print('INITIALIZATION_REVERSAL_PASS',json.dumps(result),flush=True)


if __name__=='__main__':main()
