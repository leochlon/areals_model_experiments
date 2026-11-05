"""Numerical check of the runtime settings of fast_runtime.py on one training car: FP32 query chunks of 256 and
2,048 give the same predictions and gradients."""
import argparse
import contextlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from setup_confirmation import ROOT,OLD,fb,write
sys.path.insert(0,str(OLD/'source'))
import transfer_architecture as transfer
from areal_tokens.data import to_device


def main():
    torch.set_num_threads(2);torch.backends.cuda.enable_cudnn_sdp(False)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',default=str(ROOT/'bench/models/isotropic_reference_seed2/field_best.pt'),
                        help='a field checkpoint of the reference arm, seed 2')
    source=Path(parser.parse_args().checkpoint)
    transfer.ARM,transfer.SEED='reference',2
    model,experiment,state=transfer.field_load(source,'cuda');model.eval()
    config=fb.bench_config(ROOT/'bench')
    record=next(r for r in json.loads((ROOT/'car_manifest.json').read_text())['records'] if r['split']=='train')
    field=fb.load_field(config['fields'],record['id'])
    prepared=fb.prepare_car(record,field.points,experiment)
    sample=fb.encoder_inputs(prepared,len(field.points),np.empty(0,np.int64),np.empty(0))
    batch=to_device(fb.collate_inputs([sample]),'cuda')
    indices=np.linspace(0,len(field.points)-1,16384).astype(int)
    queries=torch.tensor(field.points[indices][None],device='cuda',dtype=torch.float32)
    with torch.no_grad():
        geometry=model.encode(batch,batch['values'],batch['present'],batch['anchor_values'],batch['anchor_present'])
        outputs={}
        timings={}
        for amp in [False,True]:
            for chunk in [256,2048]:
                model.base.decoder.config=replace(model.base.decoder.config,query_chunk=chunk)
                def run():
                    with torch.autocast('cuda',dtype=torch.bfloat16) if amp else contextlib.nullcontext():
                        return model.predict(geometry,queries)
                run();torch.cuda.synchronize();start=time.perf_counter()
                result=run();torch.cuda.synchronize()
                timings[f'{amp}:{chunk}']=time.perf_counter()-start
                outputs[amp,chunk]=result.float().cpu().numpy()
        fp32=float(np.abs(outputs[False,256]-outputs[False,2048]).max())
        bf16=float(np.sqrt(np.mean((outputs[True,256]-outputs[True,2048])**2)))
        print('CHUNKING_NUMERICS',json.dumps(dict(fp32_max_abs=fp32,bf16_rms=bf16,timings=timings)),flush=True)
        assert fp32<3e-5
    parameters=[p for p in model.parameters() if p.requires_grad]
    gradients=[]
    q=queries[:,:2048];truth=torch.tensor(field.values[indices[:2048]][None],device='cuda',dtype=torch.float32)
    meta=state['benchmark'];truth=(truth-meta['mean'])/meta['std']
    for chunk in [256,2048]:
        model.base.decoder.config=replace(model.base.decoder.config,query_chunk=chunk)
        geometry=model.encode(batch,batch['values'],batch['present'],batch['anchor_values'],batch['anchor_present'])
        loss=(model.predict(geometry,q)-truth).square().mean()
        grads=torch.autograd.grad(loss,parameters,allow_unused=True)
        gradients.append(torch.cat([g.float().flatten() for g in grads if g is not None]).detach())
    a,b=gradients
    cosine=float(torch.dot(a,b)/(a.norm()*b.norm()))
    relative=float((a-b).norm()/a.norm())
    assert cosine>.999 and relative<.02
    result=dict(checkpoint=str(source),probe_car=record['id'],
        fp32_query_chunk_max_abs=fp32,bf16_query_chunk_rms=bf16,
        gradient_cosine=cosine,gradient_relative_difference=relative,forward_seconds=timings,
        bf16_larger_chunks_rejected=bool(bf16>=.002))
    write(ROOT/'results/runtime_check.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
