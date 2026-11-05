import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path('/workspace/runs/areals_decoder_transfer_20261024')
OLD = Path('/workspace/runs/areals_architecture_controls_20261024')
sys.path.insert(0, str(OLD/'source'))
import architecture
from areal_tokens.data import GeometryDataset, collate_clouds, read_manifest, to_device
from areal_tokens.losses import reconstruction_loss

native_build = architecture.build


def build(arm, seed):
    model, config, shared = native_build(arm, seed)
    with torch.no_grad():
        model.normal_head.weight.zero_()
        model.normal_head.bias.zero_()
        if config.model.normal_head == 'projector':
            # I/3 is the isotropic mean of unoriented unit-normal projectors.
            model.normal_head.bias[:3].fill_(1/3)
        else:
            # A finite unit vector avoids the singular zero-vector normaliser.
            model.normal_head.bias[2]=1
    return model, config, shared


def check():
    """The I/3 head leaves every other tensor and the distance output unchanged and gives the encoder no initial
    normal-loss gradient."""
    torch.set_num_threads(2)
    records, _ = read_manifest(OLD/'architecture_manifest.json')
    records = [r for r in records if r['split']=='train'][:2]
    rows = []
    for arm in ['reference','ordinary_attention','oriented_normal_target']:
        before, config, _ = native_build(arm, 0)
        after, after_config, _ = build(arm, 0)
        for name, value in before.state_dict().items():
            if not name.startswith('normal_head.'):
                torch.testing.assert_close(value, after.state_dict()[name], rtol=0, atol=0)
        dataset = GeometryDataset(records, config.model, config.data, 0, training=True)
        batch = to_device(collate_clouds([dataset[i] for i in range(2)]), 'cuda')
        before.cuda().train(); after.cuda().train()
        with torch.no_grad(), torch.autocast('cuda',dtype=torch.bfloat16):
            original = before(batch)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            prediction = after(batch)
            loss = reconstruction_loss(prediction,batch,after_config.train)
        torch.testing.assert_close(original['normalized_udf'],prediction['normalized_udf'],rtol=0,atol=0)
        params = list(after.encoder.parameters())
        normal_grad = torch.autograd.grad(loss['normal'],params,retain_graph=True,allow_unused=True)
        assert all(g is None or torch.count_nonzero(g)==0 for g in normal_grad)
        head_grad = torch.autograd.grad(loss['normal'],list(after.normal_head.parameters()),retain_graph=True)
        assert sum(float(g.square().sum()) for g in head_grad)>0
        loss['loss'].backward()
        assert all(torch.isfinite(p.grad).all() for p in after.parameters() if p.grad is not None)
        rows.append(dict(arm=arm,unmodified_shared_parameters=True,initial_distance_output_identical=True,
            initial_normal_gradient_into_encoder=0.,normal_head_gradient_norm=sum(float(g.square().sum()) for g in head_grad)**.5,
            repaired_normal_loss=float(loss['normal'].detach())))
    (ROOT/'results').mkdir(exist_ok=True)
    (ROOT/'results/repair_check.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(rows),flush=True)


def train(arm,seed):
    architecture.ROOT=ROOT
    architecture.build=build
    architecture.main(argparse.Namespace(arm=arm,seed=seed))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--arm',choices=['reference','ordinary_attention','oriented_normal_target'])
    parser.add_argument('--seed',type=int)
    args=parser.parse_args()
    if args.check:check()
    else:train(args.arm,args.seed)
