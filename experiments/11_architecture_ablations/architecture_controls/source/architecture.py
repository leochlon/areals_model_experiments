import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path
from types import MethodType

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from areal_tokens.config import ExperimentConfig
from areal_tokens.data import GeometryDataset, collate_clouds, read_manifest, to_device
from areal_tokens.io import save_checkpoint
from areal_tokens.losses import reconstruction_loss
from areal_tokens.model import GeometryAutoencoder, LatentGeometry, WeightedAttention
from areal_tokens.train import seed_everything

ROOT = Path('/workspace/runs/areals_architecture_controls_20261024')
ARMS = ['reference', 'ordinary_attention', 'unanchored_latents', 'oriented_normal_target']


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def uniform_attention(self, query, values, query_positions, value_positions, weights):
    equal = (weights > 0).to(weights.dtype)
    return WeightedAttention.forward(self, query, values, query_positions, value_positions, equal)


def latent_encode(encoder, batch, channels=None):
    inputs = [encoder.position(batch['points']), batch['normals']]
    if channels is not None:
        inputs.append(channels)
    points = encoder.embed(torch.cat(inputs, -1))
    batch_size = len(points)
    initial = encoder.query_latents.expand(batch_size, -1, -1)
    positions = batch['anchors'].new_zeros(batch_size, initial.shape[1], 3)
    weights = batch['anchor_weights'].new_full((batch_size, initial.shape[1]), 1 / initial.shape[1])
    tokens = encoder.read(initial, points, positions, batch['points'], batch['point_weights'])
    for block in encoder.blocks:
        tokens = block(tokens, tokens, positions, positions, weights)
    return LatentGeometry(tokens, positions, torch.zeros_like(positions), weights,
                          batch['center'], batch['scale'], batch['area'], initial)


def unanchored_forward(self, batch):
    return latent_encode(self, batch)


def install(model, arm, seed):
    if arm == 'ordinary_attention':
        for module in model.modules():
            if isinstance(module, WeightedAttention):
                module.forward = MethodType(uniform_attention, module)
    if arm == 'unanchored_latents':
        generator = torch.Generator().manual_seed(1024000 + seed)
        model.encoder.query_latents = nn.Parameter(torch.randn(1, 256, 192, generator=generator) * .02)
        model.encoder.forward = MethodType(unanchored_forward, model.encoder)
    return model


def configuration(seed, oriented=False):
    source = json.loads((ROOT / 'production_config.json').read_text())
    source['model'].update(normal_head='vector' if oriented else 'projector', normal_prior='none',
                           distance_prior='none', symmetry_prior='none', surfels_per_token=1,
                           input_condition='none')
    source['data'].update(normal_sign_flip_probability=0)
    source['train'].update(seed=seed, epochs=4, batch_size=16, workers=4, warmup_steps=100,
                          normal_loss='oriented' if oriented else 'unoriented')
    return ExperimentConfig.from_dict(source)


def build(arm, seed):
    # All shared tensors start identically, even when a head has a different shape. Returns the model, its
    # configuration and the names of the common tensors outside the normal head (compared by check_arms.py).
    seed_everything(seed)
    reference = GeometryAutoencoder(configuration(seed).model)
    config = configuration(seed, arm == 'oriented_normal_target')
    seed_everything(seed)
    model = GeometryAutoencoder(config.model)
    shared = {k: v for k, v in reference.state_dict().items()
              if k in model.state_dict() and v.shape == model.state_dict()[k].shape}
    model.load_state_dict({**model.state_dict(), **shared}, strict=True)
    install(model, arm, seed)
    return model, config, sorted(name for name in shared if not name.startswith('normal_head'))


@torch.inference_mode()
def evaluate(model, records, config, device):
    model.eval()
    dataset = GeometryDataset(records, config.model, config.data, 240024, fixed_queries=True)
    rows = []
    for sample in dataset:
        batch = to_device(collate_clouds([sample]), device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            prediction = model(batch)
        target = batch['udf'].float() / batch['scale'].float()
        distance = float(((prediction['normalized_udf'].float() - target) ** 2).mean())
        valid = batch['normal_valid']
        cosine = (prediction['normals'].float() * batch['query_normals'].float()).sum(-1).abs().clamp(0, 1)
        count = int(valid.sum())
        angle = float(torch.rad2deg(torch.acos(cosine))[valid].mean()) if count else None
        comparable = torch.nn.functional.smooth_l1_loss(prediction['normalized_udf'].float(), target, beta=.05)
        if count:
            comparable = comparable + .1 * (1 - cosine)[valid].mean()
        rows.append(dict(id=sample['id'], family=sample['family'], normalized_mse=distance,
                         normal_degrees=angle, selection=float(comparable), normal_count=count))
    return dict(normalized_mse=float(np.mean([r['normalized_mse'] for r in rows])),
                normal_degrees=float(np.mean([r['normal_degrees'] for r in rows if r['normal_degrees'] is not None])),
                selection=float(np.mean([r['selection'] for r in rows])), rows=rows)


def main(args):
    torch.set_num_threads(2)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.preferred_linalg_library('cusolver')
    model, config, _ = build(args.arm, args.seed)
    out = ROOT / 'results/architecture' / args.arm / f'seed{args.seed}'
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'last.pt').exists(), 'Refusing accidental restart'
    records, _ = read_manifest(ROOT / 'architecture_manifest.json')
    train = [r for r in records if r['split'] == 'train']
    dev = [r for r in records if r['split'] == 'val']
    test = [r for r in records if r['split'] == 'test']
    assert (len(train), len(dev), len(test)) == (8192, 128, 128)
    device = torch.device('cuda')
    model.to(device)
    train_data = GeometryDataset(train, config.model, config.data, args.seed, training=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.05)
    history, step, best = [], 0, math.inf
    started = time.monotonic()
    write(out / 'initialization.json', dict(arm=args.arm, seed=args.seed,
          parameters=sum(p.numel() for p in model.parameters()), model_config=config.to_dict(),
          gpu=torch.cuda.get_device_name(), torch=str(torch.__version__)))
    for epoch in range(4):
        train_data.set_epoch(epoch)
        generator = torch.Generator().manual_seed(args.seed * 100 + epoch)
        loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=4,
                            collate_fn=collate_clouds, generator=generator, pin_memory=True)
        model.train()
        losses = []
        for batch in loader:
            step += 1
            factor = min(1.0, step / 100) if step <= 100 else .05 + .95 * .5 * (1 + math.cos(math.pi * (step - 100) / (2048 - 100)))
            for group in optimizer.param_groups:
                group['lr'] = 3e-4 * factor
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                prediction = model(batch)
                loss = reconstruction_loss(prediction, batch, config.train)['loss']
            assert torch.isfinite(loss), (args.arm, step, 'nonfinite loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
            if step % 25 == 0:
                write(out / 'progress.json', dict(epoch=epoch + 1, step=step, total_steps=2048,
                     loss=float(np.mean(losses[-25:])), seconds=time.monotonic() - started))
                print(args.arm, args.seed, epoch + 1, step, np.mean(losses[-25:]), flush=True)
        validation = evaluate(model, dev, config, device)
        state = dict(version=1, model=model.state_dict(), config=config.to_dict(), epoch=epoch + 1,
                     global_step=step, ablation=args.arm, seed=args.seed)
        save_checkpoint(out / 'last.pt', {**state, 'optimizer': optimizer.state_dict()})
        if validation['selection'] < best:
            best = validation['selection']
            save_checkpoint(out / 'best.pt', state)
        history.append(dict(epoch=epoch + 1, loss=float(np.mean(losses)), development=validation,
                            seconds=time.monotonic() - started))
        write(out / 'history.json', history)
    state = torch.load(out / 'best.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(state['model'], strict=True)
    final = evaluate(model, test, config, device)
    final.update(seed=args.seed, arm=args.arm, selected_epoch=state['epoch'], steps=step,
                 elapsed_seconds=time.monotonic() - started)
    write(out / 'test.json', final)
    print('COMPLETE', args.arm, args.seed, final['normalized_mse'], final['normal_degrees'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--arm', choices=ARMS, required=True)
    p.add_argument('--seed', type=int, choices=[0, 1, 2], required=True)
    main(p.parse_args())
