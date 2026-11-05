"""Matched, resumable full-object/query pretraining, with native checkpoint export."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import replace
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset

from areal_tokens.config import ExperimentConfig
from .data import JointDataset, collate_joint
from .model import JointGeometry, loss_for


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def atomic_torch(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temp)
    temp.replace(path)


def deterministic(enabled=True):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(enabled)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def move(value, device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {k: move(v, device) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(move(v, device) for v in value)
    return value


def rate(step, steps, peak):
    warmup = min(200, max(1, steps // 10))
    if step <= warmup:
        return peak * step / warmup
    fraction = (step - warmup) / max(1, steps - warmup)
    return peak * (.05 + .95 * .5 * (1 + math.cos(math.pi * fraction)))


def rng_state(device):
    return dict(torch=torch.get_rng_state(), numpy=np.random.get_state(), python=random.getstate(),
                cuda=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None)


def restore_rng(state, device):
    torch.set_rng_state(state['torch'])
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])
    if device.type == 'cuda':
        torch.cuda.set_rng_state(state['cuda'], device)


def native_payload(model, config, step, recipe):
    # The field adapter loads this exact base, including the pretrained decoder.
    # Auxiliary target heads never become a new pressure decoder.
    return dict(version=1, model=model.base.state_dict(), config=config.to_dict(), step=step,
                kind='native_areals_shared_query_export', pretraining_recipe=recipe,
                transfer='entire base encoder AND spatial query decoder; new scalar field head only')


class LossModule(torch.nn.Module):
    """Expose only loss and detached metrics to DDP's unused-parameter search."""
    def __init__(self, model, train_config, mode, aux_weight):
        super().__init__()
        self.model, self.train_config = model, train_config
        self.mode, self.aux_weight = mode, aux_weight

    def forward(self, batch, aux):
        return loss_for(self.model, batch, aux, self.train_config,
                        mode=self.mode, aux_weight=self.aux_weight)


def run(args):
    rank = int(os.environ.get('RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    local = int(os.environ.get('LOCAL_RANK', 0))
    if args.batch_size % world:
        raise ValueError('Global batch size must divide by WORLD_SIZE')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    if args.amp == 'bf16' and args.device != 'cuda':
        raise ValueError('BF16 option is CUDA-only; CPU verification uses FP32')
    device = torch.device('cuda', local) if args.device == 'cuda' else torch.device('cpu')
    torch.set_num_threads(args.threads)
    deterministic(not args.allow_nondeterministic)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo', timeout=timedelta(hours=2))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    config = ExperimentConfig.from_dict(json.loads(Path(args.config).read_text()))
    # All arms use exactly the same complete-input, unrotated recipe. Auxiliary
    # caches are not silently left in a different coordinate system.
    config = replace(config, data=replace(config.data, augment_rotation='none', mask_fraction=0.,
                                          full_input_probability=1., normal_sign_flip_probability=0.),
                     train=replace(config.train, seed=args.seed, amp='none'))
    path_maps = []
    for item in args.path_map:
        if '=' not in item:
            raise ValueError('--path-map must be OLD=NEW')
        old, new = item.split('=', 1)
        path_maps.append((old, new))
    common = dict(objects_path=args.objects, index_path=args.index, config=config, seed=args.seed,
                  mode=args.mode, patches=args.patches, patch_queries=args.patch_queries,
                  hessian_queries=args.hessian_queries, path_maps=path_maps)
    train = JointDataset(split='train', length=args.steps * args.batch_size, **common)
    validation = JointDataset(split='val', length=args.validation_objects, evaluation=True, **common)
    validation.length = min(args.validation_objects, len(validation.rows))
    model = JointGeometry(config.model, hessian_scales=len(train.hessian_scales)).to(device)
    if args.init_native:
        state = torch.load(args.init_native, map_location='cpu', weights_only=False)
        if state['config']['model'] != config.to_dict()['model']:
            raise ValueError('Native initialization architecture differs from config')
        model.base.load_state_dict(state['model'], strict=True)
    recipe = dict(mode=args.mode, seed=args.seed, steps=args.steps, batch_size=args.batch_size,
                  device_type=device.type, threads=args.threads, runtime=dict(torch=str(torch.__version__),
                  cuda=torch.version.cuda, numpy=np.__version__, python=sys.version.split()[0]),
                  world_size=world, patches=args.patches, patch_queries=args.patch_queries,
                  hessian_queries=args.hessian_queries, hessian_scales=list(train.hessian_scales),
                  aux_weight=args.aux_weight, peak_lr=args.lr, weight_decay=args.weight_decay,
                  gradient_clip=args.gradient_clip, amp=args.amp,
                  deterministic=not args.allow_nondeterministic, validate_every=args.validate_every,
                  validation_objects=len(validation), config=config.to_dict(),
                  train_provenance=train.provenance, val_provenance=validation.provenance,
                  init_native=str(args.init_native) if args.init_native else None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    out = Path(args.out).resolve()
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    if world > 1:
        dist.barrier()
    start, best = 0, math.inf
    if args.resume:
        state = torch.load(out / 'last.pt', map_location='cpu', weights_only=False)
        if state['recipe'] != recipe:
            raise ValueError('Resume recipe mismatch; start a separate run')
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        start, best = state['step'], state['best_validation']
        restore_rng(state['rng'][rank], device)
        if rank == 0:
            for name in ['train.jsonl', 'validation.jsonl']:
                path = out / name
                if path.exists():
                    kept = [line for line in path.read_text().splitlines() if line.strip() and json.loads(line)['step'] <= start]
                    path.write_text('\n'.join(kept) + ('\n' if kept else ''))
    elif (out / 'run.json').exists() or (out / 'last.pt').exists():
        raise FileExistsError('Existing run: use --resume or a new output directory')
    if rank == 0:
        atomic_json(out / 'run.json', recipe)
    loss_module = LossModule(model, config.train, args.mode, args.aux_weight)
    wrapped = DistributedDataParallel(loss_module, device_ids=[local] if device.type == 'cuda' else None,
                                      find_unused_parameters=True) if world > 1 else loss_module
    end = min(args.steps, args.stop_after) if args.stop_after else args.steps
    if end <= start:
        if world > 1:
            dist.destroy_process_group()
        return
    local_batch = args.batch_size // world
    # Index is global, so object/patch/query sampling is independent of workers
    # and is recovered exactly at a checkpoint boundary.
    indices = [step * args.batch_size + rank * local_batch + slot
               for step in range(start, end) for slot in range(local_batch)]
    loader = DataLoader(Subset(train, indices), batch_size=local_batch, collate_fn=collate_joint,
                        num_workers=args.workers, pin_memory=device.type == 'cuda',
                        generator=torch.Generator().manual_seed(args.seed + rank + 9000),
                        persistent_workers=args.workers > 0)
    val_loader = DataLoader(validation, batch_size=1, collate_fn=collate_joint, num_workers=0,
                            generator=torch.Generator().manual_seed(args.seed + 10000))
    begun = time.monotonic()
    log = (out / 'train.jsonl').open('a' if args.resume else 'w', buffering=1) if rank == 0 else None
    for step, (batch, aux) in enumerate(loader, start + 1):
        wrapped.train()
        batch, aux = move(batch, device), move(aux, device)
        lr = rate(step, args.steps, args.lr)
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        context = torch.autocast('cuda', dtype=torch.bfloat16) if args.amp == 'bf16' else nullcontext()
        with context:
            loss, terms = wrapped(batch, aux)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite training loss at update {step}')
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip, error_if_nonfinite=True)
        optimizer.step()
        keys = sorted(k for k in terms if k != 'loss')
        vector = torch.stack([loss.detach().double()] + [torch.as_tensor(terms[k], device=device).detach().double() for k in keys])
        if world > 1:
            maximum = terms.get('relative_A_max')
            maximum = maximum.detach().double().clone() if maximum is not None else None
            dist.all_reduce(vector)
            vector /= world
            if maximum is not None:
                dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
                vector[1 + keys.index('relative_A_max')] = maximum
            if 'invalid_charts' in keys:
                vector[1 + keys.index('invalid_charts')] *= world
        if log:
            log.write(json.dumps(dict(step=step, loss=float(vector[0]), lr=lr, gradient_norm=float(grad),
                                      **dict(zip(keys, map(float, vector[1:])))), allow_nan=False) + '\n')
        if step % args.validate_every == 0 or step == end:
            if world > 1:
                dist.barrier()
            if rank == 0:
                model.eval()
                values, metrics = [], []
                with torch.no_grad():
                    for vb, va in val_loader:
                        vl, vt = loss_for(model, move(vb, device), move(va, device), config.train,
                                          mode=args.mode, aux_weight=args.aux_weight)
                        values.append(float(vl))
                        metrics.append({k: float(v) for k, v in vt.items()})
                score = float(np.mean(values))
                if not math.isfinite(score):
                    raise FloatingPointError('Nonfinite validation loss')
                report = dict(step=step, loss=score, objects=len(values),
                              metrics={k: float((max if k == 'relative_A_max' else sum if k == 'invalid_charts' else np.mean)([m[k] for m in metrics])) for k in metrics[0]},
                              elapsed_seconds=time.monotonic() - begun)
                atomic_json(out / 'validation_latest.json', report)
                with (out / 'validation.jsonl').open('a') as f:
                    f.write(json.dumps(report, allow_nan=False) + '\n')
                improved = score < best
                best = min(best, score)
                print(json.dumps(report), flush=True)
            else:
                improved = False
            states = [None] * world
            if world > 1:
                dist.all_gather_object(states, rng_state(device))
            else:
                states[0] = rng_state(device)
            if rank == 0:
                payload = dict(version=1, kind='shared_query_training', model=model.state_dict(),
                               optimizer=optimizer.state_dict(), step=step, best_validation=best,
                               recipe=recipe, rng=states)
                atomic_torch(out / 'last.pt', payload)
                if improved:
                    atomic_torch(out / 'best.pt', payload)
                    atomic_torch(out / 'native_best.pt', native_payload(model, config, step, recipe))
                if step == args.steps:
                    atomic_torch(out / 'native_final.pt', native_payload(model, config, step, recipe))
                    atomic_json(out / 'complete.json', dict(step=step, object_presentations=step * args.batch_size,
                                                           mode=args.mode))
            if world > 1:
                dist.barrier()
    if log:
        log.close()
    if world > 1:
        dist.destroy_process_group()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--objects', required=True, help='Original native object records JSON or JSON.gz')
    p.add_argument('--index', required=True, help='Frozen matched prepared/index.json with HDF5 patch/query packs')
    p.add_argument('--config', default=str(Path(__file__).resolve().parents[1] / 'native_config.json'))
    p.add_argument('--mode', choices=['baseline', 'hessian', 'operator'], required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--init-native', help='Optional native AREALS geometry checkpoint; fresh weights by default')
    p.add_argument('--steps', type=int, default=10626)
    p.add_argument('--batch-size', type=int, default=64, help='Global object batch across ranks')
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight-decay', type=float, default=.05)
    p.add_argument('--gradient-clip', type=float, default=1.)
    p.add_argument('--aux-weight', type=float, default=.1)
    p.add_argument('--patches', type=int, default=8)
    p.add_argument('--patch-queries', type=int, default=64)
    p.add_argument('--hessian-queries', type=int, default=256)
    p.add_argument('--validation-objects', type=int, default=128)
    p.add_argument('--validate-every', type=int, default=500)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--amp', choices=['fp32', 'bf16'], default='fp32')
    p.add_argument('--path-map', action='append', default=[], metavar='OLD=NEW')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--stop-after', type=int, help='Checkpoint early without changing the planned training schedule')
    p.add_argument('--allow-nondeterministic', action='store_true')
    return p


def main():
    args = parser().parse_args()
    for k in ['steps', 'batch_size', 'patches', 'patch_queries', 'hessian_queries', 'validation_objects',
              'validate_every', 'threads']:
        if getattr(args, k) <= 0:
            raise ValueError(f'{k} must be positive')
    for k in ['lr', 'gradient_clip', 'aux_weight']:
        if not math.isfinite(getattr(args, k)) or getattr(args, k) <= 0:
            raise ValueError(f'{k} must be finite and positive')
    if args.workers < 0 or args.weight_decay < 0 or not math.isfinite(args.weight_decay):
        raise ValueError('Invalid workers or weight decay')
    if args.stop_after is not None and args.stop_after <= 0:
        raise ValueError('stop-after must be positive')
    run(args)


if __name__ == '__main__':
    main()
