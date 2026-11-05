import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from areal_tokens.data import GeometryDataset, read_manifest, stable_seed, to_device
from areal_tokens.evaluate import evaluate_model
from areal_tokens.io import CHECKPOINT_VERSION, save_checkpoint
from areal_tokens.losses import reconstruction_loss
from areal_tokens.train import lr_at_step
from study import ARMS, ROOT, build, collate_fixed, dataset, write

path = Path(os.environ['PRODUCTION_ROOT']) / 'fixed_global_batches.py'
spec = importlib.util.spec_from_file_location('fixed_global_batches', path)
sampler_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sampler_module)
Sampler = sampler_module.FixedGlobalBatchSampler


def share(value, rank):
    values = [value if rank == 0 else None]
    dist.broadcast_object_list(values, src=0)
    return values[0]


def run(args, rank, device):
    records, metadata = share(read_manifest(ROOT / 'geometry_manifest.json') if rank == 0 else None, rank)
    train = [r for r in records if r['split'] == 'train']
    val = [r for r in records if r['split'] == 'val']
    assert (len(train), len(val)) == (61822, 1844)
    raw, optimizer, config = build(args.seed, args.arm, device, rank_seed_offset=rank)
    model = DistributedDataParallel(raw, device_ids=[device.index], broadcast_buffers=False)
    out = ROOT / 'pretraining' / f'{args.arm}_seed{args.seed}'
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        assert not (out / 'last.pt').exists(), f'{out} already holds a trained model'
        save_checkpoint(out / 'initial.pt', dict(version=CHECKPOINT_VERSION, model=raw.state_dict(),
            config=config.to_dict(), epoch=0, global_step=0))
        write(out / 'initialization.json', dict(config=config.to_dict(),
            parameter_tensors=len(list(raw.parameters())), parameters=sum(p.numel() for p in raw.parameters()),
            optimizer_groups=[dict(weight_decay=g['weight_decay'], parameter_tensors=len(g['params'])) for g in optimizer.param_groups],
            data=metadata, world_size=4, global_batch=64,
            seed=args.seed, arm=args.arm, planned_updates=10626))
    dist.barrier()
    training = dataset(train, config, args.seed, args.arm)
    # Selection diagnostics use the original native masked validation protocol for every arm.
    validation = GeometryDataset(val, config.model, config.data, args.seed, mask_fraction=.6, fixed_queries=True)
    step, best_normal, history, started = 0, float('inf'), [], time.monotonic()
    for epoch in range(11):
        training.set_epoch(epoch)
        batches = Sampler(len(train), 64, 4, rank, args.seed, epoch)
        assert len(batches) == 966
        loader = DataLoader(training, batch_sampler=batches, num_workers=8, collate_fn=collate_fixed,
            pin_memory=True, persistent_workers=False,
            generator=torch.Generator().manual_seed(stable_seed(args.seed, 'loader', epoch)))
        model.train()
        sums = torch.zeros(5, dtype=torch.float64, device=device)
        for index, batch in enumerate(loader):
            rate = lr_at_step(config.train, step, 966)
            for group in optimizer.param_groups:
                group['lr'] = rate
            batch = to_device(batch, device)
            n, global_n = len(batch['id']), batches.global_size(index)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                losses = reconstruction_loss(model(batch), batch, config.train)
            if not torch.isfinite(losses['loss']):
                raise FloatingPointError((args.arm, args.seed, step, 'loss'))
            (losses['loss'] * (4 * n / global_n)).backward()
            norm = torch.nn.utils.clip_grad_norm_(raw.parameters(), config.train.gradient_clip, error_if_nonfinite=True)
            optimizer.step()
            step += 1
            sums[:4] += torch.stack([losses[k].detach().double() for k in ['loss', 'distance', 'normal', 'normal_fraction']]) * n
            sums[4] += n
            if rank == 0 and (step == 1 or step % 25 == 0):
                elapsed = time.monotonic() - started
                progress = dict(arm=args.arm, seed=args.seed, phase='training', epoch=epoch + 1,
                    step=step, total_steps=10626, lr=rate, seconds=elapsed,
                    training_eta_seconds=elapsed / step * (10626 - step), rank0_loss=float(losses['loss'].detach()),
                    world_size=4, global_batch=64)
                write(out / 'progress.json', progress)
                print(json.dumps(progress), flush=True)
        dist.all_reduce(sums)
        assert int(sums[4]) == 61822
        checkpoint = dict(version=CHECKPOINT_VERSION, config=config.to_dict(), model=raw.state_dict(),
                          optimizer=optimizer.state_dict(), epoch=epoch + 1, global_step=step,
                          arm=args.arm, seed=args.seed, data=metadata)
        if rank == 0:
            save_checkpoint(out / f'epoch_{epoch + 1:02d}.pt', {k: v for k, v in checkpoint.items() if k != 'optimizer'})
            write(out / 'progress.json', dict(arm=args.arm, seed=args.seed, phase='validation',
                  epoch=epoch + 1, step=step, total_steps=10626, seconds=time.monotonic() - started))
        dist.barrier()
        report = None
        if rank == 0:
            torch.backends.cuda.preferred_linalg_library('magma')
            torch.backends.cuda.enable_cudnn_sdp(True)
            try:
                report = evaluate_model(raw, validation, config, device, workers=8)
                write(out / f'validation_{epoch + 1:02d}.json', report)
            finally:
                torch.backends.cuda.preferred_linalg_library('cusolver')
                torch.backends.cuda.enable_cudnn_sdp(False)
        report = share(report, rank)
        if rank == 0:
            normal = report['aggregate']['normal_degrees']
            if normal < best_normal:
                best_normal = normal
                save_checkpoint(out / 'best_normal.pt', checkpoint)
            history.append(dict(epoch=epoch + 1, updates=step, lr=rate,
                training=dict(zip(['loss', 'distance', 'normal', 'normal_fraction'], (sums[:4] / sums[4]).cpu().tolist(), strict=True)),
                validation=report['aggregate'], seconds=time.monotonic() - started))
            checkpoint['history'] = history
            save_checkpoint(out / 'last.pt', checkpoint)
            write(out / 'history.json', history)
        dist.barrier()
    assert step == 10626
    if rank == 0:
        write(out / 'complete.json', dict(passed=True, arm=args.arm, seed=args.seed, updates=step,
              epochs=11, schedule_horizon_epochs=15,
              final_validation=history[-1]['validation'], best_normal=best_normal,
              seconds=time.monotonic() - started))
        print('COMPLETE', args.arm, args.seed, step, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--seed', type=int, choices=[17, 18, 19], required=True)
    args = parser.parse_args()
    rank, local = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    assert int(os.environ['WORLD_SIZE']) == 4
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(local)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.preferred_linalg_library('cusolver')
    device = torch.device('cuda', local)
    dist.init_process_group('nccl', device_id=device)
    try:
        run(args, rank, device)
    finally:
        dist.destroy_process_group()
