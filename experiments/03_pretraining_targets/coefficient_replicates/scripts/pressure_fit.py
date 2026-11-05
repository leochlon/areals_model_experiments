import datetime
import json
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np
import torch
import torch.distributed as dist

from common import ROOT, p, read_protocol, study

OUT = None
LR = None
INITIALIZATION = None
WORLD = 4


def gather(value):
    values = [None] * WORLD
    dist.all_gather_object(values, value)
    return values


def evaluate(model, cars, ids, protocol, device, rank):
    result = p.evaluate(model, cars, ids[rank::WORLD], protocol['normalization'], device,
                        protocol['eval_chunk'])
    rows = [row for part in gather(result['cars']) for row in part]
    rows.sort(key=lambda row: row['id'])
    assert [row['id'] for row in rows] == sorted(ids)
    return dict(mean_nrmse=float(np.mean([row['nrmse'] for row in rows])), cars=rows)


def update(model, optimizer, cars, protocol, device, rank, step):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    chosen = np.random.default_rng([protocol['seed'], step, 811]).choice(
        protocol['training_ids'], protocol['batch'], replace=True)
    losses = torch.zeros(protocol['batch'], dtype=torch.float64, device=device)
    stats = protocol['normalization']
    for slot in range(rank, protocol['batch'], WORLD):
        car = cars.get(str(chosen[slot])); field = car['field']
        selected = np.random.default_rng([protocol['seed'], step, slot, 912]).choice(
            len(field['values']), protocol['queries'], replace=True, p=field['weights'])
        query = torch.as_tensor(field['points'][selected], device=device, dtype=torch.float32)[None]
        target = torch.as_tensor((field['values'][selected] - stats['mean']) / stats['std'],
                                 device=device, dtype=torch.float32)
        prediction = model.predict(p.encode(model, car['geometry'], device), query)[0]
        loss = (prediction - target).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite pressure loss at {step}, rank {rank}')
        (loss / protocol['batch']).backward()
        losses[slot] = loss.detach()
    # Each rank contributes two cars divided by the global batch of eight.
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    flat = torch.cat([gradient.reshape(-1) for gradient in gradients])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    offset = 0
    for gradient in gradients:
        gradient.copy_(flat[offset:offset + gradient.numel()].view_as(gradient))
        offset += gradient.numel()
    dist.all_reduce(losses, op=dist.ReduceOp.SUM)
    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    return dict(step=step, loss=float(losses.mean()), gradient_norm=float(grad), unix=time.time())


def qualify(model, optimizer, protocol, native_state, cars, device, rank):
    reference_path = OUT / 'qualification/serial'
    if rank == 0 and not (reference_path / 'last.pt').exists():
        p.train_candidate(protocol, INITIALIZATION, LR, reference_path, cars, native_state, stop_after=1)
    dist.barrier()
    initial = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    result = update(model, optimizer, cars, protocol, device, rank, 1)
    reference = torch.load(reference_path / 'last.pt', map_location='cpu', weights_only=False)
    serial_log = json.loads((reference_path / 'training.jsonl').read_text().splitlines()[0])
    actual = optimizer.state_dict()['state']
    expected = reference['optimizer']['state']
    assert actual.keys() == expected.keys()
    delta_sq = sum(float((actual[k]['exp_avg'].cpu().double() - expected[k]['exp_avg'].double()).square().sum())
                   for k in actual)
    reference_sq = sum(float(expected[k]['exp_avg'].double().square().sum()) for k in expected)
    gradient_relative_l2 = (delta_sq / reference_sq) ** 0.5
    current = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    state_delta_sq = sum(float((current[k].double() - reference['model'][k].double()).square().sum()) for k in current)
    update_sq = sum(float((reference['model'][k].double() - initial[k].double()).square().sum()) for k in current)
    update_relative_l2 = (state_delta_sq / update_sq) ** 0.5
    report = dict(rank=rank, loss_difference=abs(result['loss'] - serial_log['loss']),
                  gradient_relative_l2=gradient_relative_l2, update_relative_l2=update_relative_l2,
                  maximum_parameter_difference=max(float((current[k] - reference['model'][k]).abs().max()) for k in current))
    report['passed'] = (report['loss_difference'] < 1e-6 and gradient_relative_l2 < 1e-4
                        and update_relative_l2 < 0.005)
    reports = gather(report)
    passed = all(row['passed'] for row in reports)
    if rank == 0:
        p.write_json(OUT / 'qualification/result.json', dict(passed=passed, ranks=reports,
                     reference='pressure.train_candidate, first real eight-car update'))
        print('QUALIFICATION', json.dumps(reports), flush=True)
    if not passed:
        raise RuntimeError('Four-GPU update did not qualify against the original serial runner')


def main(rows, task):
    """Fit rate candidate `task` (cell task // 3, rate task % 3) of `rows` on four GPUs (torchrun)."""
    global OUT, LR, INITIALIZATION
    settings = study()
    rates = settings['pressure']['rates']
    cell = rows[task // len(rates)]
    rate_index = task % len(rates)
    LR = rates[rate_index]
    OUT = ROOT / cell['out'] / f'lr{rate_index}'
    INITIALIZATION = cell['initialization']
    rank = int(os.environ['RANK']); local_rank = int(os.environ['LOCAL_RANK'])
    assert int(os.environ['WORLD_SIZE']) == WORLD
    assert torch.cuda.device_count() == WORLD
    device = torch.device('cuda', local_rank)
    torch.cuda.set_device(device)
    p.deterministic(device)
    dist.init_process_group('nccl', device_id=device, timeout=datetime.timedelta(minutes=10))
    protocol = read_protocol(ROOT / cell['out'])
    protocol.update(distributed=dict(world_size=WORLD, local_cars=2,
                    gradient_reduction='sum after loss/global_batch'))
    assert protocol['batch'] == 8 and protocol['steps'] == 8000 and protocol['new_lr'] == 2e-4
    assert protocol['native_config']['model']['dropout'] == 0
    checkpoint_path = ROOT / cell['checkpoint']
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    assert checkpoint['step'] == settings['pretraining']['steps']
    inputs = protocol['inputs']
    cars = p.Cars(inputs['windsor_index'], inputs['field_dir'], inputs['geometry_dir'])
    devices = gather(dict(rank=rank, host=socket.gethostname(),
                          uuid=str(torch.cuda.get_device_properties(device).uuid)))
    assert len({row['uuid'] for row in devices}) == WORLD
    assert len({row['host'] for row in devices}) == 1

    def initialize():
        model = p.make_model(protocol['native_config'], INITIALIZATION, protocol['seed'], checkpoint['model']).to(device)
        optimizer = torch.optim.AdamW([dict(params=model.base.parameters(), lr=LR),
                                      dict(params=model.head.parameters(), lr=protocol['new_lr'])], weight_decay=0.)
        return model, optimizer

    model, optimizer = initialize()
    config = dict(protocol, initialization=INITIALIZATION, encoder_lr=LR)
    if rank == 0:
        OUT.mkdir(parents=True, exist_ok=True)
        if (OUT / 'config.json').exists():
            assert p.read_json(OUT / 'config.json') == config
        p.write_json(OUT / 'config.json', config)
        p.write_json(OUT / 'allocation.json', dict(devices=devices,
                     global_batch=8, local_batch=2, encoder_decoder_lr=LR, head_lr=protocol['new_lr']))
        print('START', json.dumps(dict(devices=devices, encoder_decoder_lr=LR, head_lr=protocol['new_lr'])), flush=True)
    dist.barrier()
    start, best = 0, float('inf')
    if (OUT / 'last.pt').exists():
        assert p.read_json(OUT / 'qualification/result.json')['passed']
        state = torch.load(OUT / 'last.pt', map_location='cpu', weights_only=False)
        assert state['config'] == config
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['rank_rng'][rank]['cpu'])
        torch.cuda.set_rng_state(state['rank_rng'][rank]['cuda'], device)
        start, best = state['step'], state['best_validation']
        if rank == 0:
            for name in ('training.jsonl', 'validation.jsonl'):
                path = OUT / name
                kept = [line for line in path.read_text().splitlines() if json.loads(line)['step'] <= start]
                path.write_text('\n'.join(kept) + '\n')
    else:
        if rank == 0:
            for name in ('training.jsonl', 'validation.jsonl'):
                (OUT / name).write_text('')
        qualify(model, optimizer, protocol, checkpoint['model'], cars, device, rank)
        del model, optimizer
        model, optimizer = initialize()
    dist.barrier()
    began = time.time()
    for step in range(start + 1, protocol['steps'] + 1):
        result = update(model, optimizer, cars, protocol, device, rank, step)
        if rank == 0:
            with (OUT / 'training.jsonl').open('a') as log:
                log.write(json.dumps(result) + '\n')
            if step % 50 == 0:
                print('TRAIN', json.dumps(result), flush=True)
        if step % protocol['validate_every'] != 0 and step != protocol['steps']:
            continue
        report = evaluate(model, cars, protocol['validation_ids'], protocol, device, rank)
        report['step'] = step
        improved = report['mean_nrmse'] < best
        best = min(best, report['mean_nrmse'])
        rng = gather(dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(device)))
        if rank == 0:
            with (OUT / 'validation.jsonl').open('a') as log:
                log.write(json.dumps(report) + '\n')
            state = dict(version=p.VERSION, config=config, model=model.state_dict(), optimizer=optimizer.state_dict(),
                         step=step, best_validation=best, rank_rng=rng)
            if improved:
                p.write_json(OUT / 'best_validation.json', report)
                p.save_state(OUT / 'best.pt', state)
            p.save_state(OUT / 'last.pt', state)
            print('VALIDATION', step, report['mean_nrmse'], 'elapsed', time.time() - began, flush=True)
        dist.barrier()

    if rank == 0:
        p.write_json(OUT / 'complete.json', dict(step=protocol['steps'], validation=best))
    dist.barrier()
    dist.destroy_process_group()

