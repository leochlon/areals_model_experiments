import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'patch_pressure'
sys.path.insert(0, str(BUNDLE))
from train_mesh_geometry import sha256, json_write, PatchEncoder
from pressure_data import Cases, tensor_scene, normalized_queries, neighbors
from pressure_model import load_geometry_initialization
from train_patch_pressure import parser as pressure_parser, initialize, setup
from run_pressure_study import pretraining_provenance

RATES = [2e-5, 2e-4, 1e-3]
CHECKPOINT = ROOT / 'pretraining/geometry_for_pressure.pt'
MANIFEST = ROOT / 'windsor/cases.json'
OUT = ROOT / 'pressure'
OPTIONS = ['--device', 'cuda', '--steps', '8000', '--cases-per-step', '8', '--queries', '2048',
    '--validate-every', '250', '--validation-queries', '0', '--patience', '20', '--min-steps', '1000',
    '--encoder-chunk', '1024', '--no-gradient-checkpointing', '--threads', '1',
    '--weighting', 'area', '--selection-metric', 'nrmse_std']


def export():
    source = ROOT / 'pretraining/patch/final.pt'
    state = torch.load(source, map_location='cpu', weights_only=True)
    assert state['step'] == 10626 and state['training_config']['arm'] == 'patch'
    # Repackage tensors only: the supplied loader sees the exact trained encoder.
    encoder = PatchEncoder(**state['model_config'])
    encoder.load_state_dict(state['model'], strict=True)
    payload = {key: state[key] for key in ['version', 'step', 'model', 'model_config']}
    payload.update(data_sha256=sha256(ROOT / 'prepared/geometry.json'),
        source_checkpoint_sha256=sha256(source), training_config=state['training_config'])
    torch.save(payload, CHECKPOINT)
    loaded = load_geometry_initialization(CHECKPOINT)
    assert all(torch.equal(value, loaded['model'][name]) for name, value in state['model'].items())
    proof = pretraining_provenance(CHECKPOINT, ROOT / 'prepared/geometry.json')
    assert proof['training_meshes'] == 385
    expected = json.loads((ROOT / 'prepared/receipt.json').read_text())['training_source_meshes_sha256']
    assert proof['training_source_meshes_sha256'] == expected
    json_write(ROOT / 'pretraining/export_receipt.json', dict(**proof, tensor_equality_verified=True,
        source_checkpoint_sha256=sha256(source)))


def training_args(arm, grid):
    directory = OUT / arm / f'grid_{grid:03d}' / 'seed_0'
    return ['train', *OPTIONS, '--manifest', str(MANIFEST), '--geometry-checkpoint', str(CHECKPOINT),
        '--init', arm, '--seed', '0', '--lr', '0.0002', '--encoder-lr', str(RATES[grid]),
        '--out', str(directory)]


def register():
    data = Cases(MANIFEST, weighting='area')
    source = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    sample = pressure_parser().parse_args(training_args('pretrained', 0))
    settings = {k: v for k, v in vars(sample).items() if k not in
        ('func', 'out', 'resume', 'stop_after', 'init', 'seed', 'lr', 'encoder_lr')}
    protocol = dict(version='patch-pressure-tuning-v1', grid=[dict(encoder_lr=r, lr=2e-4) for r in RATES],
        seeds=[0], selection_rule='minimum mean validation selection_score across seeds; tie -> first registered grid entry',
        training_settings=settings, data_fingerprint=data.audit['fingerprint'], scope=data.scope,
        geometry_checkpoint_sha256=sha256(CHECKPOINT), geometry_checkpoint_step=source['step'],
        model_family='polynomial_patch_encoder')
    OUT.mkdir(exist_ok=True)
    path = OUT / 'protocol.json'
    if path.exists(): assert json.loads(path.read_text()) == protocol
    else: json_write(path, protocol)
    json_write(ROOT / 'checks/windsor_audit.json', data.audit)
    print(json.dumps(dict(cars={s: len(v) for s, v in data.by_split.items()}, registered_trials=6)), flush=True)


def real_check():
    # A real-car forward/backward validates the exported cache and checkpoint interface.
    args = pressure_parser().parse_args(training_args('pretrained', 0))
    data = Cases(MANIFEST, weighting='area')
    stats = data.normalization()
    model, provenance = initialize(args, stats)
    args.init = 'scratch'
    _, control = initialize(args, stats)
    assert provenance['decoder_initial_sha256'] == control['decoder_initial_sha256']
    device = torch.device('cuda')
    model.to(device).train()
    case = data.by_split['train'][0]
    entry, field = data.entries[case], data.get(case)
    geo = data.geometry(entry)
    ids = np.random.default_rng(91).choice(len(field['cp']), 2048, replace=True,
        p=field['weights'] / field['weights'].sum())
    xyz = normalized_queries(geo, field['xyz'][ids])
    ni = neighbors(geo, xyz, args.neighbors)
    predicted = model(tensor_scene(geo, device), torch.tensor(entry['conditions'], device=device),
        torch.tensor(xyz, device=device), torch.tensor(ni, device=device), standardized=True)
    target = (torch.tensor(field['cp'][ids], device=device, dtype=torch.float32) - model.cp_mean) / model.cp_std
    loss = (predicted - target).square().mean()
    loss.backward()
    gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    assert torch.isfinite(loss) and any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())
    json_write(ROOT / 'checks/real_car.json', dict(passed=True, device=str(device), car=entry['id'],
        patches=len(geo['inputs']), queries=len(ids), loss=float(loss.detach()), gradient_norm=float(gradient),
        matched_decoder_initialization=True))


def trial(index):
    arm, grid = ('pretrained' if index < 3 else 'scratch'), index % 3
    command = training_args(arm, grid)
    directory = OUT / arm / f'grid_{grid:03d}' / 'seed_0'
    if (directory / 'validation_selected.json').exists():
        print('Already complete', directory)
        return
    if (directory / 'last.pt').exists(): command.append('--resume')
    subprocess.run([sys.executable, str(BUNDLE / 'train_patch_pressure.py'), *command], check=True)


def finish():
    subprocess.run([sys.executable, str(BUNDLE / 'tune_pressure.py'), '--manifest', str(MANIFEST),
        '--geometry-checkpoint', str(CHECKPOINT), '--out', str(OUT), '--seeds', '0',
        '--encoder-lrs', *map(str, RATES), '--decoder-lrs', '0.0002', '--resume', *OPTIONS], check=True)


if __name__ == '__main__':
    os.umask(0o077)
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['export', 'register', 'real-check', 'trial', 'finish'])
    p.add_argument('--index', type=int)
    args = p.parse_args()
    if args.command == 'export': export()
    elif args.command == 'register': register()
    elif args.command == 'real-check': real_check()
    elif args.command == 'trial': trial(args.index)
    else: finish()
