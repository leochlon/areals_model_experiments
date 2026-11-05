"""Pretrain one row of protocol/pretraining.json on four GPUs. The paper uses rows 0 and 4 (d, nn^T only, seeds 17, 29).

    python scripts/pretrain_task.py --task ROW
"""
import argparse
import subprocess
import sys

from common import ROOT, pretrain_rows, read, study


def main(task):
    row = pretrain_rows()[task]
    out = ROOT / row['out']
    if (out / 'complete.json').exists():
        assert read(out / 'complete.json')['step'] == study()['pretraining']['steps']
        return
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=4',
               str(ROOT / 'AREALS_shared_query/pretrain.py'), '--objects', str(ROOT / 'protocol/objects.json'),
               '--index', str(ROOT / 'prepared/index.json'), '--config', str(ROOT / 'AREALS_shared_query/native_config.json'),
               '--mode', row['mode'], '--out', str(out), '--seed', str(row['seed']), '--steps', '10626',
               '--batch-size', '64', '--workers', '4', '--validation-objects', '128', '--validate-every', '500',
               '--device', 'cuda', '--amp', 'fp32', '--aux-weight', str(row['weight'] if row['weight'] else .1)]
    if (out / 'last.pt').exists():
        command.append('--resume')
    subprocess.run(command, check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=int, required=True)
    main(parser.parse_args().task)
