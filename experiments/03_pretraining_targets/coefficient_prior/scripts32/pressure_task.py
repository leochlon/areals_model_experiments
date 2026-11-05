import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'AREALS_shared_query'
BENCH = Path('/workspace/runs/areals_all_simulation_efficiency_20261028/datasets/windsor')
sys.path.insert(0, str(PACKAGE))
import shared_query
from shared_query.pressure import load_selection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['tune', 'score'])
    parser.add_argument('--mode', choices=['operator'], required=True)
    parser.add_argument('--task', type=int, required=True, help='budget-seed cell, 0-14')
    args = parser.parse_args()
    task = args.task
    if not 0 <= task < 15:
        raise ValueError(task)
    budget, seed = (18, 35, 69, 137, 274)[task // 3], task % 3
    out = ROOT / f'pressure/{args.mode}/n{budget}/seed{seed}'
    selection = out / 'selection.json'
    if args.command == 'tune':
        if selection.exists():
            load_selection(selection)
            print('Already completed and verified:', selection)
            return
        command = ['tune', '--checkpoint', ROOT / 'pretraining/operator32/native_final.pt',
                   '--windsor-index', ROOT / 'windsor/index.json', '--field-dir', BENCH / 'data/fields',
                   '--geometry-dir', BENCH / 'cache/geometry', '--subsets', ROOT / 'windsor_protocol/subsets.json',
                   '--budget', budget, '--seed', seed, '--out', out, '--device', 'cuda', '--resume']
    else:
        if (out / 'test_metrics.json').exists():
            print('Preserving existing test result:', out)
            return
        command = ['score', '--selection', selection, '--device', 'cuda']
    subprocess.run([sys.executable, str(PACKAGE / 'pressure.py')] + list(map(str, command)), check=True, cwd=PACKAGE)


if __name__ == '__main__':
    main()
