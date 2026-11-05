import argparse
import sys
from pathlib import Path

ORIGINAL = Path('/workspace/runs/areals_multicondition_wings_20261023')
sys.path.insert(0, str(ORIGINAL))
import study


if __name__ == '__main__':
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--role', required=True, choices=['geometry', 'measurements', 'conditioned'])
    args, remaining = p.parse_known_args()
    study.ROOT = Path('/workspace/runs/areals_architecture_controls_20261024/unseen_conditions')
    study.configure(args.role)
    study.fb.main(remaining)
