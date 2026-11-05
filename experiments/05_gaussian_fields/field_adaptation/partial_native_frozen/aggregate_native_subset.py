"""aggregate.py restricted to the three frozen arms. It wrote the stored frozen-trunk summary."""
from pathlib import Path

from aggregate import summarise

ROOT = Path('/workspace/runs/areals_production_mechanism_20261025/partial_native_frozen')
GROUPS = ['masked_native_frozen', 'unmasked_native_frozen', 'random_frozen']
PAIRS = [('masked_native_frozen', 'unmasked_native_frozen'), ('masked_native_frozen', 'random_frozen'),
         ('unmasked_native_frozen', 'random_frozen')]

if __name__ == '__main__':
    summarise(ROOT, GROUPS, PAIRS)
