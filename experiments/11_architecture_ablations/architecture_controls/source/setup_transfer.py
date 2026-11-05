import sys
from pathlib import Path

ROOT = Path('/workspace/runs/areals_architecture_controls_20261024')
sys.path.insert(0, '/workspace/runs/field_benchmark_4gpu_20261023')
import field_benchmark as fb


if __name__ == '__main__':
    source = fb.bench_config(Path('/workspace/runs/field_benchmark_4gpu_20261023/bench'))
    fb.main(['observations', '--manifest', source['manifest'], '--fields', source['fields'],
             '--bench', str(ROOT / 'architecture_transfer/bench'), '--split', 'val',
             '--patch-counts', '64', '--hidden', '.6', '--sensors', '5', '--unobserved'])
