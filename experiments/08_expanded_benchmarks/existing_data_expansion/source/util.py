import hashlib
import json
import os
from pathlib import Path

ROOT = Path(os.environ['STUDY_ROOT'])
BASE = Path(os.environ['BASE_CHECKPOINT'])
DATASETS = ['doubledelta', 'blendednet', 'drivaerml', 'drivaernet']


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def stable(text):
    return int.from_bytes(hashlib.sha256(str(text).encode()).digest()[:8], 'little')


