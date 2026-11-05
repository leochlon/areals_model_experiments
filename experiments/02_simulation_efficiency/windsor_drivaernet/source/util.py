import hashlib
import json
import os
from pathlib import Path

ROOT = Path(os.environ['SUITE_ROOT'])


def stable(value):
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], 'little')


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)
