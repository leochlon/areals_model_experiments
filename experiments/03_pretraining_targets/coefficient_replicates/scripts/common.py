import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'AREALS_shared_query'))
import shared_query  # noqa: F401  (puts the package's vendored areal_tokens on the path)
from shared_query import pressure as p


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    p.write_json(path, value)


def study():
    return read(ROOT / 'protocol/study.json')


def read_protocol(out):
    return read(Path(out) / 'protocol.json')


def pretrain_rows():
    return read(ROOT / 'protocol/pretraining.json')
