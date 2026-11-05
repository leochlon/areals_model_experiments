"""Index the prepared Objaverse-XL target packs (aggregate) and the WindsorML patch packs (windsor)."""
import argparse
from collections import Counter
import json

from prepare_corpus import ROOT, write, consolidate
from prepare_windsor import consolidate as consolidate_windsor


def aggregate():
    consolidate()
    index = json.loads((ROOT / 'prepared/index.json').read_text())
    assert index['complete']
    rows = json.loads((ROOT / 'protocol/objects.json').read_text())
    assert Counter(r['split'] for r in rows) == dict(train=61822, val=1844, test=1825)
    actual = {r['id']: r for r in index['objects']}
    assert set(actual) == {r['id'] for r in rows}
    assert all(actual[r['id']]['split'] == r['split'] for r in rows)
    write(ROOT / 'results/corpus_survival.json', dict(
        requested=dict(train=61822, val=1844, test=1825), survival=index['survival'],
        details='Every attempted ID and exclusion reason is retained in prepared/index.json.'))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['aggregate', 'windsor'])
    if p.parse_args().command == 'aggregate':
        aggregate()
    else:
        consolidate_windsor()
