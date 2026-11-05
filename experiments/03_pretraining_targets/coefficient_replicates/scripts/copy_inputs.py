"""Copy the cohort and WindsorML inputs from the coefficient-prior study; write protocol/pretraining.json."""
import collections
import json
import shutil
from pathlib import Path

from common import ROOT, study, write

SOURCE = Path('/workspace/runs/shared_query_controlled_20261031')


def main():
    for folder in ('prepared', 'windsor', 'windsor_protocol', 'protocol', 'results'):
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    for name in ('protocol/objects.json', 'prepared/index.json', 'protocol/cohort.json', 'windsor/index.json',
                 'windsor_protocol/subsets.json'):
        shutil.copyfile(SOURCE / name, ROOT / name)
    objects = json.loads((ROOT / 'protocol/objects.json').read_text())
    cohort = dict(collections.Counter(row['split'] for row in objects))
    assert cohort == dict(train=14304, val=637, test=632)
    rows = []
    for seed in study()['pretraining']['seeds']:
        for weight in (0., .1, .01, .001):
            arm = 'areals' if weight == 0 else 'hessian_w' + str(weight).replace('.', 'p')
            name = f'{arm}/seed{seed}'
            rows.append(dict(name=name, out='pretraining/' + name, mode='baseline' if weight == 0 else 'hessian',
                             weight=weight, seed=seed))
    write(ROOT / 'protocol/pretraining.json', rows)
    print(json.dumps(dict(pretraining_rows=len(rows), cohort=cohort)), flush=True)


if __name__ == '__main__':
    main()
