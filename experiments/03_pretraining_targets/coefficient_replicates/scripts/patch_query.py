"""Fine-tuning seed 0 of the coefficient priors with pretraining seeds 29 and 43, and the three-seed report.

    python scripts/patch_query.py prepare --task CELL                                   # CELL 0-9
    torchrun --standalone --nproc-per-node=4 scripts/patch_query.py fit --seed 29 --task FIT   # FIT 0-14
    python scripts/patch_query.py report

Seed 17 is the coefficient-prior study's own seed-0 result at each budget.
"""
import argparse
import os
import statistics

from common import ROOT, read, read_protocol, study, write
import early_areals
import pipeline


def cells():
    return read(ROOT / 'protocol/patch_query_cells.json')


def followup():
    manifest = read(ROOT / 'protocol/patch_query_windsor.json')
    for key, value in manifest['pressure_settings'].items():
        assert study()['pressure'][key] == value, key
    return manifest


def task_index(seed, task):
    assert seed in (29, 43) and 0 <= task < 15
    return (0 if seed == 29 else 15) + task


def prepare(task):
    followup()
    cell = cells()[task]
    pipeline.prepare(cell, study()['pretraining']['steps'])
    protocol = read_protocol(ROOT / cell['out'])
    reference = read_protocol(ROOT / f"pressure/areals/seed17/n{cell['budget']}/seed0")
    fields = ['seed', 'budget', 'steps', 'batch', 'queries', 'new_lr', 'validate_every',
              'rates', 'training_ids', 'validation_ids', 'normalization', 'precision',
              'optimizer', 'weight_decay', 'schedule']
    checks = {key: protocol[key] == reference[key] for key in fields}
    checks['native_model'] = protocol['native_config']['model'] == reference['native_config']['model']
    checks['native_data'] = protocol['native_config']['data'] == reference['native_config']['data']
    assert all(checks.values()), checks
    write(ROOT / cell['out'] / 'comparability.json', dict(passed=True, checks=checks))


def fit(seed, task):
    import pressure_fit
    index = task_index(seed, task)
    pressure_fit.main(cells(), index)
    if int(os.environ['RANK']) == 0:
        early_areals.score_ready(cells()[index // 3])


def report():
    manifest = followup()
    scores = {}
    for cell in cells():
        candidates = pipeline.candidate_records(cell)
        winner = min(candidates, key=lambda x: (x['validation'], x['lr']))
        result = early_areals.load_score(cell, winner)
        assert result is not None
        scores[(cell['pretraining_seed'], cell['budget'])] = dict(
            pretraining_seed=cell['pretraining_seed'], fine_tuning_seed=0,
            selected_lr=winner['lr'], validation=winner['validation'],
            test=result['metrics']['mean_nrmse'], source=cell['out'] + '/early_test_metrics.json')
    rows = []
    for budget in manifest['budgets']:
        reference = manifest['reference_seed17'][str(budget)]
        test = read(reference['test'])['metrics']['pretrained']
        winner = read(reference['selection'])['winners']['pretrained']
        runs = [dict(pretraining_seed=17, fine_tuning_seed=0, selected_lr=winner['lr'],
                     validation=winner['validation'], test=test['mean_nrmse'], source=reference['test'])]
        runs += [scores[(seed, budget)] for seed in (29, 43)]
        rows.append(dict(training_cars=budget, runs=runs,
                         mean_test_nrmse=statistics.mean(x['test'] for x in runs),
                         std_test_nrmse=statistics.stdev(x['test'] for x in runs)))
    write(ROOT / 'results/patch_query_windsor.json', dict(
        complete=True, pretraining_seeds=[17, 29, 43], fine_tuning_seeds=[0],
        metric='Mean per-car area-weighted pressure RMSE / reference standard deviation',
        test_cars=35, results=rows))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'fit', 'report'])
    parser.add_argument('--task', type=int)
    parser.add_argument('--seed', type=int, choices=[29, 43])
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.task)
    elif args.command == 'fit':
        fit(args.seed, args.task)
    else:
        report()
