import csv
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    result = json.loads((ROOT / 'results/complete/summary.json').read_text())
    protocol = json.loads((ROOT / 'protocol.json').read_text())
    catalog = json.loads((ROOT / 'models.json').read_text())
    csv_path = ROOT / 'results/complete/per_problem.csv'
    rows = list(csv.DictReader(csv_path.open()))
    buckets = defaultdict(list)
    for row in rows:
        buckets[row['family'], row['group'], row['setting'], row['metric'], int(row['seed']), row['car']].append(float(row['value']))
    assert len({r['car'] for r in rows}) == 48
    means, per_seed, max_difference = {}, {}, 0.0
    for family, groups in result['summaries'].items():
        setting = 'all-25-original-settings' if family == 'replication' else 'fixed-2048-query' if family == 'frozen' else 'unobserved'
        metric = 'rel_l2' if family == 'frozen' else 'nrmse'
        for group, summary in groups.items():
            seed_means = []
            for seed in range(3):
                car_scores = []
                for car in protocol['test_ids']:
                    if family == 'replication':
                        selected = [v for key, values in buckets.items() if key[0] == family and key[1] == group and key[2] != 'unobserved' and key[3] == metric and key[4] == seed and key[5] == car for v in values]
                        assert len(selected) == 25
                    else:
                        selected = buckets[family, group, setting, metric, seed, car]
                        assert len(selected) == 1
                    car_scores.append(math.fsum(selected) / len(selected))
                seed_means.append(math.fsum(car_scores) / 48)
            key = family + '/' + group
            means[key] = math.fsum(seed_means) / 3
            per_seed[key] = seed_means
            delta = abs(means[key] - summary[setting][metric]['mean'])
            assert delta < 1e-12, (key, delta)
            max_difference = max(max_difference, delta)
    comparisons = []
    for comparison in result['comparisons']:
        family, first, second = comparison['family'], comparison['first'], comparison['second']
        a, b = per_seed[family + '/' + first], per_seed[family + '/' + second]
        differences = [x - y for x, y in zip(a, b)]
        for calculated, saved in zip(differences, comparison['per_seed_differences']):
            assert abs(calculated - saved) < 1e-12
        comparisons.append(dict(family=family, first=first, second=second, per_seed_differences=differences,
                                mean_difference=means[family + '/' + first] - means[family + '/' + second],
                                car_bootstrap_95_from_saved_summary=comparison['first_minus_second']))
    log_counts = []
    for model in catalog:
        done = json.loads((ROOT / 'results/tasks' / f"{model['task']:02d}.json").read_text())
        assert done['passed']
        if model['family'] == 'frozen':
            info_path = ROOT / 'predictions/frozen' / model['name'] / 'test.json'
            info = json.loads(info_path.read_text())
            passes, failures = len(info['pressure_perturbation_checks']), 0
            assert info['passed'] and passes == 4
        else:
            text = (ROOT / 'logs' / f"evaluate_{model['task']:02d}.log").read_text()
            passes = sum(line.startswith('PASS ') for line in text.splitlines())
            failures = sum(line.startswith('FAIL ') for line in text.splitlines())
            assert passes == (16 if model['family'] == 'replication' else 4) and failures == 0
        log_counts.append(dict(name=model['name'], family=model['family'], passed=passes, failed=failures))
    # Means are recomputed from the CSV with the standard library; bootstrap intervals are copied from summary.json.
    audit = dict(passed=True, test_cars=48, completed_models=len(catalog), means_from_csv=means, per_seed_means_from_csv=per_seed,
                 maximum_mean_disagreement=max_difference, comparisons=comparisons,
                 hidden_pressure_checks_passed=sum(r['passed'] for r in log_counts),
                 hidden_pressure_checks_failed=sum(r['failed'] for r in log_counts), check_counts=log_counts)
    path = ROOT / 'results/complete/raw_csv_and_log_audit.json'
    path.write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps({k: v for k, v in audit.items() if k not in ['check_counts', 'per_seed_means_from_csv', 'comparisons']}, indent=2))


if __name__ == '__main__':
    main()
