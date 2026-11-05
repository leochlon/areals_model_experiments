import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def main():
    rows = []
    for mode in ('operator',):
        for budget in (18, 35, 69, 137, 274):
            seeds = []
            for seed in range(3):
                path = ROOT / f'pressure/{mode}/n{budget}/seed{seed}/test_metrics.json'
                result = json.loads(path.read_text())
                seeds.append(dict(seed=seed, pretrained=result['metrics']['pretrained']['mean_nrmse'],
                                  scratch=result['metrics']['scratch']['mean_nrmse'], source=str(path)))
            pretrained = statistics.mean(s['pretrained'] for s in seeds)
            scratch = statistics.mean(s['scratch'] for s in seeds)
            rows.append(dict(mode=mode, training_cars=budget, pretrained_mean_nrmse=pretrained,
                             scratch_mean_nrmse=scratch, relative_error_change=pretrained / scratch - 1,
                             seeds=seeds))
    report = dict(complete=True, metric='Mean per-car area-weighted pressure RMSE / reference standard deviation',
                  test_cars=35, seeds=[0, 1, 2], geometry_cohort=json.loads((ROOT / 'protocol/cohort.json').read_text())['counts'],
                  tuning='Independent three-rate search for every objective, initialization, car budget and seed.',
                  results=rows)
    (ROOT / 'results/operator32_comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
