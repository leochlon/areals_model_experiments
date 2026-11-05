import csv
import json
import math
from pathlib import Path
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(os.environ.get('STUDY_ROOT', Path(__file__).resolve().parents[1]))
OUT = ROOT / 'results'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def crossing(curve, budgets, target):
    ix = np.flatnonzero(curve <= target)
    return float(budgets[ix[0]]) if len(ix) else math.inf


def interval(values):
    return [float(v) if math.isfinite(v) else None
            for v in np.quantile(values, [.025, .975], method='nearest')]


def main():
    OUT.mkdir(exist_ok=True)
    p = read(ROOT / 'protocol.json')
    arms, budgets, seeds, sensors = p['arms'], p['budgets'], p['seeds'], [0, 20]
    jobs = read(ROOT / 'matrix.json')
    cases = sorted({r['id'] for r in read(ROOT / 'data/manifest.json')['records'] if r['split'] == 'test'})
    assert len(cases) == 50
    errors = np.full((len(arms), len(budgets), len(seeds), len(cases), len(sensors)), np.nan)
    all_rows, completions, histories = [], [], {}
    for job in jobs:
        folder = ROOT / 'runs' / job['name']
        test, complete = read(folder / 'test.json'), read(folder / 'complete.json')
        assert len(test['rows']) == 100
        completions.append(complete)
        histories[job['name']] = read(folder / 'history.json')
        a, n, s = arms.index(job['arm']), budgets.index(job['budget']), seeds.index(job['seed'])
        seen = set()
        for row in test['rows']:
            key = (row['case'], row['sensors'])
            assert key not in seen
            seen.add(key)
            errors[a, n, s, cases.index(row['case']), sensors.index(row['sensors'])] = row['nrmse']
            all_rows.append(dict(arm=job['arm'], budget=job['budget'], seed=job['seed'], **row))
    assert np.isfinite(errors).all()
    with (OUT / 'per_case.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    means = errors.mean(axis=(2, 3))
    rng = np.random.default_rng(284901)
    boot = np.empty((5000, len(arms), len(budgets), len(sensors)))
    for b in range(len(boot)):
        ss = rng.integers(len(seeds), size=len(seeds))
        cc = rng.integers(len(cases), size=len(cases))
        boot[b] = np.take(np.take(errors, ss, axis=2), cc, axis=3).mean(axis=(2, 3))
    summary = []
    for a, arm in enumerate(arms):
        for n, budget in enumerate(budgets):
            for k, sensor in enumerate(sensors):
                per_seed = errors[a, n, :, :, k].mean(axis=1)
                ci = interval(boot[:, a, n, k])
                summary.append(dict(arm=arm, training_simulations=budget, including_validation=budget+34,
                    sensors=sensor, mean_nrmse=float(means[a, n, k]), ci95_low=ci[0], ci95_high=ci[1],
                    seed_sd=float(per_seed.std(ddof=1)), seed_0=float(per_seed[0]),
                    seed_1=float(per_seed[1]), seed_2=float(per_seed[2])))
    with (OUT / 'learning_curves.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    differences = []
    for n, budget in enumerate(budgets):
        for k, sensor in enumerate(sensors):
            differences.append(dict(budget=budget, sensors=sensor,
                pretrained_minus_scratch=float(means[0, n, k] - means[1, n, k]),
                ci95=interval(boot[:, 0, n, k] - boot[:, 1, n, k])))
    thresholds = []
    for k, sensor in enumerate(sensors):
        for target in [.20, .25, .30]:
            point = [crossing(means[a, :, k], budgets, target) for a in range(2)]
            draws = np.array([[crossing(b[a, :, k], budgets, target) for a in range(2)] for b in boot])
            both = np.isfinite(draws).all(axis=1)
            npre, nscratch = point
            row = dict(sensors=sensor, target=target,
                pretrained_minimum_tested_budget=int(npre) if math.isfinite(npre) else None,
                scratch_minimum_tested_budget=int(nscratch) if math.isfinite(nscratch) else None,
                censoring_limit=max(budgets), pretrained_grid_interval95=interval(draws[:, 0]),
                scratch_grid_interval95=interval(draws[:, 1]),
                bootstrap_both_cross_fraction=float(both.mean()),
                bootstrap_pretrained_needs_fewer_fraction=float((draws[:, 0] < draws[:, 1]).mean()),
                bootstrap_both_censored_fraction=float((~np.isfinite(draws).any(axis=1)).mean()))
            if all(math.isfinite(v) for v in point):
                row.update(training_simulations_saved=int(nscratch-npre),
                           training_reduction=float(1-npre/nscratch),
                           reduction_including_validation=float(1-(npre+34)/(nscratch+34)),
                           training_ratio=float(nscratch/npre))
            if both.any():
                row['training_ratio_interval95_conditional_on_both_crossing'] = interval(draws[both, 1] / draws[both, 0])
            thresholds.append(row)
    primary = next(t for t in thresholds if t['sensors'] == 0 and t['target'] == .25)
    npre, nscratch = primary['pretrained_minimum_tested_budget'], primary['scratch_minimum_tested_budget']
    if npre is not None and nscratch is not None and npre < nscratch:
        headline = (f'Geometry pretraining reached nRMSE <= 0.25 with {npre} training simulations, '
                    f'against {nscratch} from scratch: {1-npre/nscratch:.0%} fewer training simulations '
                    f'({1-(npre+34)/(nscratch+34):.1%} fewer including validation).')
    elif npre is not None and nscratch is not None:
        headline = (f'The fixed nRMSE target of 0.25 required {npre} training simulations with geometry pretraining '
                    f'and {nscratch} from scratch on the measured grid; this experiment did not establish a saving at that target.')
    elif npre is not None:
        headline = (f'Geometry pretraining reached nRMSE <= 0.25 with {npre} training simulations; '
                    f'the scratch model did not reach it with up to 400, so the saving is right-censored.')
    else:
        headline = ('Geometry pretraining did not reach the fixed nRMSE target of 0.25 within the tested budgets; '
                    'a simulation-count saving at that target was not established.')
    limited = sum(c['budget_limited'] for c in completions)
    study = dict(headline=headline, primary=primary, summary=summary, paired_differences=differences,
        thresholds=thresholds, completed_runs=len(jobs), optimisation_budget_limited_runs=limited,
        summed_fit_wall_gpu_hours=sum(c['wall_seconds'] for c in completions)/3600,
        total_field_updates=sum(c['updates'] for c in completions),
        uncertainty='Paired hierarchical bootstrap over three seeds and 50 test cars; pointwise 95% intervals, 5,000 draws.',
        geometry_pretraining_cost_excluded_from_fit_gpu_hours=True)
    write(OUT / 'summary.json', study)
    write(OUT / 'thresholds.json', thresholds)

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.bbox': 'tight', 'pdf.fonttype': 42})
    colours = ['#1765a7', '#c35d24']
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4), sharey=True)
    for k, ax in enumerate(axes):
        for a, arm in enumerate(arms):
            ci = np.quantile(boot[:, a, :, k], [.025, .975], axis=0)
            ax.plot(budgets, means[a, :, k], 'o-', color=colours[a], label='Geometry pretrained' if a == 0 else 'From scratch')
            ax.fill_between(budgets, ci[0], ci[1], color=colours[a], alpha=.15)
        ax.axhline(.25, color='#666666', lw=1, ls='--', label='Fixed target: 0.25')
        ax.set_xscale('log', base=2); ax.set_xticks(budgets, budgets)
        ax.set_xlabel('Labelled training simulations')
        ax.set_title('No sensors (primary)' if k == 0 else '20 sensors (secondary)')
        ax.grid(alpha=.18)
    axes[0].set_ylabel('Mean area-weighted pressure nRMSE')
    axes[1].legend(frameon=False, fontsize=9)
    fig.suptitle('DrivAerML: simulation data required for field prediction', y=1.02)
    fig.text(.5, -.03, 'Three paired seeds; 50 official test cars. Bands: paired seed/car bootstrap, pointwise 95% intervals.\n'
             'Both arms: 8,000 updates, batch 8. Add 34 validation simulations to every training budget.', ha='center', fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / 'simulation_efficiency.png', dpi=180)
    fig.savefig(OUT / 'simulation_efficiency.pdf')
    fig.savefig(OUT / 'simulation_efficiency.svg')

    convergence, panels = plt.subplots(2, 3, figsize=(10.6, 6.5), sharex=True)
    for n, budget in enumerate(budgets):
        ax = panels.flat[n]
        for a, arm in enumerate(arms):
            h = [histories[f'{arm}_n{budget}_seed{s}'] for s in seeds]
            x = [r['update'] for r in h[0]]
            y = np.array([[r['val_nrmse'] for r in hh] for hh in h])
            ax.plot(x, y.mean(0), color=colours[a], label=arm)
            ax.fill_between(x, y.min(0), y.max(0), color=colours[a], alpha=.12)
        ax.axhline(.25, color='#666666', lw=.7, ls='--')
        ax.set_title(f'{budget} training simulations'); ax.grid(alpha=.18)
        ax.set_ylim(bottom=0)
        ax.set_xlabel('Optimiser updates'); ax.set_ylabel('Validation nRMSE')
    panels.flat[-1].axis('off')
    panels.flat[-1].text(0, .85, 'Fixed training recipe\n\n8,000 updates per fit\nBatch size 8\nLR 0.001, one-cycle schedule\nValidation every 500 updates\n\nLines: mean over three seeds\nShading: minimum to maximum seed\n\nThese curves assess the chosen\nrecipe, not optimal sample complexity.', va='top', fontsize=10)
    convergence.suptitle('Optimisation check: validation only')
    convergence.tight_layout()
    convergence.savefig(OUT / 'validation_curves.png', dpi=180)
    convergence.savefig(OUT / 'validation_curves.pdf')

    plt.close('all')
    print(headline, flush=True)
    print(json.dumps(primary, indent=2), flush=True)


if __name__ == '__main__':
    main()
