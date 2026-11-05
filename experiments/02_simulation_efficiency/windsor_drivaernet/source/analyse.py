import csv
import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(os.environ['STUDY_ROOT'])
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


def threshold_text(row):
    p, s = row['pretrained_minimum_tested_budget'], row['scratch_minimum_tested_budget']
    limit = row['censoring_limit']
    if p is None and s is None:
        return f'Neither arm reached the target within {limit:,} training simulations.'
    if p is None:
        return f'Scratch reached the target with {s:,}; pretrained did not reach it within {limit:,}.'
    if s is None:
        return f'Pretrained reached the target with {p:,}; scratch did not reach it within {limit:,} (right-censored).'
    saving = 1 - p / s
    if saving > 0:
        return (f'Pretrained: {p:,}; scratch: {s:,} training simulations. Saving: {saving:.1%} '
                f'({row["reduction_including_validation"]:.1%} including validation).')
    if saving == 0:
        if p == row['smallest_tested_budget']:
            return f'Both arms already reached the target at the smallest tested budget ({p:,}); savings below this grid remain unresolved.'
        return f'Both arms first reached the target at {p:,} training simulations; no saving on the tested grid.'
    return f'Pretrained required {p:,}, against {s:,} for scratch; pretraining used more simulations on the tested grid.'


def main():
    OUT.mkdir(exist_ok=True)
    protocol, jobs = read(ROOT / 'protocol.json'), read(ROOT / 'matrix.json')
    records = read(ROOT / 'data/manifest.json')['records']
    test = sorted((r for r in records if r['split'] == 'test'), key=lambda r: r['id'])
    case_index = {r['id']: i for i, r in enumerate(test)}
    arms, budgets, seeds, sensors = protocol['arms'], protocol['budgets'], protocol['seeds'], [0, 20]
    errors = np.full((2, len(budgets), len(seeds), len(test), 2), np.nan)
    rows, completions, histories = [], [], {}
    for job in jobs:
        folder = ROOT / 'runs' / job['name']
        result, complete = read(folder / 'test.json'), read(folder / 'complete.json')
        assert complete['actual_simulations_exposed'] == job['budget']
        assert len(result['rows']) == 2 * len(test)
        completions.append(complete)
        histories[job['name']] = read(folder / 'history.json')
        a, n, s = arms.index(job['arm']), budgets.index(job['budget']), seeds.index(job['seed'])
        seen = set()
        for row in result['rows']:
            key = (row['case'], row['sensors'])
            assert key not in seen
            seen.add(key)
            errors[a, n, s, case_index[row['case']], sensors.index(row['sensors'])] = row['nrmse']
            rows.append(dict(arm=job['arm'], budget=job['budget'], seed=job['seed'], **row))
    assert np.isfinite(errors).all()
    with (OUT / 'per_case.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    group_names = sorted({r['family'] for r in test})
    groups = [[i for i, r in enumerate(test) if r['family'] == g] for g in group_names]
    group_counts = np.array([len(g) for g in groups])
    sums = np.stack([errors[:, :, :, ids, :].sum(axis=3) for ids in groups], axis=3)
    rng = np.random.default_rng(284901)
    boot = np.empty((5000, 2, len(budgets), 2))
    for b in range(len(boot)):
        ss = rng.integers(len(seeds), size=len(seeds))
        gg = rng.integers(len(groups), size=len(groups))
        boot[b] = np.take(np.take(sums, ss, axis=2), gg, axis=3).sum(axis=3).mean(axis=2) / group_counts[gg].sum()
    means = errors.mean(axis=(2, 3))
    summary, differences = [], []
    for a, arm in enumerate(arms):
        for n, budget in enumerate(budgets):
            for k, sensor in enumerate(sensors):
                per_seed = errors[a, n, :, :, k].mean(axis=1)
                ci = interval(boot[:, a, n, k])
                summary.append(dict(arm=arm, training_simulations=budget,
                    including_validation=budget+protocol['split']['val'], sensors=sensor,
                    mean_nrmse=float(means[a, n, k]), ci95_low=ci[0], ci95_high=ci[1],
                    seed_sd=float(per_seed.std(ddof=1)), seed_0=float(per_seed[0]), seed_1=float(per_seed[1]), seed_2=float(per_seed[2])))
    for n, budget in enumerate(budgets):
        for k, sensor in enumerate(sensors):
            differences.append(dict(budget=budget, sensors=sensor,
                pretrained_minus_scratch=float(means[0, n, k]-means[1, n, k]),
                ci95=interval(boot[:, 0, n, k]-boot[:, 1, n, k])))
    with (OUT / 'learning_curves.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    thresholds = []
    for k, sensor in enumerate(sensors):
        for target in [.20, .25, .30]:
            point = [crossing(means[a, :, k], budgets, target) for a in range(2)]
            draws = np.array([[crossing(b[a, :, k], budgets, target) for a in range(2)] for b in boot])
            both = np.isfinite(draws).all(axis=1)
            np_, ns = point
            row = dict(sensors=sensor, target=target,
                smallest_tested_budget=min(budgets),
                pretrained_minimum_tested_budget=int(np_) if math.isfinite(np_) else None,
                scratch_minimum_tested_budget=int(ns) if math.isfinite(ns) else None,
                pretrained_crossing_at_lower_grid_boundary=bool(np_==min(budgets)),
                scratch_crossing_at_lower_grid_boundary=bool(ns==min(budgets)),
                censoring_limit=max(budgets), pretrained_grid_interval95=interval(draws[:, 0]),
                scratch_grid_interval95=interval(draws[:, 1]), bootstrap_both_cross_fraction=float(both.mean()),
                bootstrap_pretrained_needs_fewer_fraction=float((draws[:, 0] < draws[:, 1]).mean()),
                bootstrap_both_censored_fraction=float((~np.isfinite(draws).any(axis=1)).mean()))
            if all(math.isfinite(v) for v in point):
                validation = protocol['split']['val']
                row.update(training_simulations_saved=int(ns-np_), training_reduction=float(1-np_/ns),
                    reduction_including_validation=float(1-(np_+validation)/(ns+validation)), training_ratio=float(ns/np_))
            if both.any():
                row['training_ratio_interval95_conditional_on_both_crossing'] = interval(draws[both, 1]/draws[both, 0])
            row['interpretation'] = threshold_text(row)
            thresholds.append(row)
    endpoints = [r for r in thresholds if r['target'] == .25]
    limited = {str(k): sum(c['budget_limited'][str(k)] for c in completions) for k in sensors}
    result = dict(dataset=protocol['dataset'], completed_runs=len(jobs), split=protocol['split'],
        physical_families=protocol['families'], endpoints=endpoints, summary=summary,
        training_geometry_counts=[dict(budget=c['budget'], seed=c['seed'],
            physical_geometries=c['unique_training_geometries'], exposures_per_simulation=c['per_simulation_exposures'])
            for c in completions if c['arm']=='pretrained'],
        paired_differences=differences, thresholds=thresholds, optimisation_budget_limited_runs=limited,
        summed_fit_wall_gpu_hours=sum(c['wall_seconds'] for c in completions)/3600,
        total_field_updates=sum(c['updates'] for c in completions),
        uncertainty='Paired bootstrap over three downstream seeds and physical test geometry groups; 5,000 draws; pointwise 95% intervals.',
        checkpoint_selection='Separate validation-selected checkpoint for each sensor endpoint',
        geometry_pretraining_cost_excluded_from_fit_gpu_hours=True)
    write(OUT / 'summary.json', result)
    write(OUT / 'thresholds.json', thresholds)
    np.savez_compressed(OUT / 'bootstrap.npz', means=boot, budgets=budgets, sensors=sensors)
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False, 'pdf.fonttype':42})
    colours = ['#1765a7', '#c35d24']
    fig, axes = plt.subplots(1, 2, figsize=(10.6,4.5), sharey=True)
    for k, ax in enumerate(axes):
        for a, arm in enumerate(arms):
            ci = np.quantile(boot[:, a, :, k], [.025,.975], axis=0)
            ax.plot(budgets, means[a,:,k], 'o-', color=colours[a], label='Geometry pretrained' if a==0 else 'From scratch')
            ax.fill_between(budgets, ci[0], ci[1], color=colours[a], alpha=.15)
        ax.axhline(.25, color='#666666', lw=1, ls='--', label='Fixed target: 0.25')
        ax.set_xscale('log',base=2); ax.set_xticks(budgets, [f'{n:,}' for n in budgets])
        ax.set_xlabel('Labelled training simulations'); ax.set_title(f'{sensors[k]} sensors'); ax.grid(alpha=.18)
    axes[0].set_ylabel('Mean area-weighted pressure nRMSE'); axes[1].legend(frameon=False,fontsize=9)
    fig.suptitle(protocol['dataset'] + ': simulation efficiency')
    fig.text(.5,-.035,f'Three paired seeds; {len(test):,} test simulations in {len(groups):,} physical geometry groups. Pointwise 95% bootstrap bands.\n'
        f'8,000 updates per fit, batch 8. Add {protocol["split"]["val"]:,} validation simulations to every training budget.',ha='center',fontsize=9)
    fig.tight_layout()
    for extension in ['png','pdf','svg']:
        fig.savefig(OUT/f'simulation_efficiency.{extension}',dpi=180,bbox_inches='tight')
    convergence, panels = plt.subplots(2,len(budgets),figsize=(15,5.8),sharex=True)
    for k, sensor in enumerate(sensors):
        column = 'val_nrmse' if sensor==0 else 'val_nrmse20'
        for n, budget in enumerate(budgets):
            ax = panels[k,n]
            for a, arm in enumerate(arms):
                histories_ = [histories[f'{arm}_n{budget}_seed{s}'] for s in seeds]
                x = [h['update'] for h in histories_[0]]
                y = np.array([[h[column] for h in history] for history in histories_])
                ax.plot(x,y.mean(0),color=colours[a]); ax.fill_between(x,y.min(0),y.max(0),color=colours[a],alpha=.15)
            ax.axhline(.25,color='#666666',ls='--',lw=.7); ax.set_title(f'N={budget:,}; {sensor} sensors',fontsize=9)
            ax.grid(alpha=.18); ax.set_xlabel('Updates'); ax.set_ylabel('Validation nRMSE')
    convergence.tight_layout()
    convergence.savefig(OUT/'validation_curves.pdf',bbox_inches='tight')
    convergence.savefig(OUT/'validation_curves.png',bbox_inches='tight',dpi=150)
    plt.close('all')
    print(json.dumps(dict(dataset=protocol['dataset'],endpoints=endpoints),indent=2),flush=True)


if __name__ == '__main__':
    main()
