#!/usr/bin/env python3
"""Score field models on the known-answer benchmark against the exact Bayes posterior mean.

For every observation set of a split, the posterior mean of the Whittle-Matern prior that generated the fields
(a mixture over its length scales, weighted by their marginal likelihoods) is computed exactly in the eigenbasis
written by make_grf_fields.py. Each model is then scored on the hidden points by

  efficiency = (R_prior - R_model) / (R_prior - R_Bayes)

the share of the attainable gain, from the prior mean (zero) to the exact posterior mean, that the model achieves:
1 matches the posterior reference, 0 is no better than ignoring the observations. Also reported: the Bayes gap
E||f_model - E[f|y]||^2 / E||f - E[f|y]||^2 and the raw risks. Intervals resample shapes.

  python grf_eval.py --bench grf/bench --bases grf/bases --split val \
      --methods areal_pre areal_scratch areal_pre_frozen areal_scratch_frozen
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from field_benchmark import bench_config, load_field, read_observations, records_of
from surface_fields import SurfaceGRF


def load_basis(folder: Path, shape: str, sigma: float) -> SurfaceGRF:
    with np.load(folder / f"{shape}.npz") as b:
        return SurfaceGRF(b["lam"], b["phi"].astype(np.float64), b["mass"], ells=tuple(b["ells"]), nu=float(b["nu"]),
                          sigma=sigma, scale=float(b["scale"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bench", required=True)
    parser.add_argument("--bases", required=True)
    parser.add_argument("--split", default="val", choices=("val", "test"))
    parser.add_argument("--methods", nargs="+", required=True, help="prediction folders under BENCH/predictions")
    parser.add_argument("--sigma", type=float, default=1e-3,
                        help="observation noise of the posterior; the benchmark observes exact values")
    parser.add_argument("--pairs", nargs="*", default=None,
                        help="paired differences to report, as A:B (default: the first method against each other)")
    parser.add_argument("--boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    bench = Path(args.bench)
    config = bench_config(bench)
    shape_of = {r["id"]: r["family"] for r in records_of(config)}
    meta, arrays = read_observations(bench, args.split)
    predictions = {}
    for name in args.methods:
        with np.load(bench / "predictions" / name / f"{args.split}.npz") as packed:
            predictions[name] = {key: packed[key].astype(np.float64) for key in packed.files}
    for name, preds in predictions.items():
        info = json.loads((bench / 'predictions' / name / f'{args.split}.json').read_text())
        patch_only = info['method'].get('kind', '').startswith('zero-shot ')
        allowed = {e['key'] for e in meta['entries']
                   if not patch_only or e['setting'].startswith('patch-')}
        assert set(preds) == allowed, (name, 'prediction coverage mismatch', len(preds), len(allowed))
    bases, rows = {}, []
    for number, entry in enumerate(meta["entries"], 1):
        if entry["setting"] == "unobserved":
            continue
        shape = shape_of[entry["car"]]
        if shape not in bases:
            bases[shape] = load_basis(Path(args.bases), shape, args.sigma)
        grf = bases[shape]
        field = load_field(config["fields"], entry["car"])
        observed = arrays[entry["key"]]
        posterior, _ = grf.posterior_mean(observed, field.values[observed])
        hidden = np.ones(len(field.values), bool)
        hidden[observed] = False
        w = field.weights[hidden] / field.weights[hidden].sum()
        truth = field.values[hidden]
        base = dict(shape=shape, case=entry["car"], setting=entry["setting"], prior=float(w @ truth ** 2),
                    bayes=float(w @ (posterior[hidden] - truth) ** 2))
        for name, preds in predictions.items():
            if entry["key"] not in preds:
                continue
            p = preds[entry["key"]][hidden]
            rows.append(dict(base, method=name, risk=float(w @ (p - truth) ** 2),
                             excess=float(w @ (p - posterior[hidden]) ** 2), pp=float(w @ p ** 2),
                             pt=float(w @ (p * truth))))
        if number % 200 == 0:
            print(f"{number}/{len(meta['entries'])} observation sets", flush=True)

    rng = np.random.default_rng(args.seed)
    shapes = sorted({r["shape"] for r in rows})
    index = {s: i for i, s in enumerate(shapes)}
    settings = [s["name"] for s in meta["settings"] if s["name"] != "unobserved"]
    table, lines = {}, []
    lines.append(f"## Known-answer inversion, {args.split} split, {len(shapes)} shapes\n")
    lines.append("Efficiency = (R_prior - R_model) / (R_prior - R_Bayes) on hidden points (1 = the regularised posterior "
                 "mean); gap = E||model - posterior||^2 / E||truth - posterior||^2. Brackets: 95% intervals over shapes.\n")
    lines.append(f"Reference uses sigma={args.sigma:g} as a numerical observation nugget on exact generated values.\n")
    lines.append("| setting | Bayes risk | " + " | ".join(f"{m} efficiency" for m in args.methods) + " |")
    lines.append("|---|---|" + "---|" * len(args.methods))
    draws = rng.multinomial(len(shapes), np.full(len(shapes), 1 / len(shapes)), size=args.boot)
    sums = {}
    for r in rows:
        key = (r["method"], r["setting"])
        if key not in sums:
            sums[key] = np.zeros((6, len(shapes)))
        sums[key][:, index[r["shape"]]] += (r["prior"], r["bayes"], r["risk"], r["excess"], r["pp"], r["pt"])

    def efficiency(s, weights=None):
        t = s.sum(1) if weights is None else weights @ s.T          # [6] or [boot, 6]
        return (t[..., 0] - t[..., 2]) / (t[..., 0] - t[..., 1])

    def gained(s, weights=None):
        """Efficiency at the single least-squares gain c on the predictions, refitted in every resample."""
        t = s.sum(1) if weights is None else weights @ s.T
        risk = t[..., 0] - t[..., 5] ** 2 / np.maximum(t[..., 4], 1e-300)
        return (t[..., 0] - risk) / (t[..., 0] - t[..., 1]), t[..., 5] / np.maximum(t[..., 4], 1e-300)

    for setting in settings:
        cells, bayes_cell = [], ""
        for name in args.methods:
            if (name, setting) not in sums:            # the method skips this setting (e.g. zero-shot: patches only)
                cells.append("-")
                continue
            s = sums[(name, setting)]
            prior, bayes, risk, excess = s.sum(1)[:4]
            lo, hi = np.percentile(efficiency(s, draws), [2.5, 97.5])
            table[(name, setting)] = dict(efficiency=efficiency(s), low=lo, high=hi, gap=excess / bayes,
                                          risk=risk / len(shapes), bayes=bayes / len(shapes))
            cells.append(f"{efficiency(s):.3f} [{lo:.3f}, {hi:.3f}]")
            bayes_cell = f"{bayes / max(prior, 1e-12):.3f} of prior"
        lines.append(f"| {setting} | {bayes_cell} | " + " | ".join(cells) + " |")
    lines.append("\nDiagnostic: efficiency with each method's predictions multiplied by the one least-squares gain c "
                 "(fitted on these cases, so slightly optimistic; for zero-mean known-answer fields only):\n")
    lines.append("| setting | " + " | ".join(f"{m}: c, efficiency" for m in args.methods) + " |")
    lines.append("|---|" + "---|" * len(args.methods))
    for setting in settings:
        cells = []
        for name in args.methods:
            if (name, setting) not in sums:
                cells.append("-")
                continue
            s = sums[(name, setting)]
            (point, c), (boot, _) = gained(s), gained(s, draws)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            table[(name, setting)].update(gain=c, gained_efficiency=point, gained_low=lo, gained_high=hi)
            cells.append(f"{c:.2f}, {point:.3f} [{lo:.3f}, {hi:.3f}]")
        lines.append(f"| {setting} | " + " | ".join(cells) + " |")
    pairs = ([tuple(p.split(":")) for p in args.pairs] if args.pairs is not None
             else [(args.methods[0], other) for other in args.methods[1:]])
    if pairs:
        lines.append("\nPaired differences in efficiency (A minus B, same problems), 95% intervals over shapes:\n")
        for first, other in pairs:
            for setting in settings:
                if (first, setting) not in sums or (other, setting) not in sums:
                    continue
                a, b = sums[(first, setting)], sums[(other, setting)]
                diff = efficiency(a, draws) - efficiency(b, draws)
                lo, hi = np.percentile(diff, [2.5, 97.5])
                lines.append(f"- {first} - {other}, {setting}: {efficiency(a) - efficiency(b):+.3f} "
                             f"[{lo:+.3f}, {hi:+.3f}]")
    text = "\n".join(lines)
    out = bench / "scores" / f"grf_{args.split}"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".md").write_text(text)
    (out.parent / f"grf_{args.split}_rows.json").write_text(json.dumps(rows))
    out.with_suffix(".json").write_text(json.dumps({f"{k[0]}|{k[1]}": v for k, v in table.items()}, indent=1))
    (out.parent / f"grf_{args.split}_reference.json").write_text(json.dumps(dict(
        sigma=args.sigma, reference="finite-basis posterior mean with the specified numerical observation nugget",
        observed_values="exact generated field values", shapes=len(shapes), observation_sets=len(meta["entries"])
    ), indent=2))
    print(text)


if __name__ == "__main__":
    main()
