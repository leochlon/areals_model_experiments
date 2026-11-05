#!/usr/bin/env python3
"""Equal-budget, independent validation tuning for pretrained and scratch.

Example:
  python tune_pressure.py --manifest cases.json --geometry-checkpoint geometry.pt \
    --out comparison --seeds 17 29 43 --encoder-lrs 2e-5 2e-4 1e-3 \
    --decoder-lrs 2e-4 --device cuda --steps 20000

Other train_patch_pressure.py training options are forwarded to BOTH arms.
The grid and selection rule are saved before training. Test scoring begins only
after all candidates finish and the selection receipt has been written.
"""
from __future__ import annotations
import argparse
import itertools
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from pressure_data import Cases
from train_mesh_geometry import json_write, sha256
from train_patch_pressure import parser as train_parser

ROOT = Path(__file__).resolve().parent
ARMS = ("pretrained", "scratch")


def _read_selection(path):
    path = Path(path).resolve()
    selection = json.loads(path.read_text())
    protocol_path = path.parent / selection["protocol_path"]
    if sha256(protocol_path) != selection["protocol_sha256"]:
        raise ValueError("Tuning protocol changed after selection")
    protocol = json.loads(protocol_path.read_text())
    expected = {(arm, grid, seed) for arm in ARMS for grid in range(len(protocol["grid"])) for seed in protocol["seeds"]}
    seen, scores, reports = set(), {}, {}
    for trial in selection["trials"]:
        key = (trial["arm"], trial["grid_index"], trial["seed"])
        if key in seen or key not in expected:
            raise ValueError("Missing, duplicate or unregistered tuning trial")
        seen.add(key)
        report_path = path.parent / trial["validation_path"]
        if sha256(report_path) != trial["validation_sha256"]:
            raise ValueError("Validation report changed after selection")
        report = json.loads(report_path.read_text())
        config = report["training_config"]
        pair = protocol["grid"][key[1]]
        if report["split"] != "val" or report["data_fingerprint"] != protocol["data_fingerprint"]:
            raise ValueError("Tuning must use the registered validation split")
        if config["seed"] != key[2] or config["init"] != key[0] or any(config[k] != pair[k] for k in ("lr", "encoder_lr")):
            raise ValueError("Trial settings differ from the registered grid")
        common = {k: v for k, v in config.items() if k not in ("init", "seed", "lr", "encoder_lr")}
        if common != protocol["training_settings"]:
            raise ValueError("Candidate training budget or settings differ from the registered protocol")
        if report["provenance"]["geometry_checkpoint_sha256"] != protocol["geometry_checkpoint_sha256"]:
            raise ValueError("Source geometry checkpoint changed")
        value = report["selection_score"]
        if value is None or not np.isfinite(value):
            raise ValueError("Undefined validation score")
        scores.setdefault(key[:2], []).append(value)
        reports[key] = report
    if seen != expected:
        raise ValueError("The registered search budget was not completed for both arms")
    for arm in ARMS:
        winner = min(range(len(protocol["grid"])), key=lambda g: (float(np.mean(scores[(arm, g)])), g))
        if selection["winner_grid"][arm] != winner:
            raise ValueError("Selected hyperparameters are not the registered validation minimum")
    return selection, protocol, reports


def verify_selection(path, pretrained=None, scratch=None):
    selection, protocol, reports = _read_selection(path)
    if (pretrained is None) != (scratch is None):
        raise ValueError("Provide both comparison reports")
    if pretrained is not None:
        for arm, report in (("pretrained", pretrained), ("scratch", scratch)):
            key = (arm, selection["winner_grid"][arm], report["training_config"]["seed"])
            if key not in reports or report["checkpoint_sha256"] != reports[key]["checkpoint_sha256"]:
                raise ValueError("Test result is not from the validation-selected checkpoint")
            if report["training_config"] != reports[key]["training_config"] or report["model_config"] != reports[key]["model_config"]:
                raise ValueError("Test report configuration differs from the selected validation report")
            if report["data_fingerprint"] != protocol["data_fingerprint"] or report["split"] != "test":
                raise ValueError("Different test population or scope")
    return True


def execute(command, logpath):
    print("Running " + str(logpath.parent), flush=True)
    with logpath.open("a", buffering=1) as log:
        subprocess.run([sys.executable, str(ROOT/"train_patch_pressure.py")] + command,
                       stdout=log, stderr=subprocess.STDOUT, check=True)


def run(args, extra):
    forbidden = {"--init", "--seed", "--lr", "--encoder-lr", "--stop-after", "--evaluate-test"}
    if any(s.split("=")[0] in forbidden for s in extra):
        raise ValueError("Tuning owns initialization, seeds, rate grid and test timing; do not override them in training options")
    if any(v <= 0 or not np.isfinite(v) for v in args.encoder_lrs + args.decoder_lrs):
        raise ValueError("Learning rates must be finite and positive")
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 for s in args.seeds):
        raise ValueError("Seeds must be distinct nonnegative integers")
    grid = [dict(encoder_lr=e, lr=d) for e, d in itertools.product(args.encoder_lrs, args.decoder_lrs)]
    if len({(v["encoder_lr"], v["lr"]) for v in grid}) != len(grid):
        raise ValueError("Duplicate learning-rate candidate")
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    manifest, checkpoint = str(Path(args.manifest).resolve()), str(Path(args.geometry_checkpoint).resolve())
    source = torch.load(checkpoint, weights_only=True, map_location="cpu")
    if source.get("step", 0) <= 0:
        raise ValueError("Pretrained/scratch tuning requires a geometry checkpoint with actual training updates")
    base = ["train"] + extra + ["--manifest", manifest, "--geometry-checkpoint", checkpoint]
    sample = train_parser().parse_args(base + ["--out", str(out/"template")])
    settings = {k: v for k, v in vars(sample).items() if k not in ("func", "out", "resume", "stop_after", "init", "seed", "lr", "encoder_lr")}
    data = Cases(manifest, weighting=sample.weighting)
    if data.scope != "held_out_cars" and not args.diagnostic_same_car:
        raise ValueError("The main comparison requires held-out cars; --diagnostic-same-car explicitly labels a within-car test")
    if not data.by_split["test"]:
        raise ValueError("A predeclared test split is required")
    protocol = dict(version="patch-pressure-tuning-v1", grid=grid, seeds=args.seeds,
                    selection_rule="minimum mean validation selection_score across seeds; tie -> first registered grid entry",
                    training_settings=settings, data_fingerprint=data.audit["fingerprint"], scope=data.scope,
                    geometry_checkpoint_sha256=sha256(checkpoint), geometry_checkpoint_step=source["step"],
                    model_family="polynomial_patch_encoder")
    protocol_path = out/"protocol.json"
    if protocol_path.exists():
        if not args.resume or json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("Existing protocol: resume unchanged settings or use a new directory")
    else:
        json_write(protocol_path, protocol)
    trials, scores, states = [], {}, {}
    for arm in ARMS:
        for gi, pair in enumerate(grid):
            for seed in args.seeds:
                directory = out/arm/f"grid_{gi:03d}"/f"seed_{seed}"
                directory.mkdir(parents=True, exist_ok=True)
                command = base + ["--init", arm, "--seed", str(seed), "--lr", str(pair["lr"]),
                                  "--encoder-lr", str(pair["encoder_lr"]), "--out", str(directory)]
                status_path = directory/"status.json"
                completed = status_path.exists() and json.loads(status_path.read_text())["reason"] != "paused"
                if not completed:
                    if (directory/"last.pt").exists():
                        command.append("--resume")
                    execute(command, directory/"run.log")
                report_path = directory/"validation_selected.json"
                report = json.loads(report_path.read_text())
                if sha256(directory/"best.pt") != report["checkpoint_sha256"]:
                    raise ValueError("Candidate checkpoint changed after validation")
                value = report["selection_score"]
                if value is None or not np.isfinite(value):
                    raise ValueError("Candidate validation score is undefined")
                key = (arm, gi)
                scores.setdefault(key, []).append(value)
                states[(arm, gi, seed)] = directory
                trials.append(dict(arm=arm, grid_index=gi, seed=seed,
                                   validation_path=str(report_path.relative_to(out)),
                                   validation_sha256=sha256(report_path),
                                   training_stop_reason=json.loads(status_path.read_text())["reason"]))
    winner = {arm: min(range(len(grid)), key=lambda g: (float(np.mean(scores[(arm, g)])), g)) for arm in ARMS}
    selection = dict(protocol_path="protocol.json", protocol_sha256=sha256(protocol_path), trials=trials,
                     winner_grid=winner, selected_rates={arm: grid[winner[arm]] for arm in ARMS},
                     test_scores_used_for_selection=False,
                     all_trials_stopped_by_validation_patience=all(t["training_stop_reason"] == "validation_patience" for t in trials))
    selection_path = out/"selection.json"
    if selection_path.exists():
        if json.loads(selection_path.read_text()) != selection:
            raise ValueError("Existing selection receipt differs; no test scoring performed")
    else:
        json_write(selection_path, selection)
    verify_selection(selection_path)
    # No test evaluation is launched before the validated selection is saved.
    per_seed = []
    for seed in args.seeds:
        test_paths = {}
        for arm in ARMS:
            directory = states[(arm, winner[arm], seed)]
            test_path = directory/"selected_test.json"
            if not test_path.exists():
                execute(["evaluate", "--manifest", manifest, "--checkpoint", str(directory/"best.pt"),
                         "--split", "test", "--out", str(test_path), "--device", sample.device,
                         "--threads", str(sample.threads), "--query-chunk", str(sample.query_chunk),
                         "--weighting", sample.weighting], directory/"test.log")
            test_paths[arm] = test_path
        result_path = out/f"comparison_seed_{seed}.json"
        execute(["compare", "--pretrained", str(test_paths["pretrained"]), "--scratch", str(test_paths["scratch"]),
                 "--selection", str(selection_path), "--out", str(result_path)], out/f"compare_seed_{seed}.log")
        result = json.loads(result_path.read_text()); result["seed"] = seed; per_seed.append(result)
    summary = dict(scope=data.scope, model_family="polynomial_patch_encoder", selected_rates=selection["selected_rates"],
                   selection_sha256=sha256(selection_path), equal_search_budget=True,
                   per_seed=per_seed, metric="case_mean_area_weighted_reference_standard_deviation_nrmse" if sample.weighting == "area" else "case_mean_weighted_reference_standard_deviation_nrmse",
                   all_trials_stopped_by_validation_patience=selection["all_trials_stopped_by_validation_patience"],
                   original_areals_or_hessian_evaluated=False)
    json_write(out/"summary.json", summary)
    print(json.dumps(dict(selected_rates=selection["selected_rates"], summary=str(out/"summary.json")), indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--manifest", required=True); p.add_argument("--geometry-checkpoint", required=True)
    p.add_argument("--out", required=True); p.add_argument("--seeds", type=int, nargs="+", default=[17])
    p.add_argument("--encoder-lrs", type=float, nargs="+", default=[2e-5, 2e-4, 1e-3])
    p.add_argument("--decoder-lrs", type=float, nargs="+", default=[2e-4])
    p.add_argument("--resume", action="store_true"); p.add_argument("--diagnostic-same-car", action="store_true")
    args, extra = p.parse_known_args()
    run(args, extra)
