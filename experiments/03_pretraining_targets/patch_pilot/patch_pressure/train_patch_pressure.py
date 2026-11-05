#!/usr/bin/env python3
"""Fine-tune a train_mesh_geometry.py checkpoint to predict CFD pressure.

See README.md for data conversion, same-car diagnostics, held-out-car training,
matched scratch controls, metrics and inference without pressure labels.
"""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

from train_mesh_geometry import choose_device, json_write, sha256
from pressure_model import PressureModel, load_geometry_initialization
from pressure_data import Cases, load_geometry, tensor_scene, normalized_queries, neighbors, case_metrics, summarize_metrics

VERSION = "patch-pressure-v2"


def state_hash(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def setup(seed, threads, deterministic=True):
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(deterministic)


def initialize(args, stats):
    source = load_geometry_initialization(args.geometry_checkpoint)
    # Reset AFTER inspecting the source, so scratch and pretrained receive the
    # exact same decoder initialization and sampling RNG for a given seed.
    setup(args.seed, args.threads, not args.allow_nondeterministic)
    model = PressureModel(source["model_config"], len(stats["condition_mean"]),
                          width=args.width, heads=args.heads, global_tokens=args.global_tokens,
                          global_depth=args.global_depth, neighbors=args.neighbors,
                          encoder_chunk=args.encoder_chunk,
                          gradient_checkpointing=not args.no_gradient_checkpointing)
    model.set_normalization(stats)
    if args.init == "pretrained":
        model.encoder.load_state_dict(source["model"], strict=True)
    model.freeze_encoder(args.encoder_mode == "frozen")
    provenance = dict(geometry_checkpoint_sha256=sha256(args.geometry_checkpoint),
                      geometry_checkpoint_step=source["step"], initialization=args.init,
                      model_family="polynomial_patch_encoder", pretraining_data_sha256=source.get("data_sha256"),
                      encoder_initial_sha256=state_hash(model.encoder.state_dict()),
                      decoder_initial_sha256=state_hash({k: v for k, v in model.state_dict().items() if not k.startswith("encoder.")}))
    return model, provenance


def optimizer_for(model, args):
    decoder = [p for name, p in model.named_parameters() if not name.startswith("encoder.") and p.requires_grad]
    groups = [dict(params=decoder, lr=args.lr)]
    encoder = [p for p in model.encoder.parameters() if p.requires_grad]
    if encoder:
        groups.append(dict(params=encoder, lr=args.encoder_lr))
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


def save_checkpoint(path, model, optimizer, step, rng, best, stale, data, config, provenance, history):
    payload = dict(version=VERSION, model_config=model.config, model=model.state_dict(),
                   optimizer=optimizer.state_dict(), step=step, best=best, stale=stale,
                   rng=json.dumps(rng.bit_generator.state), torch_rng=torch.get_rng_state(),
                   cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                   data_fingerprint=data.audit["fingerprint"], condition_names=data.condition_names,
                   training_data_audit=data.audit,
                   training_config=config, provenance=provenance, validation_history=history)
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def restore(path, device):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("version") not in (VERSION, "patch-pressure-v1"):
        raise ValueError("Expected a pressure fine-tuning checkpoint, not a geometry checkpoint")
    model = PressureModel(**state["model_config"])
    model.load_state_dict(state["model"], strict=True)
    model.freeze_encoder(state["training_config"]["encoder_mode"] == "frozen")
    return model.to(device), state


def check_evaluation_split(state, data, split):
    if state["condition_names"] != data.condition_names:
        raise ValueError("Condition names/order differ from the pressure checkpoint")
    if data.scope in ("held_out_queries_same_car", "held_out_regions_same_car"):
        if data.audit["fingerprint"] != state["data_fingerprint"]:
            raise ValueError("Same-car scoring requires the original audited query splits; use predict for new coordinates")
    elif split != "train":
        training = [e for e in state["training_data_audit"]["cases"] if e["split"] == "train"]
        cars = {e["car_id"] for e in training}
        meshes = {e["source_mesh_sha256"] for e in training}
        fields = {e["field_sha256"] for e in training}
        for e in data.audit["cases"]:
            if e["split"] == split and (e["car_id"] in cars or e["source_mesh_sha256"] in meshes or e["field_sha256"] in fields):
                raise ValueError("Evaluation car/mesh/field overlaps pressure training data in checkpoint")


@torch.no_grad()
def score(model, data, split, device, chunk=2048, max_queries=0, seed=1000017, predictions_dir=None, selection_metric="nrmse_std"):
    if not data.by_split[split]:
        raise ValueError(f"Requested {split} split is empty; no fallback")
    if model.config["condition_dim"] != len(data.condition_names):
        raise ValueError("Condition dimension mismatch")
    model.eval()
    records = []
    for case_id in data.by_split[split]:
        e, f = data.entries[case_id], data.get(case_id)
        g = data.geometry(e)
        if model.config["encoder_config"]["degree"] != g["degree"]:
            raise ValueError("Geometry polynomial degree mismatch")
        ids = np.arange(len(f["cp"]))
        if max_queries and len(ids) > max_queries:
            # Same held-out subset at every validation, never resampled using
            # model results. Final evaluation defaults to all test points.
            stable = int(hashlib.sha256(e["id"].encode()).hexdigest()[:8], 16)
            ids = np.sort(np.random.default_rng(seed + stable).choice(ids, max_queries, replace=False))
        encoded = model.encode_scene(tensor_scene(g, device), torch.tensor(e["conditions"], device=device))
        outputs = []
        for part in (ids[j:j + chunk] for j in range(0, len(ids), chunk)):
            q = normalized_queries(g, f["xyz"][part])
            ni = neighbors(g, q, model.config["neighbors"])
            outputs.append(model.decode(encoded, torch.tensor(q, device=device), torch.tensor(ni, device=device)).cpu().numpy())
        pred = np.concatenate(outputs)
        if not np.isfinite(pred).all():
            raise ValueError("Nonfinite pressure prediction")
        metrics = case_metrics(pred, f["cp"][ids], f["weights"][ids], float(model.cp_std))
        q_dynamic = f["metadata"].get("dynamic_pressure")
        if q_dynamic is not None:
            metrics["rmse_static_pressure"] = metrics["rmse_cp"] * q_dynamic
            metrics["static_pressure_units"] = f["metadata"].get("pressure_units", "input pressure units")
        records.append(dict(id=e["id"], car_id=e["car_id"],
                            weight_measure=f["metadata"].get("weight_measure", "unspecified"),
                            query_indices_sha256=hashlib.sha256(np.asarray(f["indices"][ids], dtype="<i8").tobytes()).hexdigest(),
                            **metrics))
        if predictions_dir:
            dest = Path(predictions_dir)
            dest.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dest / f"case_{case_id:06d}.npz", indices=f["indices"][ids], xyz=f["xyz"][ids],
                                cp_predicted=pred, cp_reference=f["cp"][ids], weights=f["weights"][ids],
                                metadata=np.array(json.dumps(dict(id=e["id"], car_id=e["car_id"]))))
        print(json.dumps(dict(split=split, case=e["id"], rmse_cp=metrics["rmse_cp"],
                              nrmse_reference_std=metrics["nrmse_reference_std"])), flush=True)
    result = summarize_metrics(records, data.scope, data.audit["fingerprint"], selection_metric)
    result.update(version=VERSION, split=split)
    return result


def train(args):
    device = choose_device(args.device)
    data = Cases(args.manifest, weighting=args.weighting)
    stats = data.normalization()
    model, provenance = initialize(args, stats)
    expected_degree = model.config["encoder_config"]["degree"]
    if any(r["degree"] != expected_degree for r in data.audit["cases"]):
        raise ValueError("Geometry cache and checkpoint polynomial degrees differ")
    model.to(device)
    optimizer = optimizer_for(model, args)
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = {k: v for k, v in vars(args).items() if k not in ("func", "out", "resume", "stop_after")}
    start, best, stale, history = 0, float("inf"), 0, []
    if args.resume:
        model, state = restore(out / "last.pt", device)
        if state["data_fingerprint"] != data.audit["fingerprint"] or state["training_config"] != config or state["provenance"] != provenance:
            raise ValueError("Resume requires unchanged data, initialization and training settings")
        optimizer = optimizer_for(model, args)
        optimizer.load_state_dict(state["optimizer"])
        start, best, stale, history = state["step"], state["best"], state["stale"], state["validation_history"]
        rng.bit_generator.state = json.loads(state["rng"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    elif any((out / name).exists() for name in ("last.pt", "best.pt")):
        raise ValueError("Output already contains checkpoints; use --resume or a new output directory")
    json_write(out / "data_audit.json", data.audit)
    json_write(out / "config.json", dict(**config, model=model.config, normalization=stats, provenance=provenance))
    json_write(out / "environment.json", dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
                                             device=str(device), cuda=torch.version.cuda,
                                             deterministic=torch.are_deterministic_algorithms_enabled()))
    print(f"Pressure training: {sum(p.numel() for p in model.parameters()):,} parameters; {args.init}; {args.encoder_mode}; {data.scope}", flush=True)
    if start == 0:
        initial = score(model, data, "val", device, args.query_chunk, args.validation_queries, args.seed + 1000000,
                        selection_metric=args.selection_metric)
        best = initial["selection_score"]
        if best is None or not np.isfinite(best):
            raise ValueError("Validation metric is undefined (e.g. constant reference pressure); choose --selection-metric mse for that diagnostic")
        history.append(dict(step=0, selection_score=best))
        json_write(out / "validation_initial.json", initial)
        save_checkpoint(out / "best.pt", model, optimizer, 0, rng, best, stale, data, config, provenance, history)
    limit = min(args.steps, args.stop_after) if args.stop_after else args.steps
    if limit < start:
        raise ValueError("stop-after precedes the resumed step")
    started = time.monotonic()
    reason, step = "maximum_steps", start
    with (out / "train.jsonl").open("a" if args.resume else "w", buffering=1) as log:
        for step in range(start + 1, limit + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            case_ids = rng.choice(data.by_split["train"], size=args.cases_per_step, replace=True)
            total_loss = 0.
            for i in case_ids:
                e, f = data.entries[i], data.get(int(i))
                g = data.geometry(e)
                ids = rng.choice(len(f["cp"]), size=args.queries, replace=True, p=f["weights"] / f["weights"].sum())
                q = normalized_queries(g, f["xyz"][ids])
                ni = neighbors(g, q, args.neighbors)
                target = (torch.tensor(f["cp"][ids], device=device, dtype=torch.float32) - model.cp_mean) / model.cp_std
                pred = model(tensor_scene(g, device), torch.tensor(e["conditions"], device=device),
                             torch.tensor(q, device=device), torch.tensor(ni, device=device), standardized=True)
                loss = (pred - target).square().mean() / args.cases_per_step
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite pressure training loss")
                loss.backward()
                total_loss += float(loss.detach())
            grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad, error_if_nonfinite=True))
            optimizer.step()
            log.write(json.dumps(dict(step=step, standardized_mse=total_loss, gradient_norm=grad)) + "\n")
            checked = step % args.validate_every == 0 or step == args.steps
            if checked:
                metrics = score(model, data, "val", device, args.query_chunk, args.validation_queries, args.seed + 1000000,
                                selection_metric=args.selection_metric)
                value = metrics["selection_score"]
                history.append(dict(step=step, selection_score=value))
                if value < best - args.min_delta:
                    best, stale = value, 0
                    save_checkpoint(out / "best.pt", model, optimizer, step, rng, best, stale, data, config, provenance, history)
                else:
                    stale += 1
                json_write(out / "validation_latest.json", dict(step=step, **metrics))
                print(json.dumps(dict(step=step, loss=total_loss, validation=value, best=best,
                                      elapsed_seconds=round(time.monotonic() - started, 2))), flush=True)
            converged = checked and args.patience > 0 and step >= args.min_steps and stale >= args.patience
            if checked or step == limit or converged:
                save_checkpoint(out / "last.pt", model, optimizer, step, rng, best, stale, data, config, provenance, history)
            if converged:
                reason = "validation_patience"
                break
    if step < args.steps and reason != "validation_patience":
        json_write(out / "status.json", dict(step=step, reason="paused", pressure_accuracy_claimed=False))
        print(f"Paused at {step}. Resume the same command with --resume.", flush=True)
        return
    best_model, best_state = restore(out / "best.pt", device)
    selected = score(best_model, data, "val", device, args.query_chunk, args.validation_queries, args.seed + 1000000,
                     selection_metric=args.selection_metric)
    selected.update(checkpoint_sha256=sha256(out / "best.pt"), checkpoint_step=best_state["step"],
                    training_config=config, model_config=best_model.config, provenance=provenance)
    json_write(out / "validation_selected.json", selected)
    json_write(out / "status.json", dict(step=step, best_step=best_state["step"], reason=reason,
                                       stopped_by_validation_patience=reason == "validation_patience",
                                       pressure_accuracy_claimed=False))
    if args.evaluate_test and data.by_split["test"]:
        evaluate(argparse.Namespace(manifest=args.manifest, checkpoint=str(out / "best.pt"),
                                   split="test", out=str(out / "test_metrics.json"), device=args.device,
                                   query_chunk=args.query_chunk, max_queries=0, seed=args.seed + 2000000,
                                   predictions_dir=None, threads=args.threads, weighting=args.weighting,
                                   allow_nondeterministic=args.allow_nondeterministic))
    else:
        print("Validation saved. Test scoring is separate; select hyperparameters before evaluate.", flush=True)


def evaluate(args):
    device = choose_device(args.device)
    data = Cases(args.manifest, require_train=False, weighting=args.weighting)
    before = sha256(args.checkpoint)
    model, state = restore(args.checkpoint, device)
    allow_nondeterministic = getattr(args, "allow_nondeterministic", None)
    if allow_nondeterministic is None:
        allow_nondeterministic = state["training_config"].get("allow_nondeterministic", False)
    setup(args.seed, args.threads, not allow_nondeterministic)
    check_evaluation_split(state, data, args.split)
    result = score(model, data, args.split, device, args.query_chunk, args.max_queries, args.seed, args.predictions_dir,
                   selection_metric=state["training_config"].get("selection_metric", "nrmse_std"))
    result.update(checkpoint_sha256=before, checkpoint_step=state["step"], provenance=state["provenance"],
                  training_config=state["training_config"], model_config=state["model_config"],
                  training_data_fingerprint=state["data_fingerprint"], deterministic=torch.are_deterministic_algorithms_enabled(),
                  device=str(device))
    if before != sha256(args.checkpoint):
        raise AssertionError("Evaluation mutated the checkpoint")
    json_write(args.out, result)
    print(json.dumps({k: v for k, v in result.items() if k in ("case_mean_nrmse_reference_std", "case_balanced_rmse_cp", "checkpoint_step", "scope", "evaluated_queries")}))
    return result


def predict(args):
    device = choose_device(args.device)
    model, state = restore(args.checkpoint, device)
    allow_nondeterministic = getattr(args, "allow_nondeterministic", None)
    if allow_nondeterministic is None:
        allow_nondeterministic = state["training_config"].get("allow_nondeterministic", False)
    setup(0, args.threads, not allow_nondeterministic)
    model.eval()
    g = load_geometry(args.geometry, args.mesh_index)
    if g["degree"] != model.config["encoder_config"]["degree"]:
        raise ValueError("Geometry degree differs from the checkpoint")
    if len(args.conditions) != model.config["condition_dim"] or not np.isfinite(args.conditions).all():
        raise ValueError(f"Provide conditions in this order: {state['condition_names']}")
    with np.load(args.query_file, allow_pickle=False) as z:
        xyz = np.asarray(z["xyz"], dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError("Query file needs finite xyz[N,3] in the geometry mesh's units/frame")
    outputs = []
    with torch.no_grad():
        encoded = model.encode_scene(tensor_scene(g, device), torch.tensor(args.conditions, device=device))
        for part in (xyz[i:i + args.query_chunk] for i in range(0, len(xyz), args.query_chunk)):
            q = normalized_queries(g, part)
            ni = neighbors(g, q, model.config["neighbors"])
            outputs.append(model.decode(encoded, torch.tensor(q, device=device), torch.tensor(ni, device=device)).cpu().numpy())
    cp = np.concatenate(outputs)
    if not np.isfinite(cp).all():
        raise ValueError("Nonfinite prediction")
    arrays = dict(xyz=xyz, cp_predicted=cp,
                  metadata=np.array(json.dumps(dict(checkpoint_sha256=sha256(args.checkpoint),
                                                   conditions=args.conditions, condition_names=state["condition_names"],
                                                   deterministic=torch.are_deterministic_algorithms_enabled(), device=str(device)))))
    if args.dynamic_pressure is not None or args.reference_pressure is not None:
        if args.dynamic_pressure is None or args.reference_pressure is None or args.dynamic_pressure <= 0:
            raise ValueError("Physical pressure output requires positive --dynamic-pressure and --reference-pressure")
        arrays["pressure_predicted"] = cp * args.dynamic_pressure + args.reference_pressure
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    print(f"Predicted {len(cp):,} pressure queries -> {args.out}")


def compare(args):
    a, b = (json.loads(Path(p).read_text()) for p in (args.pretrained, args.scratch))
    if a["provenance"]["initialization"] != "pretrained" or b["provenance"]["initialization"] != "scratch":
        raise ValueError("Expected pretrained and scratch evaluation reports in that order")
    for key in ("data_fingerprint", "training_data_fingerprint", "split", "model_config", "metric_definitions", "area_weighted"):
        if a[key] != b[key]:
            raise ValueError(f"Comparison mismatch: {key}")
    for key in ("decoder_initial_sha256", "geometry_checkpoint_sha256"):
        if a["provenance"][key] != b["provenance"][key]:
            raise ValueError(f"Comparison mismatch: {key}")
    ignore = {"init", "lr", "encoder_lr"}
    for key in set(a["training_config"]) | set(b["training_config"]):
        if key not in ignore and a["training_config"].get(key) != b["training_config"].get(key):
            raise ValueError(f"Training recipes differ: {key}")
    rows = []
    for ar, br in zip(a["per_case"], b["per_case"], strict=True):
        if ar["id"] != br["id"] or ar["queries"] != br["queries"] or ar["query_indices_sha256"] != br["query_indices_sha256"]:
            raise ValueError("Different evaluated cases/query counts")
        rows.append(dict(id=ar["id"], pretrained_rmse_cp=ar["rmse_cp"], scratch_rmse_cp=br["rmse_cp"],
                         pretrained_nrmse=ar["nrmse_reference_std"], scratch_nrmse=br["nrmse_reference_std"],
                         rmse_reduction_fraction=1-ar["rmse_cp"]/br["rmse_cp"] if br["rmse_cp"] > 0 else None))
    rates = {arm: {k: report["training_config"][k] for k in ("lr", "encoder_lr")}
             for arm, report in (("pretrained", a), ("scratch", b))}
    selection_verified = False
    if getattr(args, "selection", None):
        from tune_pressure import verify_selection
        selection_verified = verify_selection(args.selection, a, b)
    json_write(args.out, dict(scope=a["scope"], matched_data_architecture_and_seed=True,
                             validation_tuning_verified=selection_verified, learning_rates=rates,
                             pretrained_case_mean_nrmse=a["case_mean_nrmse_reference_std"],
                             scratch_case_mean_nrmse=b["case_mean_nrmse_reference_std"], per_case=rows))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    s = p.add_subparsers(dest="command", required=True)
    t = s.add_parser("train")
    t.add_argument("--manifest", required=True); t.add_argument("--geometry-checkpoint", required=True)
    t.add_argument("--out", required=True); t.add_argument("--init", choices=["pretrained", "scratch"], default="pretrained")
    t.add_argument("--encoder-mode", choices=["finetune", "frozen"], default="finetune")
    t.add_argument("--steps", type=int, default=20000); t.add_argument("--queries", type=int, default=512)
    t.add_argument("--cases-per-step", type=int, default=1)
    t.add_argument("--lr", type=float, default=2e-4); t.add_argument("--encoder-lr", type=float, default=2e-4)
    t.add_argument("--weight-decay", type=float, default=0.); t.add_argument("--clip-grad", type=float, default=1.)
    t.add_argument("--width", type=int, default=128); t.add_argument("--heads", type=int, default=4)
    t.add_argument("--global-tokens", type=int, default=32); t.add_argument("--global-depth", type=int, default=2)
    t.add_argument("--neighbors", type=int, default=8); t.add_argument("--encoder-chunk", type=int, default=32)
    t.add_argument("--validate-every", type=int, default=250); t.add_argument("--validation-queries", type=int, default=8192)
    t.add_argument("--patience", type=int, default=20); t.add_argument("--min-steps", type=int, default=1000)
    t.add_argument("--min-delta", type=float, default=0.)
    t.add_argument("--seed", type=int, default=17); t.add_argument("--stop-after", type=int, default=0)
    t.add_argument("--resume", action="store_true"); t.add_argument("--no-gradient-checkpointing", action="store_true")
    t.add_argument("--allow-nondeterministic", action="store_true"); t.set_defaults(func=train)
    t.add_argument("--weighting", choices=["area", "provided"], default="area")
    t.add_argument("--selection-metric", choices=["nrmse_std", "mse"], default="nrmse_std")
    t.add_argument("--evaluate-test", action="store_true", help="Only for a recipe already fixed before test inspection; tuning never uses this")
    e = s.add_parser("evaluate")
    e.add_argument("--manifest", required=True); e.add_argument("--checkpoint", required=True)
    e.add_argument("--split", choices=["train", "val", "test"], default="test")
    e.add_argument("--out", required=True); e.add_argument("--predictions-dir")
    e.add_argument("--max-queries", type=int, default=0); e.add_argument("--seed", type=int, default=2000017)
    e.add_argument("--weighting", choices=["area", "provided"], default="area")
    e.set_defaults(func=evaluate)
    r = s.add_parser("predict")
    r.add_argument("--geometry", required=True); r.add_argument("--mesh-index", type=int, default=0)
    r.add_argument("--checkpoint", required=True); r.add_argument("--queries", dest="query_file", required=True)
    r.add_argument("--conditions", nargs="*", type=float, required=True); r.add_argument("--out", required=True)
    r.add_argument("--dynamic-pressure", type=float); r.add_argument("--reference-pressure", type=float)
    r.set_defaults(func=predict)
    for cmd in (e, r):
        deterministic = cmd.add_mutually_exclusive_group()
        deterministic.add_argument("--allow-nondeterministic", action="store_true", default=None)
        deterministic.add_argument("--deterministic", dest="allow_nondeterministic", action="store_false")
    for cmd in (t, e, r):
        cmd.add_argument("--device", default="auto"); cmd.add_argument("--threads", type=int, default=1)
        cmd.add_argument("--query-chunk", type=int, default=2048)
    c = s.add_parser("compare")
    c.add_argument("--pretrained", required=True); c.add_argument("--scratch", required=True)
    c.add_argument("--selection", help="Selection receipt from tune_pressure.py")
    c.add_argument("--out", required=True); c.set_defaults(func=compare)
    return p


def main():
    p = parser()
    args = p.parse_args()
    for name in ("steps", "queries", "cases_per_step", "lr", "encoder_lr", "clip_grad", "width", "heads",
                 "global_tokens", "neighbors", "encoder_chunk", "validate_every", "threads", "query_chunk"):
        if hasattr(args, name) and getattr(args, name) <= 0:
            p.error(f"{name} must be positive")
    for name in ("stop_after", "validation_queries", "max_queries", "patience", "min_steps", "min_delta", "global_depth", "weight_decay", "seed"):
        if hasattr(args, name) and getattr(args, name) < 0:
            p.error(f"{name} must be nonnegative")
    args.func(args)


if __name__ == "__main__":
    main()
