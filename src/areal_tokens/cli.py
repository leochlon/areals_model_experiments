"""Command-line entry points; every command also has a Python API."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from .config import ExperimentConfig
from .evaluate import encode_checkpoint, evaluate_checkpoint
from .field import FieldTrainConfig, train_field
from .io import write_json
from .prepare import DISTANCE_BACKENDS, make_demo, prepare_mesh, prepare_mesh_manifest
from .probe import run_probe
from .train import resolve_device, train
from .validation import validate_manifest, write_filtered_manifest


def _compute_options(parser, *, workers=False):
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument(
        "--threads", type=int, default=min(8, os.cpu_count() or 1), help="PyTorch CPU threads per process"
    )
    if workers:
        parser.add_argument("--workers", type=int, default=0)


def build_parser():
    parser = argparse.ArgumentParser(prog="areal", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("make-demo", help="Create analytical spheres/boxes for software verification")
    demo.add_argument("--out", required=True)
    for name, default in (
        ("train", 12),
        ("val", 4),
        ("test", 4),
        ("points", 512),
        ("queries", 1024),
        ("seed", 11),
    ):
        demo.add_argument("--" + name, type=int, default=default)

    mesh = sub.add_parser(
        "prepare-mesh", help="Area-sample meshes and generate true triangle-distance labels"
    )
    source = mesh.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="One mesh; --out is a .npz file")
    source.add_argument("--manifest", help="A split manifest with mesh paths; --out is a new directory")
    mesh.add_argument("--out", required=True)
    mesh.add_argument("--points", type=int, default=16384)
    mesh.add_argument("--queries", type=int, default=32768)
    mesh.add_argument("--seed", type=int, default=17)
    mesh.add_argument(
        "--backend",
        choices=DISTANCE_BACKENDS,
        default="trimesh",
        help="Point-to-triangle distance: trimesh (as 0.2.0), igl (libigl, much faster), or auto",
    )
    mesh.add_argument("--workers", type=int, default=1, help="Worker processes (manifest mode)")
    mesh.add_argument("--resume", action="store_true", help="Continue a partly converted directory")
    mesh.add_argument(
        "--skip-bad", action="store_true", help="List unconvertible meshes in failures.json instead of stopping"
    )

    validate = sub.add_parser("validate", help="Check schema, normals, weights and split separation")
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--target-mode", choices=("provided", "surface_samples"), default="provided")
    validate.add_argument("--out", help="Optional JSON audit report; train --audit reuses it")
    validate.add_argument("--workers", type=int, default=1, help="Worker processes")
    validate.add_argument(
        "--skip-bad", action="store_true", help="Report unreadable records under failures instead of stopping"
    )
    validate.add_argument(
        "--filtered-manifest", help="With --skip-bad: write the manifest without the failed records here"
    )

    fit = sub.add_parser("train", help="Masked geometry pretraining (also supports torchrun)")
    fit.add_argument("--manifest", required=True)
    fit.add_argument("--config", help="Experiment JSON; required for a new run")
    fit.add_argument("--out", required=True)
    fit.add_argument("--resume", help="The same output directory's last.pt")
    fit.add_argument(
        "--stop-after-epochs", type=int, help="Stop at this absolute epoch; preserve planned schedule"
    )
    fit.add_argument("--deterministic", action="store_true")
    fit.add_argument("--audit", help="A validate --out report for this manifest, to skip rereading every file")
    _compute_options(fit)

    field = sub.add_parser("train-field", help="Fine-tune a checkpoint into a shape-and-field encoder by masked completion")
    field.add_argument("--manifest", required=True)
    field.add_argument("--fields", required=True, help="Directory of <id>.npz files with points, values and weights")
    field.add_argument("--checkpoint", required=True, help="Geometry checkpoint to start from")
    field.add_argument("--out", required=True)
    defaults = FieldTrainConfig()
    field.add_argument("--epochs", type=int, default=defaults.epochs)
    field.add_argument("--batch-size", type=int, default=defaults.batch_size)
    field.add_argument("--lr", type=float, default=defaults.learning_rate, help="Pretrained weights")
    field.add_argument("--lr-new", type=float, default=defaults.new_learning_rate, help="The new input columns and head")
    field.add_argument("--hidden", type=float, nargs=2, default=defaults.hidden_range, help="Range of the hidden fraction")
    field.add_argument("--full-hidden", type=float, default=defaults.full_hidden_probability,
                       help="Probability that a car has no values given")
    field.add_argument("--queries", type=int, default=defaults.queries)
    field.add_argument("--no-augment", action="store_true", help="No sign flip or ratio change of the field")
    field.add_argument("--seed", type=int, default=defaults.seed)
    field.add_argument("--amp", choices=("none", "bf16", "fp16"), default=defaults.amp)
    _compute_options(field, workers=True)

    evaluate = sub.add_parser(
        "evaluate", help="Frozen reconstruction on fixed queries with per-family metrics"
    )
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--out", required=True)
    evaluate.add_argument("--split", choices=("train", "val", "test"), default="test")
    evaluate.add_argument("--masks", type=float, nargs="+", help="Default: full and configured training mask")
    evaluate.add_argument("--ablation", choices=("none", "zero_tokens", "initial_tokens"), default="none")
    evaluate.add_argument(
        "--normal-readout",
        choices=("head", "gradient"),
        default="head",
        help="Normals from the normal head, or from the gradient of the predicted distance field",
    )
    _compute_options(evaluate, workers=True)

    encode = sub.add_parser("encode", help="Export vector tokens, coordinates, weights and retained frames")
    encode.add_argument("--checkpoint", required=True)
    encode.add_argument("--manifest", required=True)
    encode.add_argument("--out", required=True)
    encode.add_argument("--split", choices=("all", "train", "val", "test"), default="all")
    encode.add_argument(
        "--common-queries", help="Optional .npy world coordinates [Q,3] for shared query features"
    )
    _compute_options(encode, workers=True)

    probe = sub.add_parser(
        "probe", help="Frozen token/geometry/random controls, with split-safe ridge selection"
    )
    probe.add_argument("--checkpoint", required=True)
    probe.add_argument("--manifest", required=True)
    probe.add_argument("--out", required=True)
    probe.add_argument("--queries", type=int, default=512)
    probe.add_argument("--bandwidth", type=float, default=0.35)
    probe.add_argument("--alphas", type=float, nargs="+", default=[1e-5, 1e-3, 0.1, 1.0, 10.0])
    probe.add_argument("--random-seed", type=int)
    _compute_options(probe)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if hasattr(args, "threads"):
        if args.threads < 1:
            parser.error("--threads must be positive")
        torch.set_num_threads(args.threads)
    if hasattr(args, "workers") and args.workers < 0:
        parser.error("--workers must be nonnegative")
    if args.command == "make-demo":
        manifest = make_demo(
            args.out,
            train=args.train,
            val=args.val,
            test=args.test,
            points=args.points,
            queries=args.queries,
            seed=args.seed,
        )
        result = dict(manifest=str(manifest))
    elif args.command == "prepare-mesh":
        options = dict(points=args.points, queries=args.queries, seed=args.seed, backend=args.backend)
        if args.input:
            if args.workers != 1 or args.resume or args.skip_bad:
                parser.error("--workers, --resume and --skip-bad apply to --manifest conversion")
            result = prepare_mesh(args.input, args.out, **options)
        else:
            if args.workers < 1:
                parser.error("--workers must be positive")
            manifest = prepare_mesh_manifest(
                args.manifest,
                args.out,
                workers=args.workers,
                resume=args.resume,
                bad_records="skip" if args.skip_bad else "raise",
                **options,
            )
            result = dict(manifest=str(manifest), failures=str(Path(args.out) / "failures.json"))
    elif args.command == "validate":
        if args.workers < 1:
            parser.error("--workers must be positive")
        if args.filtered_manifest and not args.skip_bad:
            parser.error("--filtered-manifest requires --skip-bad")
        report = validate_manifest(
            args.manifest,
            require_targets=args.target_mode == "provided",
            workers=args.workers,
            bad_records="skip" if args.skip_bad else "raise",
        )
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            write_json(args.out, report)
        result = dict(
            splits=report["splits"],
            units=report["units"],
            failures=len(report["failures"]),
        )
        if args.filtered_manifest:
            result["filtered_manifest"] = str(write_filtered_manifest(args.manifest, report, args.filtered_manifest))
    elif args.command == "train":
        if not args.config and not args.resume:
            parser.error("New training runs require --config")
        config = ExperimentConfig.load(args.config or Path(args.out) / "config.json")
        result = train(
            args.manifest,
            config,
            args.out,
            device=args.device,
            resume=args.resume,
            stop_after_epochs=args.stop_after_epochs,
            deterministic=args.deterministic,
            audit=args.audit,
        )
    elif args.command == "train-field":
        config = FieldTrainConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            new_learning_rate=args.lr_new,
            hidden_range=tuple(args.hidden),
            full_hidden_probability=args.full_hidden,
            queries=args.queries,
            sign_flip=not args.no_augment,
            ratio_range=(1.0, 1.0) if args.no_augment else FieldTrainConfig().ratio_range,
            workers=args.workers,
            seed=args.seed,
            amp=args.amp,
        )
        result = train_field(args.manifest, args.fields, args.checkpoint, args.out, config, resolve_device(args.device))
    elif args.command == "evaluate":
        report = evaluate_checkpoint(
            args.checkpoint,
            args.manifest,
            args.out,
            resolve_device(args.device),
            split=args.split,
            masks=args.masks,
            ablation=args.ablation,
            workers=args.workers,
            normal_readout=args.normal_readout,
        )
        result = dict(
            output=args.out,
            settings=[
                dict(mask_fraction=value["mask_fraction"], **value["aggregate"])
                for value in report["settings"]
            ],
        )
    elif args.command == "encode":
        report = encode_checkpoint(
            args.checkpoint,
            args.manifest,
            args.out,
            resolve_device(args.device),
            split=args.split,
            common_queries=args.common_queries,
            workers=args.workers,
        )
        result = dict(output=args.out, shapes=len(report["records"]), width=report["width"])
    else:
        report = run_probe(
            args.checkpoint,
            args.manifest,
            args.out,
            resolve_device(args.device),
            queries=args.queries,
            bandwidth=args.bandwidth,
            alphas=args.alphas,
            random_seed=args.random_seed,
        )
        result = dict(
            output=args.out,
            test={name: value["test"]["normalized_rmse"] for name, value in report["results"].items()},
        )
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return 0
