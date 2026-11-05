#!/usr/bin/env python3
"""Run GABI (Vadeboncoeur, Duthe, Girolami & Chatzi, ICLR 2026) on the field benchmark's observation sets with the
authors' code (github.com/gduthe/GABI, MIT licence), and score it with field_benchmark.py like every other method.

  export   the benchmark's train and val cars as GABI car datasets (pickled lists of torch_geometric Data: field
           point positions, a graph, the field), plus a config with the paper's car settings (its appendix B.5): GCN
           autoencoder, 6 layers, 100 channels, 100-dimensional latent, batch 100, 10k iterations, AdamW, learning
           rate 1e-3 to 1e-5. The graph is the mesh when the field files hold `faces` [F,3|4] over their points, else
           each point joined to its 8 nearest neighbours
  train    GABI's training script on that config, unchanged (best.pt = lowest validation loss on the val cars)
  predict  GABI-ABC on every observation set of a split (paper: 50k prior samples, the 100 nearest the observations
           kept; it does not state the observation noise for cars, so abc_inference's default 0.01 is used). The
           prediction is the mean of the kept samples; with nothing visible, the mean of the prior samples
  check    the benchmark's leak check, run on GABI (on the CPU by default: GPU sums are not bitwise repeatable,
           which can swap the 100th and 101st nearest sample)

The ABC is GABI's (sampling/abc_inference.py): decode latents drawn from N(0, I) on the car, add noise at the
observed points, keep the samples nearest the observations. Decoding does not depend on the observations, so by
default one car's prior samples serve all its observation sets (--abc shared); each set still draws its own noise and
keeps its own nearest samples, so each prediction is GABI-ABC exactly, and runs in the same draw order with the same
generator give GABI's result to rounding (the adapter's self-test checks this). Reported time per problem is then
amortised over a car's sets. --abc exact calls abc_inference itself for every set.
The paper's Direct Map baseline (a GCN given the coordinates, the observed values with zeros elsewhere and an
observed-node indicator) is not in the released code and is not run; Areal trained from scratch takes the same
inputs.

  python gabi_adapter.py export  --bench bench --out bench/gabi
  python gabi_adapter.py train   --out bench/gabi --gabi-repo GABI
  python gabi_adapter.py predict --bench bench --out bench/gabi --gabi-repo GABI
  python gabi_adapter.py check   --bench bench --out bench/gabi --gabi-repo GABI
  python field_benchmark.py score --bench bench
Test split: predict, check and score with --split test --final.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import pickle
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

import field_benchmark as fb
from areal_tokens.data import stable_seed

NODES_PER_DECODE = 500_000  # nodes decoded together when --decode-batch is 0 (auto): a few GB of GCN messages


def load_module(path: Path, name: str):
    """One file of the GABI repository, without importing its package (whose __init__ pulls in pyro and h5py)."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_faces(fields, identity: str) -> np.ndarray | None:
    """Mesh faces over a car's field points, if its field file has them (triangles, or quads split in two)."""
    with np.load(Path(fields) / f"{identity}.npz", allow_pickle=False) as packed:
        if "faces" not in packed.files:
            return None
        faces = np.asarray(packed["faces"], np.int64)
    if faces.ndim != 2 or faces.shape[1] not in (3, 4):
        raise ValueError(f"{identity}: faces must be [F,3] or [F,4] indices into the field points")
    if faces.shape[1] == 4:
        faces = np.concatenate([faces[:, [0, 1, 2]], faces[:, [0, 2, 3]]])
    return faces


def graph_edges(points: np.ndarray, faces: np.ndarray | None, neighbours: int) -> np.ndarray:
    """Undirected edges [2, E], each pair once per direction, ordered as GABI's car dataset orders mesh edges: the
    mesh's edges, or each point joined to its k nearest neighbours (in the mesh's own coordinates)."""
    if faces is not None:
        if faces.min() < 0 or faces.max() >= len(points):
            raise ValueError("face indices outside the field points")
        pairs = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    else:
        _, nearest = cKDTree(points).query(points, k=neighbours + 1, workers=-1)
        pairs = np.stack([np.repeat(np.arange(len(points)), neighbours), nearest[:, 1:].reshape(-1)], axis=1)
    pairs = np.unique(np.sort(pairs, axis=1), axis=0)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    return np.ascontiguousarray(np.concatenate([pairs, pairs[:, ::-1]]).T)


def graph(points: np.ndarray, faces: np.ndarray | None, neighbours: int, values: np.ndarray | None = None,
          pos: np.ndarray | None = None):
    """A car as GABI's car dataset holds it. The graph comes from `points`; node positions are `pos` (default the
    points). Without values, x is zero: decoding never reads it."""
    from torch_geometric.data import Data

    n = len(points)
    return Data(pos=torch.as_tensor(points if pos is None else pos, dtype=torch.float32),
                face=torch.as_tensor(np.zeros((3, 0), np.int64) if faces is None else faces.T.copy()),
                edge_index=torch.as_tensor(graph_edges(points, faces, neighbours)),
                x=torch.as_tensor(np.zeros(n) if values is None else values, dtype=torch.float32)[:, None])


@torch.no_grad()
def abc_pool(model, data, problems, z_generator, noise_generators, samples: int, accept: int, sigma: float,
             batch_size: int, device) -> list[torch.Tensor]:
    """GABI's quantile ABC for several observation sets of one shape, from one pool of prior samples.
    Each batch: z ~ N(0, I) from z_generator, decode, then for each set add sigma * N(0, 1) noise (from that set's
    generator) to the decoded values at its observed nodes and record the L2 distance to its observations. Each set
    keeps the `accept` nearest samples and returns their mean field. problems: (observed node indices, observed values
    in GABI's normalised units), or None for nothing visible, which returns the mean over all prior samples.
    With one set and one generator for both draws, the draws are abc_inference's, in its order."""
    from torch_geometric.data import Batch

    n = data.num_nodes
    batches = {}

    def decode(z):
        if len(z) not in batches:
            batches[len(z)] = Batch.from_data_list([data.clone() for _ in range(len(z))]).to(device)
        return model.decode(z, batches[len(z)]).view(len(z), n)

    latents = torch.empty((samples, model.z_dim), device=device)
    distances = [None if p is None else torch.empty(samples, device=device) for p in problems]
    total = torch.zeros(n, dtype=torch.float64, device=device)
    for start in range(0, samples, batch_size):
        size = min(batch_size, samples - start)
        z = torch.randn((size, model.z_dim), generator=z_generator, device=device)
        latents[start:start + size] = z
        decoded = decode(z)
        total += decoded.sum(0, dtype=torch.float64)
        for problem, generator, distance in zip(problems, noise_generators, distances):
            if problem is None:
                continue
            index, y = problem
            predicted = decoded[:, index]
            noisy = predicted + sigma * torch.randn(predicted.shape, generator=generator, device=device)
            distance[start:start + size] = torch.linalg.norm(noisy - y, dim=1)
    means = []
    for distance in distances:
        if distance is None:
            means.append(total / samples)
            continue
        kept = torch.sort(distance).indices[:accept]
        accepted = torch.zeros(n, dtype=torch.float64, device=device)
        for start in range(0, accept, batch_size):
            accepted += decode(latents[kept[start:start + batch_size]]).sum(0, dtype=torch.float64)
        means.append(accepted / accept)
    return means


# ---------------------------------------------------------------------------------------------------------------
# export and train


def cmd_export(args) -> None:
    config = fb.bench_config(Path(args.bench))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sizes, kinds, nodes = {}, set(), []
    for split in ("train", "val"):
        records = fb.records_of(config, split)
        if not records:
            raise SystemExit(f"no {split} cars: GABI trains on train and keeps the checkpoint best on val")
        graphs = []
        for record in records:
            field = fb.load_field(config["fields"], record["id"])
            faces = read_faces(config["fields"], record["id"])
            kinds.add("mesh" if faces is not None else f"{args.neighbours}-nearest-neighbour")
            graphs.append(graph(field.points, faces, args.neighbours, field.values))
            nodes.append(len(field.points))
        with open(out / f"{split}.pkl", "wb") as handle:
            pickle.dump(graphs, handle)
        sizes[split] = len(graphs)
    if len(kinds) > 1:
        raise SystemExit("some field files have faces and some do not: use one kind of graph for every car")
    steps = math.ceil(sizes["train"] / args.batch_size)
    epochs = max(1, math.ceil(args.iterations / steps))
    settings = dict(
        run_settings=dict(validate=True, num_t_workers=args.workers, num_v_workers=args.workers),
        io_settings=dict(train_dataset_path=str((out / "train.pkl").resolve()),
                         valid_dataset_path=str((out / "val.pkl").resolve()), run_dir=str((out / "runs").resolve()),
                         save_epochs=max(1, epochs // 10), pretrained_model=""),
        hyperparameters=dict(batch_size=args.batch_size, epochs=epochs, start_lr=args.lr,
                             lr_decay=float((args.final_lr / args.lr) ** (1.0 / epochs)),
                             weight_decay=args.weight_decay),
        data_settings=dict(dataset_type="HelmholtzCar", transform=""),
        model_settings=dict(model_type="GCN", n_layers=args.layers, latent_dim=args.width, z_dim=args.z_dim,
                            use_boundary_encoding=False, use_pos=True))
    import yaml

    (out / "config.yml").write_text(yaml.safe_dump(settings, sort_keys=False))
    fb.save_json(out / "export.json", dict(
        bench=str(Path(args.bench).resolve()), graph=kinds.pop(), neighbours=args.neighbours, cars=sizes,
        nodes_per_car=dict(mean=float(np.mean(nodes)), max=int(np.max(nodes))), iterations=epochs * steps,
        epochs=epochs, note="GABI's training script decays the learning rate exponentially, once per epoch; the paper "
                            "used a cosine decay over the same range", created=fb.now()))
    print(f"exported {sizes} cars, {np.mean(nodes):.0f} nodes per car (largest {np.max(nodes)}); "
          f"{epochs} epochs x {steps} batches of {args.batch_size} = {epochs * steps} iterations -> "
          f"{out / 'config.yml'}\nif training runs out of GPU memory, export again with a smaller --batch-size "
          "(the number of iterations is kept)")


def cmd_train(args) -> None:
    out = Path(args.out).resolve()
    runs = out / "runs"
    before = set(runs.glob("*")) if runs.exists() else set()
    environment = dict(os.environ)
    if args.device == "cpu":
        environment["CUDA_VISIBLE_DEVICES"] = ""
    repo = Path(args.gabi_repo).resolve()
    finished = subprocess.run([sys.executable, str(repo / "training_scripts" / "train_single_gpu.py"),
                               "--config", str(out / "config.yml")], cwd=repo, env=environment)
    if finished.returncode:
        raise SystemExit(f"GABI training failed ({finished.returncode})")
    new = sorted(set(runs.glob("*")) - before, key=lambda p: p.stat().st_mtime)
    if not new:
        raise SystemExit(f"no new run under {runs}")
    run = new[-1]
    best, selected = run / "trained_models" / "best.pt", "lowest validation loss"
    if not best.exists():  # GABI writes best.pt only when validation improves on the first epoch
        saved = sorted(run.glob("trained_models/e*.pt"), key=lambda p: int(p.stem[1:]))
        if not saved:
            raise SystemExit(f"{run} has no checkpoint")
        best, selected = saved[-1], "last periodic checkpoint (validation never improved on the first epoch)"
    fb.save_json(out / "trained.json", dict(run=str(run), checkpoint=str(best), selected=selected,
                                            config=str(run / "config.yml"), created=fb.now()))
    print(f"GABI checkpoint: {best} ({selected})")


# ---------------------------------------------------------------------------------------------------------------
# inference


class Gabi(fb.Method):
    def __init__(self, name: str, args, device: torch.device):
        from box import Box
        import yaml

        self.name, self.args, self.device = name, args, device
        out, repo = Path(args.out), Path(args.gabi_repo)
        exported = json.loads((out / "export.json").read_text())
        trained = json.loads((out / "trained.json").read_text())
        self.neighbours, self.graph_kind = exported["neighbours"], exported["graph"]
        run_config = Box.from_yaml(filename=trained["config"], Loader=yaml.FullLoader)
        if run_config.model_settings.model_type != "GCN":
            raise SystemExit("this adapter runs GABI's GCN model, the paper's car setting")
        gcn = load_module(repo / "models" / "gae_gcn.py", "gabi_gae_gcn")
        self.model = gcn.GCNGeomAutoencoder(**run_config.data_dims, **run_config.hyperparameters,
                                            **run_config.model_settings)
        state = torch.load(trained["checkpoint"], map_location="cpu", weights_only=False)
        self.model.load_state_dict(state["model_state_dict"])
        self.model.to(device).eval()
        stats = state["trainset_stats"]
        self.x_mean = float(stats["x"]["mean"].reshape(-1)[0])
        self.x_std = float(stats["x"]["std"].reshape(-1)[0])
        self.pos_mean = stats["pos"]["mean"].double().cpu().numpy()
        self.pos_std = stats["pos"]["std"].double().cpu().numpy()
        self.abc_inference = load_module(repo / "sampling" / "abc_inference.py", "gabi_abc").abc_inference
        commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True)
        self.source = dict(code="github.com/gduthe/GABI", commit=commit.stdout.strip() or None,
                           checkpoint=trained["checkpoint"], selected=trained.get("selected"))

    def supports(self, setting) -> bool:
        return True  # nothing visible: the mean of the prior samples, GABI's shape-only prediction

    def describe(self) -> dict:
        a = self.args
        return dict(name=self.name, source=self.source, graph=self.graph_kind, abc=a.abc, prior_samples=a.samples,
                    accepted=a.accept, noise_sigma_normalised=a.sigma, decode_batch=a.decode_batch or "auto",
                    estimate="mean of the accepted samples; nothing visible: mean of the prior samples",
                    timing="prior samples decoded once per car and shared by its observation sets"
                    if a.abc == "shared" else "abc_inference once per observation set")

    def _graph(self, surface):
        """The car as GABI sees it: its graph (built as in export) and positions normalised with the training
        statistics. No field values; mesh faces, when the field files have them, are read from the original
        field directory (they are geometry)."""
        faces = read_faces(self.args.fields, surface.id)
        return graph(surface.points, faces, self.neighbours, pos=(surface.points - self.pos_mean) / self.pos_std)

    def _seed(self, car: str, what: str) -> int:
        return stable_seed(self.args.seed, f"gabi|{car}|{what}", 0) % (2 ** 62)

    def _generator(self, car: str, what: str) -> torch.Generator:
        return torch.Generator(device=self.device).manual_seed(self._seed(car, what))

    def predict_car(self, surface, observed):
        a, device = self.args, self.device
        data = self._graph(surface)
        batch = a.decode_batch or max(1, min(500, NODES_PER_DECODE // data.num_nodes))
        problems = []
        for o in observed:
            if len(o.index):
                y = torch.as_tensor((o.values - self.x_mean) / self.x_std, dtype=torch.float32, device=device)
                problems.append((torch.as_tensor(o.index, device=device), y))
            else:
                problems.append(None)
        if a.abc == "shared":  # each set's noise comes from its own generator: its result does not depend on the rest
            means = abc_pool(self.model, data, problems, self._generator(surface.id, "prior"),
                             [self._generator(surface.id, o.key) for o in observed], a.samples, a.accept, a.sigma,
                             batch, device)
        else:
            means = [self._abc_exact(data, p, self._seed(surface.id, o.key), batch) if p is not None else
                     abc_pool(self.model, data, [None], self._generator(surface.id, "prior"), [None], a.samples,
                              a.accept, a.sigma, batch, device)[0] for p, o in zip(problems, observed)]
        return [m.double().cpu().numpy() * self.x_std + self.x_mean for m in means]

    def _abc_exact(self, data, problem, seed: int, batch: int) -> torch.Tensor:
        a = self.args
        index, y = problem
        observed = data.clone()
        values = torch.full((data.num_nodes, 1), float("nan"))
        values[index.cpu(), 0] = y.cpu()
        observed.x, observed.y = values, values.clone()  # abc_inference reads the observations at non-NaN nodes
        observed.boundary_encoding = torch.zeros((data.num_nodes, 0))
        torch.manual_seed(seed)
        result = self.abc_inference(self.model, observed.to(self.device), n_total_samples=a.samples,
                                    batch_size=batch, n_accepted_samples=a.accept, sigma_tc=a.sigma)
        return result["u_posterior_samples"].reshape(a.accept, data.num_nodes).double().mean(0)


def with_gabi(args):
    """Run field_benchmark's predict or check with the method `gabi` resolved to this adapter."""
    args.fields = fb.bench_config(Path(args.bench))["fields"]  # for mesh faces only, never field values
    original = fb.make_method

    def make(method_args, config, device):
        if method_args.method != "gabi":
            return original(method_args, config, device)
        return Gabi(args.name, args, device)

    fb.make_method = make
    common = dict(bench=args.bench, split=args.split, final=args.final, method="gabi", model=None, name=args.name,
                  batch_size=1, device=args.device, amp="none")
    try:
        if args.command == "predict":
            fb.cmd_predict(Namespace(**common))
        else:
            fb.cmd_check(Namespace(**common, cars=args.cars, per_car=args.per_car, tolerance=args.tolerance))
    finally:
        fb.make_method = original


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("export", help="the benchmark's train and val cars as GABI datasets, with a config")
    p.add_argument("--bench", required=True)
    p.add_argument("--out", required=True, help="GABI working folder, e.g. bench/gabi")
    p.add_argument("--neighbours", type=int, default=8, help="graph degree when the field files have no faces")
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--width", type=int, default=100)
    p.add_argument("--z-dim", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--iterations", type=int, default=10000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--final-lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=4)

    p = sub.add_parser("train", help="GABI's training script on the exported config")
    p.add_argument("--out", required=True)
    p.add_argument("--gabi-repo", required=True, help="a clone of github.com/gduthe/GABI")
    p.add_argument("--device", default="auto", help="cpu hides the GPUs from the training script")

    for command in ("predict", "check"):
        p = sub.add_parser(command, help="GABI-ABC on a split's observation sets" if command == "predict"
                           else "the benchmark's leak check on GABI")
        p.add_argument("--bench", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--gabi-repo", required=True)
        p.add_argument("--split", choices=("train", "val", "test"), default="val")
        p.add_argument("--final", action="store_true", help="required for the test split")
        p.add_argument("--name", default="gabi")
        p.add_argument("--abc", choices=("shared", "exact"), default="shared")
        p.add_argument("--samples", type=int, default=50000 if command == "predict" else 2000,
                       help="prior samples per car (paper: 50k)")
        p.add_argument("--accept", type=int, default=100 if command == "predict" else 20,
                       help="samples kept (paper: 100)")
        p.add_argument("--sigma", type=float, default=0.01,
                       help="observation noise in GABI's normalised units (abc_inference's default)")
        p.add_argument("--decode-batch", type=int, default=0,
                       help=f"prior samples decoded together; 0: up to 500, at most {NODES_PER_DECODE} nodes")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--device", default="auto" if command == "predict" else "cpu")
        if command == "check":
            p.add_argument("--cars", type=int, default=2)
            p.add_argument("--per-car", type=int, default=3)
            p.add_argument("--tolerance", type=float, default=1e-3)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "export":
        cmd_export(args)
    elif args.command == "train":
        cmd_train(args)
    else:
        if args.samples < args.accept:
            raise SystemExit("--samples must be at least --accept")
        with_gabi(args)


if __name__ == "__main__":
    main()
