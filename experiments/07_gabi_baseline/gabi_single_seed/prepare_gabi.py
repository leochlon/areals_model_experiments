import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch, Data

import gabi_adapter as ga
import field_benchmark as fb

ROOT = Path(__file__).resolve().parent
REPO = ROOT / "GABI"
OUT = ROOT / "bench/gabi"


def abc_check():
    """The shared-pool ABC matches the authors' abc_inference on a small synthetic graph."""
    gcn = ga.load_module(REPO / "models/gae_gcn.py", "check_gcn")
    abc = ga.load_module(REPO / "sampling/abc_inference.py", "check_abc").abc_inference
    torch.manual_seed(7)
    model = gcn.GCNGeomAutoencoder(node_feature_dim=1, edge_feature_dim=0, latent_dim=100,
        z_dim=100, n_layers=6, use_boundary_encoding=False, be_dim=0, use_pos=True, pos_dim=3).eval()
    points = np.random.default_rng(8).normal(size=(24, 3)).astype(np.float32)
    data = ga.graph(points, None, 8)
    data.boundary_encoding = torch.zeros((len(points), 0))
    index = torch.tensor([1, 4, 9, 18])
    y = torch.tensor([.2, -.3, .4, .1])
    observed = data.clone()
    observed.x[:] = float("nan")
    observed.x[index, 0] = y
    observed.y = observed.x.clone()
    torch.manual_seed(91)
    exact = abc(model, observed, n_total_samples=100, batch_size=10,
        n_accepted_samples=10, sigma_tc=.01)["u_posterior_samples"].double().mean(0)
    generator = torch.Generator().manual_seed(91)
    shared = ga.abc_pool(model, data, [(index, y)], generator, [generator], 100, 10, .01, 10, "cpu")[0]
    parity = float((exact - shared).abs().max())
    assert parity < 1e-6, parity
    first = ga.abc_pool(model, data, [(index, y)], torch.Generator().manual_seed(33),
        [torch.Generator().manual_seed(44)], 100, 10, .01, 10, "cpu")[0]
    changed = data.clone()
    changed.x = torch.randn_like(changed.x) * 1000
    changed.y = torch.randn_like(changed.x) * 1000
    second = ga.abc_pool(model, changed, [(index, y)], torch.Generator().manual_seed(33),
        [torch.Generator().manual_seed(44)], 100, 10, .01, 10, "cpu")[0]
    assert torch.equal(first, second)
    joint = ga.abc_pool(model, data, [(index, y), (index, -y)], torch.Generator().manual_seed(33),
        [torch.Generator().manual_seed(44), torch.Generator().manual_seed(55)], 100, 10, .01, 10, "cpu")
    assert torch.equal(first, joint[0])
    result = {"status": "PASS", "synthetic_author_abc_max_difference": parity,
        "synthetic_hidden_values_max_difference": float((first-second).abs().max()),
        "shared_pool_other_observations_max_difference": float((first-joint[0]).abs().max()),
        "torch": torch.__version__, "cuda": torch.cuda.is_available()}
    (ROOT / "abc_check.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def export(out=OUT, iterations=10000):
    original = ga.graph
    templates = {}

    def cached(points, faces, neighbours, values=None, pos=None):
        assert faces is None and pos is None
        key = hashlib.sha256(np.ascontiguousarray(points).tobytes()).hexdigest()
        if key not in templates:
            templates[key] = original(points, faces, neighbours)
            if len(templates) % 20 == 0:
                print(f"built {len(templates)} distinct geometry graphs", flush=True)
        template = templates[key]
        return Data(pos=template.pos, face=template.face, edge_index=template.edge_index,
            x=torch.as_tensor(np.zeros(len(points)) if values is None else values, dtype=torch.float32)[:, None])

    ga.graph = cached
    args = ga.build_parser().parse_args(["export", "--bench", str(ROOT / "bench"), "--out", str(out),
        "--iterations", str(iterations)])
    try:
        ga.cmd_export(args)
    finally:
        ga.graph = original
    # Byte-identical geometries share one graph; the pressure values stay per case.
    (Path(out) / "export_cache.json").write_text(json.dumps({"distinct_graphs": len(templates)}, indent=2) + "\n")


def probe():
    from box import Box
    import yaml
    sys.path.insert(0, str(REPO))
    from datatools import HelmholtzCarDataset
    from training_scripts.train_single_gpu import compute_loss
    from models import GCNGeomAutoencoder
    config = Box.from_yaml(filename=str(OUT / "config.yml"), Loader=yaml.FullLoader)
    dataset = HelmholtzCarDataset(str(OUT / "train.pkl"))
    examples = [dataset.get(i) for i in range(100)]
    x = torch.cat([d.x for d in examples])
    pos = torch.cat([d.pos for d in examples])
    xm, xs, pm, ps = x.mean(0), x.std(0), pos.mean(0), pos.std(0)
    del x, pos
    for d in examples:
        d.x = (d.x-xm)/xs
        d.y = (d.y-xm)/xs
        d.pos = (d.pos-pm)/ps
    results = []
    selected = None
    for size in [100, 50, 25, 10]:
        model = optimizer = batch = loss = None
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print(f"probing full {examples[0].num_nodes}-node training graphs, batch {size}", flush=True)
        try:
            model = GCNGeomAutoencoder(**dataset.get_data_dims_dict(), **config.hyperparameters,
                **config.model_settings).cuda()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            batch = Batch.from_data_list(examples[:size]).cuda()
            torch.cuda.synchronize()
            start = time.perf_counter()
            loss, _, _ = compute_loss(model(batch))
            assert torch.isfinite(loss)
            loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            optimizer.step()
            torch.cuda.synchronize()
            seconds = time.perf_counter()-start
            peak = torch.cuda.max_memory_allocated()/2**30
            result = {"batch_size": size, "status": "PASS", "step_seconds": seconds,
                "peak_gpu_gib": peak, "loss": float(loss.detach())}
            results.append(result)
            print(json.dumps(result), flush=True)
            selected = size
            break
        except torch.cuda.OutOfMemoryError as exc:
            results.append({"batch_size": size, "status": "OOM", "message": str(exc)[:400]})
            print(json.dumps(results[-1]), flush=True)
        finally:
            del model, optimizer, batch, loss
            gc.collect()
            torch.cuda.empty_cache()
    assert selected is not None, results
    if selected != config.hyperparameters.batch_size:
        (OUT / "config_requested.yml").write_text((OUT / "config.yml").read_text())
        config.hyperparameters.batch_size = selected
        batches = int(np.ceil(len(dataset)/selected))
        epochs = int(np.ceil(10000/batches))
        config.hyperparameters.epochs = epochs
        config.hyperparameters.lr_decay = float((1e-5/1e-3)**(1/epochs))
        config.io_settings.save_epochs = max(1, epochs//10)
        config.to_yaml(filename=str(OUT / "config.yml"))
        exported = json.loads((OUT / "export.json").read_text())
        exported.update(epochs=epochs, iterations=epochs*batches,
            batch_size=selected, requested_batch_size=100, reason="full-resolution GPU memory probe")
        fb.save_json(OUT / "export.json", exported)
    fb.save_json(ROOT / "memory_probe.json", {"results": results, "selected_batch": selected,
        "nodes_per_graph": examples[0].num_nodes})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["abc_check", "export", "probe"])
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--iterations", type=int, default=10000)
    args = parser.parse_args()
    if args.command == "export":
        export(args.out, args.iterations)
    else:
        globals()[args.command]()
