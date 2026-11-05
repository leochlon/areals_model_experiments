"""Released Utonia/Point-MAE features at the existing 1,024 geometry anchors."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
OLD = Path("/workspace/home/areal_objaverse_060_61822_15ep_20260928T121633Z")
UTONIA = OLD / "field_utonia_compare_420"
POINTMAE = OLD / "field_pointmae_released_420"
sys.path.insert(0, str(UTONIA / "utonia_source"))
from extend_anchors import add_anchor_features


class GeometryInputs(Dataset):
    def __init__(self, rows, source):
        self.rows, self.source = rows, source

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with np.load(ROOT / "data" / (row["id"] + ".npz"), allow_pickle=False) as archive:
            p = {name: archive[name] for name in ("context_pos", "context_feat")}
        add_anchor_features(p, 1024)
        selected = np.argsort(p["context_feat"][:, 6], kind="stable")[-1024:]
        coords = p["context_pos"][selected]
        weights = p["context_feat"][selected, 7]
        with np.load(UTONIA / "cache" / (row["id"] + ".npz"), allow_pickle=False) as archive:
            points = archive["points"].astype(np.float32)
            normals = archive["normals"].astype(np.float32)
            original = archive["anchor_index"].astype(np.int64)
        order = np.concatenate([original, np.setdiff1d(np.arange(len(points)), original)])
        result = dict(id=row["id"], split=row["split"], coords=coords, weights=weights,
                      anchor_index=order[selected], points=points, normals=normals)
        if self.source == "pointmae":
            seed = int.from_bytes(hashlib.sha256((row["id"] + ":pointmae:0").encode()).digest()[:8], "little")
            rng = np.random.default_rng(seed)
            sampled = points[rng.choice(len(points), 1024, replace=False)]
            center = sampled.mean(0)
            radius = np.sqrt(((sampled-center)**2).sum(-1)).max()
            result["points"] = (sampled-center)/radius
            result["encoder_anchors"] = (coords-center)/radius
            del result["normals"]
            del result["anchor_index"]
        return result


def save_features(out, name, split, coords, weights, features):
    array = np.concatenate([np.asarray(coords, np.float32), np.asarray(weights, np.float32)[:, None],
                            np.asarray(features, np.float32)], axis=-1)
    path = out / (name + ".npy")
    np.save(path, array, allow_pickle=False)
    return dict(id=name, split=split, shape=list(array.shape))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=("pointmae", "utonia"), required=True)
    args = parser.parse_args()
    out = ROOT / "backbone_tokens" / args.source
    out.mkdir(parents=True, exist_ok=True)
    if (out / "features_ready.json").exists():
        print("Using completed feature cache", out, flush=True)
        return
    torch.set_num_threads(2)
    torch.manual_seed(0)
    started = time.time()
    rows = json.loads((ROOT / "data_ready.json").read_text())["records"]
    inputs = GeometryInputs(rows, args.source)
    written = []
    if args.source == "pointmae":
        spec = importlib.util.spec_from_file_location("released_pointmae", POINTMAE / "extract_features.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model, grouping, metadata = module.author_encoder()
        for batch in DataLoader(inputs, batch_size=8, num_workers=4, shuffle=False):
            points, anchors = batch["points"].cuda(), batch["encoder_anchors"].cuda()
            neighbors, centers = grouping(points)
            tokens, mask = model(neighbors, centers, noaug=True)
            squared = ((anchors[:, :, None]-centers[:, None])**2).sum(-1)
            distances, nearest = squared.topk(3, dim=-1, largest=False, sorted=True)
            weights = distances.clamp_min(1e-10).reciprocal()
            weights = weights/weights.sum(-1, keepdim=True)
            index = torch.arange(len(points), device=points.device)[:, None, None]
            features = (tokens[index, nearest]*weights[..., None]).sum(-2).float().cpu().numpy()
            for i, name in enumerate(batch["id"]):
                written.append(save_features(out, name, batch["split"][i], batch["coords"][i],
                                             batch["weights"][i], features[i]))
            if len(written) % 64 == 0 or len(written) == len(rows):
                print(f"pointmae features {len(written)}/{len(rows)}", flush=True)
        metadata.update(native_groups=64, input_points=1024, group_size=32,
                        coordinates_only=True, anchor_mapping="3NN inverse squared distance")
    else:
        import utonia
        from areal_tokens.data import stable_seed
        checkpoint = UTONIA / "weights/utonia.pth"
        model = utonia.load(str(checkpoint), custom_config=dict(enable_flash=False)).cuda().eval().requires_grad_(False)
        transform = utonia.transform.default(scale=1.0, apply_z_positive=True, normalize_coord=True)
        metadata = {"anchor_mapping": "Restore all five hierarchy levels, inverse map, exact anchor selection",
                    "inputs": "coordinates, normals, zero color"}
        for i, sample in enumerate(inputs):
            seed = stable_seed(17, sample["id"] + ":utonia", 0) % (2**32)
            utonia.utils.set_seed(seed)
            point = transform(dict(coord=sample["points"].copy(), normal=sample["normals"].copy(),
                                   color=np.zeros_like(sample["points"])))
            point = {key: value.cuda() if isinstance(value, torch.Tensor) else value for key, value in point.items()}
            point = model(point)
            for _ in range(4):
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            dense = point.feat[point.inverse]
            features = dense[torch.from_numpy(sample["anchor_index"]).cuda()].float().cpu().numpy()
            written.append(save_features(out, sample["id"], sample["split"], sample["coords"],
                                         sample["weights"], features))
            if (i+1) % 64 == 0 or i+1 == len(rows):
                print(f"utonia features {i+1}/{len(rows)}", flush=True)
    report = dict(source=args.source, records=written, geometry_tokens=1024,
                  seconds=time.time()-started, **metadata)
    (out / "features_ready.json").write_text(json.dumps(report, indent=2)+"\n")
    print(f"FEATURES READY source={args.source} cases={len(written)} seconds={report['seconds']:.1f}", flush=True)


if __name__ == "__main__":
    main()
