"""nuisance audit and test-time orbit averaging for an Areal geometry checkpoint.

For each mesh in a split and each mask level, the visible region and the query set are
held fixed and one nuisance is drawn R times:
  resample  new area-weighted point sample, hence new anchors and surfels
  rotation  uniform SO(3) rotation of input and queries; predicted normals rotated back
  sign      each input normal negated with probability 0.5 (targets are unsigned)
Reported per nuisance, paired over meshes with a bootstrap:
  base       error of the standard view (reproduces `areal evaluate` exactly)
  view       mean error of one nuisance view (cost of leaving the training orbit)
  orbit      error of the orbit average (distance: mean; normal: top eigenvector of mean n n^T)
  gap        dispersion across views

Usage:
  python author_nuisance_audit.py --checkpoint best.pt --manifest prepared/manifest.json \
      --draws 8 --masks 0.0,0.6 --out audit.json
Requires areal_tokens importable; target_mode must be "provided".
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from areal_tokens.data import collate_clouds, load_cloud_file, read_manifest, stable_seed
from areal_tokens.geometry import flip_normal_signs, prepare_cloud, random_rotation, rotate_cloud, visible_mask
from areal_tokens.io import load_model


def build_view(record, root, config, mask_fraction, nuisance, draw):
    seed = config.train.seed
    cloud, targets = load_cloud_file(str(root / record["path"]))
    mask_rng = np.random.default_rng(stable_seed(seed, record["id"] + ":mask", 0))
    query_rng = np.random.default_rng(stable_seed(seed, record["id"] + ":queries", 0))
    visible = visible_mask(cloud, mask_fraction, config.data.mask_patches, mask_rng)
    indices = query_rng.choice(
        len(targets["udf"]), config.data.queries, replace=len(targets["udf"]) < config.data.queries
    )
    queries, udf, qn = (targets[k][indices] for k in ("queries", "udf", "query_normals"))
    hidden = ~visible[cKDTree(cloud.points).query(queries, workers=1)[1]]
    part, rotation, point_epoch = cloud.subset(visible), None, 0
    if draw > 0:
        rng = np.random.default_rng(stable_seed(seed, record["id"] + f":{nuisance}", draw))
        if nuisance == "resample":
            point_epoch = draw
        elif nuisance == "rotation":
            rotation = random_rotation("so3", rng)
            part = rotate_cloud(part, rotation)
        elif nuisance == "sign":
            part = flip_normal_signs(part, 0.5, rng)
        else:
            raise ValueError(nuisance)
    point_rng = np.random.default_rng(stable_seed(seed, record["id"] + ":points", point_epoch))
    surfels = config.model.latents * config.model.surfels_per_token if config.model.surfels_per_token > 1 else 0
    sample = prepare_cloud(part, config.model.latents, config.data.encoder_points, point_rng, surfels)
    sample.update(id=record["id"], family=record["family"], split=record["split"])
    sample["complete"] = np.array([float(visible.all())], dtype=np.float32)
    length = np.linalg.norm(qn, axis=-1)
    sample.update(
        queries=(queries if rotation is None else queries @ rotation.T).astype(np.float32),
        udf=udf.astype(np.float32),
        query_normals=(qn / np.maximum(length[:, None], 1e-8)).astype(np.float32),
        normal_valid=(length > 1e-8) & (udf / float(sample["scale"][0]) <= config.data.normal_band),
        query_hidden=hidden,
    )
    return sample, rotation


@torch.inference_mode()
def predict(model, samples, device):
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in collate_clouds(samples).items()}
    out = model.decode(model.encoder(batch), batch["queries"])
    return out["udf"].float().cpu().numpy(), out["normals"].float().cpu().numpy()


def angle_deg(n, target):
    return np.degrees(np.arccos(np.abs((n * target).sum(-1)).clip(0, 1)))


def metrics(udf, normals, sample):
    err = (udf - sample["udf"]) / float(sample["scale"][0])
    valid, hid = sample["normal_valid"], sample["query_hidden"]
    ang = angle_deg(normals, sample["query_normals"])
    pick = lambda m: float(ang[m].mean()) if m.any() else float("nan")  # noqa: E731
    return dict(nmse=float(np.mean(err**2)), normal_all=pick(valid), normal_hidden=pick(valid & hid))


def orbit_average(udfs, normals):
    projector = np.mean(normals[..., :, None] * normals[..., None, :], axis=0)
    return np.mean(udfs, axis=0), np.linalg.eigh(projector)[1][..., :, -1]


def summarize(rows):
    rng = np.random.default_rng(0)

    def boot(d):
        d = np.asarray(d)
        means = d[rng.integers(0, len(d), (4000, len(d)))].mean(1)
        return d.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)

    for mask in sorted({r["mask"] for r in rows}):
        print(f"=== mask {mask}")
        for nuisance in ("resample", "rotation", "sign"):
            sel = [r for r in rows if r["mask"] == mask and r["nuisance"] == nuisance]
            if not sel:
                continue
            print(f"-- {nuisance}: gap distance {np.mean([r['gap_udf_rms'] for r in sel]):.4f}, "
                  f"normal {np.mean([r['gap_normal_deg'] for r in sel]):.2f} deg")
            for key in ("nmse", "normal_all", "normal_hidden"):
                ok = [r for r in sel if np.isfinite(r["base"][key])]
                if not ok:
                    continue
                b, v, o = (np.array([r[s][key] for r in ok]) for s in ("base", "view", "orbit"))
                if key == "nmse":
                    b, v, o = np.sqrt(b), np.sqrt(v), np.sqrt(o)
                dv, do = boot(v - b), boot(o - b)
                print(f"   {key:13s} base {b.mean():.4f} | one view {dv[0]:+.4f} [{dv[1]:+.4f},{dv[2]:+.4f}]"
                      f" | orbit avg {do[0]:+.4f} [{do[1]:+.4f},{do[2]:+.4f}] better on {(o < b).sum()}/{len(ok)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--draws", type=int, default=8)
    parser.add_argument("--nuisances", default="resample,rotation,sign")
    parser.add_argument("--masks", default="0.0,0.6")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="nuisance_audit.json")
    args = parser.parse_args()
    model, config, _ = load_model(args.checkpoint, args.device)
    if config.data.target_mode != "provided":
        raise SystemExit("This audit expects target_mode='provided' (fixed reference queries).")
    records, _ = read_manifest(args.manifest)
    chosen = [r for r in records if r["split"] == args.split]
    root, rows, start = Path(args.manifest).parent, [], time.time()
    for mask in (float(m) for m in args.masks.split(",")):
        for i, record in enumerate(chosen):
            base, _ = build_view(record, root, config, mask, "resample", 0)
            u0, n0 = predict(model, [base], args.device)
            m0 = metrics(u0[0], n0[0], base)
            for nuisance in args.nuisances.split(","):
                views, rotations = zip(
                    *(build_view(record, root, config, mask, nuisance, d) for d in range(1, args.draws + 1))
                )
                udfs, normals = predict(model, list(views), args.device)
                normals = np.stack([n if r is None else n @ r for n, r in zip(normals, rotations)])
                u_bar, n_bar = orbit_average(udfs, normals)
                per_view = [metrics(u, n, base) for u, n in zip(udfs, normals)]
                rows.append(dict(
                    id=record["id"], mask=mask, nuisance=nuisance, base=m0,
                    view={k: float(np.nanmean([p[k] for p in per_view])) for k in m0},
                    orbit=metrics(u_bar, n_bar, base),
                    gap_udf_rms=float(np.sqrt(np.mean(np.var(udfs, axis=0))) / float(base["scale"][0])),
                    gap_normal_deg=float(np.mean(
                        [angle_deg(n, n_bar)[base["normal_valid"]].mean() for n in normals])),
                ))
            print(f"mask {mask}: {i + 1}/{len(chosen)} meshes, {time.time() - start:.0f}s", flush=True)
    Path(args.out).write_text(json.dumps(rows))
    summarize(rows)


if __name__ == "__main__":
    main()
