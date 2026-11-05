"""The symmetry-map prior (model.symmetry_prior = "maps")."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import trimesh
from scipy.spatial import cKDTree

from areal_tokens.config import ExperimentConfig
from areal_tokens.data import GeometryDataset, collate_clouds, read_manifest
from areal_tokens.model import GeometryAutoencoder, LatentGeometry
from areal_tokens.prepare import make_demo
from areal_tokens.symmetry import symmetry_candidates

SMALL = {"width": 48, "heads": 2, "latents": 16, "encoder_depth": 1, "decoder_depth": 1, "ff_multiplier": 2,
         "fourier_bands": 3, "query_chunk": 64}


def config(**model):
    return ExperimentConfig.from_dict(
        {
            "model": dict(SMALL, normal_head="projector", normal_prior="surfels", distance_prior="surfels",
                          surfels_per_token=4, **model),
            "data": {"encoder_points": 256, "queries": 64, "mask_fraction": 0.6, "mask_patches": 8},
            "train": {"epochs": 1, "batch_size": 2, "workers": 0, "amp": "none", "normal_loss": "unoriented"},
        }
    )


def test_symmetry_prior_validates():
    with pytest.raises(ValueError, match="none or maps"):
        config(symmetry_prior="rigid")
    with pytest.raises(ValueError, match="alternatives"):
        config(symmetry_prior="maps", mirror_prior="surfels")
    with pytest.raises(ValueError, match="distance_prior surfels"):
        ExperimentConfig.from_dict({"model": {"normal_head": "projector", "normal_prior": "surfels",
                                              "symmetry_prior": "maps"}})


def _rotated_copy(seed=0):
    """Two copies of an asymmetric part, the second rotated 70 degrees and moved; the far end of copy B hidden."""
    def part():
        e = trimesh.creation.icosphere(4)
        e.apply_scale([1.0, 0.6, 0.35])
        b = trimesh.creation.icosphere(3, radius=0.25)
        b.apply_translation([0.55, 0.35, 0.15])
        return trimesh.util.concatenate([e, b])
    a, b = part(), part()
    b.apply_transform(trimesh.transformations.rotation_matrix(np.radians(70), [0.3, 0.2, 1.0]))
    b.apply_translation([2.6, 0.8, 0.3])
    mesh = trimesh.util.concatenate([a, b])
    points, face = trimesh.sample.sample_surface(mesh, 16384, seed=seed)
    normals = mesh.face_normals[face]
    points = np.asarray(points)
    points = (points - points.mean(0)) / np.sqrt(((points - points.mean(0)) ** 2).sum(1).mean())
    hidden = np.linalg.norm(points - points[np.argmax(points[:, 0])], axis=1) < 0.35
    return points, np.asarray(normals), hidden


def test_carried_readout_recovers_a_hidden_repeated_part():
    points, normals, hidden = _rotated_copy()
    rng = np.random.default_rng(0)
    visible = rng.choice(np.flatnonzero(~hidden), 3000, replace=False)
    surfels = visible[rng.choice(3000, 2048, replace=False)]
    rotation, translation, support, valid, local = symmetry_candidates(
        points[visible], normals[visible], points[surfels], normals[surfels], 8, rng)
    assert valid.sum() >= 1 and support[0] > 0.3  # the copy map (or its inverse) is found and holds on ~half
    cfg = config(symmetry_prior="maps", symmetry_maps=8).model
    model = GeometryAutoencoder(cfg).eval()
    t = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32)[None]
    geometry = LatentGeometry(
        tokens=torch.zeros(1, 16, 48), positions=t(points[surfels[:16]]), normals=t(normals[surfels[:16]]),
        weights=torch.ones(1, 16), center=torch.zeros(1, 3), scale=torch.ones(1, 1), area=torch.ones(1, 1),
        surfel_positions=t(points[surfels]), surfel_normals=t(normals[surfels]), surfel_weights=torch.ones(1, 2048),
        symmetry_rotation=t(rotation), symmetry_translation=t(translation), symmetry_support=t(support),
        symmetry_valid=t(valid), symmetry_local=t(local))
    feet = rng.choice(np.flatnonzero(hidden), 300, replace=False)
    queries = t(points[feet])
    with torch.no_grad():
        distance, tensor, logits = model._symmetry(geometry, queries, torch.zeros(1, 300, 48), torch.ones(1, 300))
    assert distance.shape == (1, 300, 8) and tensor.shape == (1, 300, 8, 3, 3) and logits.shape == (1, 300, 8, 2)
    assert torch.all(logits[0, :, valid == 0] < -1e3)  # padded maps never blend
    # the best map per query (nearest carried surface) reproduces the hidden normal
    best = distance[0].masked_fill(torch.as_tensor(valid == 0)[None], torch.inf).argmin(-1)
    normal = torch.linalg.eigh(tensor[0, torch.arange(300), best])[1][..., -1].numpy()
    error = np.degrees(np.arccos(np.clip(np.abs((normal * normals[feet]).sum(1)), 0, 1)))
    near = distance[0, torch.arange(300), best].numpy() < 0.02
    assert near.mean() > 0.8
    assert np.median(error[near]) < 6.0


def test_forward_backward_and_initial_weight(tmp_path):
    demo = make_demo(tmp_path, train=2, val=1, test=1, points=512, queries=512)
    cfg = config(symmetry_prior="maps", symmetry_maps=4)
    records, _ = read_manifest(demo)
    data = GeometryDataset([r for r in records if r["split"] == "train"], cfg.model, cfg.data, 0, training=True)
    batch = collate_clouds([data[0], data[1]])
    assert batch["symmetry_rotation"].shape == (2, 4, 3, 3) and batch["symmetry_local"].shape[:2] == (2, 4)
    model = GeometryAutoencoder(cfg.model)
    out = model(batch)
    assert torch.isfinite(out["normal_tensor"]).all() and torch.isfinite(out["normalized_udf"]).all()
    assert float(out["symmetry_weight"].max()) < 4 * np.exp(-4) + 1e-6
    loss = out["normalized_udf"].mean() + out["normal_tensor"].square().mean()
    loss.backward()
    assert model.symmetry_gate.bias.grad is not None


def test_packets_export_and_reload_their_symmetry_maps(tmp_path):
    from areal_tokens.evaluate import encode_checkpoint
    from areal_tokens.io import load_latents
    from areal_tokens.train import train

    demo = make_demo(tmp_path / "demo", train=2, val=1, test=2, points=512, queries=512)
    cfg = config(symmetry_prior="maps", symmetry_maps=4)
    result = train(demo, cfg, tmp_path / "run", device="cpu")
    encode_checkpoint(result["checkpoint"], demo, tmp_path / "packets", "cpu", split="test")
    files = sorted((tmp_path / "packets").glob("shape_*.npz"))
    assert files
    for path in files:
        packet = load_latents(path)
        assert packet.symmetry_rotation.shape == (1, 4, 3, 3)
        assert packet.symmetry_local.shape == (1, 4, packet.surfel_positions.shape[1])
        model = GeometryAutoencoder(cfg.model).eval()
        with torch.no_grad():
            out = model.decode(packet, packet.world_positions[:, :5])
        assert torch.isfinite(out["udf"]).all() and "symmetry_weight" in out
    with np.load(files[0]) as z:
        arrays = {k: z[k] for k in z.files if k != "symmetry_local"}
    np.savez(tmp_path / "broken.npz", **arrays)
    with pytest.raises(ValueError, match="must come together"):
        load_latents(tmp_path / "broken.npz")
