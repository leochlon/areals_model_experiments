"""The 0.3.0 options: identical defaults, and the documented behaviour when switched on."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from areal_tokens import geometry as geom
from areal_tokens.config import ExperimentConfig, TrainConfig
from areal_tokens.data import GeometryDataset, load_cloud_file, read_manifest, stable_seed
from areal_tokens.losses import reconstruction_loss, reconstruction_metrics
from areal_tokens.prepare import make_demo, save_npz
from areal_tokens.train import lr_at_step, train
from areal_tokens.validation import load_audit, validate_manifest, write_filtered_manifest

TINY = ExperimentConfig.from_dict(
    {
        "model": {"width": 48, "heads": 2, "latents": 16, "encoder_depth": 1, "decoder_depth": 1,
                  "ff_multiplier": 2, "fourier_bands": 3, "query_chunk": 64},
        "data": {"encoder_points": 128, "queries": 64, "mask_fraction": 0.6, "mask_patches": 8},
        "train": {"epochs": 2, "batch_size": 2, "accumulation": 1, "learning_rate": 0.001,
                  "warmup_epochs": 1, "workers": 0, "amp": "none"},
    }
)


@pytest.fixture(scope="module")
def demo(tmp_path_factory) -> Path:
    return make_demo(tmp_path_factory.mktemp("demo"), train=4, val=2, test=2, points=256, queries=512)


def corrupt_copy(demo: Path, tmp_path: Path, victim: str = "train_0001") -> Path:
    """The demo manifest with one record's normals zeroed, which the loader rejects."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    records, _ = read_manifest(demo)
    for record in records:
        name = Path(record["path"]).name
        if record["id"] == victim:
            with np.load(record["path"]) as packed:
                arrays = {key: packed[key] for key in packed.files}
            arrays["normals"] = np.zeros_like(arrays["normals"])
            save_npz(tmp_path / name, **arrays)
        else:
            os.symlink(record["path"], tmp_path / name)
        record["path"] = name
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema": "geometry_latents_v1", "units": "m", "records": records}))
    return manifest


# --- farthest point sampling -------------------------------------------------------------------


@pytest.mark.skipif(geom._farthest_loop_numba is None, reason="numba not installed")
def test_compiled_fps_matches_numpy_including_ties():
    rng = np.random.default_rng(0)
    for trial in range(60):
        n = int(rng.integers(2, 1500))
        if trial % 3 == 0:
            points = rng.integers(-3, 4, size=(n, 3)).astype(np.float64)  # many exact ties
        elif trial % 3 == 1:
            points = np.stack([rng.normal(size=n), np.zeros(n), rng.choice([0.0, 1.0], n)], 1)
        else:
            points = rng.normal(size=(n, 3))
        points = np.unique(points, axis=0)
        count = int(rng.integers(1, min(len(points), 300) + 1))
        weights = rng.random(len(points)) + 1e-3
        center = (points * (weights / weights.sum())[:, None]).sum(axis=0)
        first = int(np.argmax(((points - center) ** 2).sum(axis=1)))
        expected = geom._farthest_loop_numpy(points, first, count)
        got = geom._farthest_loop_numba(np.ascontiguousarray(points), first, count)
        assert np.array_equal(expected, got)


# --- augmentation ----------------------------------------------------------------------------


def test_augmentation_off_is_identity(demo):
    records, _ = read_manifest(demo)
    dataset = GeometryDataset(records[:1], TINY.model, TINY.data, 17, training=True)
    cloud, targets = load_cloud_file(records[0]["path"])
    same_cloud, same_targets = dataset._augment(records[0], cloud, targets)
    assert same_cloud is cloud and same_targets is targets


def test_rotation_augmentation_rotates_inputs_and_targets_consistently(demo):
    records, _ = read_manifest(demo)
    record = records[0]
    plain = GeometryDataset([record], TINY.model, TINY.data, 17, training=True)
    rotated = GeometryDataset(
        [record], TINY.model, replace(TINY.data, augment_rotation="so3"), 17, training=True
    )
    for epoch in (0, 3):
        plain.set_epoch(epoch), rotated.set_epoch(epoch)
        a, b = plain[0], rotated[0]
        rng = np.random.default_rng(stable_seed(17, record["id"] + ":augment", epoch))
        rotation = geom.random_rotation("so3", rng)
        assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-12) and np.linalg.det(rotation) > 0
        # distances are invariant, queries and their normals rotate with the shape
        assert np.array_equal(a["udf"], b["udf"])
        assert np.allclose(b["queries"], a["queries"] @ rotation.T.astype(np.float32), atol=1e-5)
        assert np.allclose(b["query_normals"], a["query_normals"] @ rotation.T.astype(np.float32), atol=1e-5)
        assert np.allclose(b["scale"], a["scale"], rtol=1e-5) and np.allclose(b["area"], a["area"])
        assert np.allclose(b["center"], rotation @ a["center"].astype(np.float64), atol=1e-4)
        # the encoder sees the same surface samples, rotated (order may differ after canonicalisation)
        from scipy.spatial import cKDTree

        distance, _ = cKDTree(a["points"] @ rotation.T).query(b["points"])
        assert distance.max() < 1e-4
        distance, _ = cKDTree(a["anchors"] @ rotation.T).query(b["anchors"])
        assert distance.max() < 1e-4


def test_octahedral_rotations_are_proper_and_axis_aligned():
    rng = np.random.default_rng(1)
    seen = set()
    for _ in range(300):
        rotation = geom.random_rotation("octahedral", rng)
        assert np.linalg.det(rotation) == pytest.approx(1.0)
        assert set(np.abs(rotation).ravel()) <= {0.0, 1.0}
        seen.add(rotation.tobytes())
    assert len(seen) == 24


def test_normal_sign_flips_touch_only_input_normals(demo):
    records, _ = read_manifest(demo)
    record = records[0]
    plain = GeometryDataset([record], TINY.model, TINY.data, 17, training=True)
    flipped = GeometryDataset(
        [record], TINY.model, replace(TINY.data, normal_sign_flip_probability=0.5), 17, training=True
    )
    a, b = plain[0], flipped[0]
    assert np.array_equal(a["points"], b["points"]) and np.array_equal(a["anchors"], b["anchors"])
    assert np.array_equal(a["query_normals"], b["query_normals"]) and np.array_equal(a["udf"], b["udf"])
    agreement = (a["normals"] * b["normals"]).sum(-1)
    assert set(np.round(agreement).astype(int)) == {-1, 1}
    assert 0.35 < (agreement < 0).mean() < 0.65
    # anchors carry the same (possibly flipped) normal as the point they sit on
    from scipy.spatial import cKDTree

    owner = cKDTree(b["points"]).query(b["anchors"])[1]
    assert np.array_equal(b["anchor_normals"], b["normals"][owner])


def test_validation_split_is_never_augmented(demo):
    records, _ = read_manifest(demo)
    config = replace(TINY.data, augment_rotation="so3", normal_sign_flip_probability=0.5)
    a = GeometryDataset(records[:1], TINY.model, TINY.data, 17, fixed_queries=True)[0]
    b = GeometryDataset(records[:1], TINY.model, config, 17, fixed_queries=True)[0]
    for key in ("points", "normals", "anchors", "queries", "query_normals", "udf"):
        assert np.array_equal(a[key], b[key])


# --- losses ------------------------------------------------------------------------------------


def test_unoriented_normal_loss_ignores_target_sign():
    torch.manual_seed(0)
    normals = torch.nn.functional.normalize(torch.randn(2, 50, 3), dim=-1)
    prediction = dict(normalized_udf=torch.rand(2, 50), udf=torch.rand(2, 50), normals=normals)
    target = torch.nn.functional.normalize(normals + 0.3 * torch.randn(2, 50, 3), dim=-1)
    sign = torch.where(torch.rand(2, 50, 1) < 0.5, -1.0, 1.0)
    batch = dict(udf=torch.rand(2, 50), scale=torch.ones(2, 1), normal_valid=torch.ones(2, 50, dtype=torch.bool))
    oriented, unoriented = TrainConfig(), TrainConfig(normal_loss="unoriented")
    clean = reconstruction_loss(prediction, dict(batch, query_normals=target), oriented)["normal"]
    flipped = reconstruction_loss(prediction, dict(batch, query_normals=target * sign), oriented)["normal"]
    assert flipped > clean  # the oriented loss punishes a flipped target
    for normals_target in (target, target * sign):
        loss = reconstruction_loss(prediction, dict(batch, query_normals=normals_target), unoriented)["normal"]
        assert torch.allclose(loss, clean, atol=1e-6)  # the unoriented one is blind to the sign
        angle = reconstruction_metrics(prediction, dict(batch, query_normals=normals_target), unoriented=True)
        assert angle["normal_angle_sum"].sum() / angle["normal_count"].sum() <= 90


# --- tolerant loading and auditing -------------------------------------------------------------


def test_bad_record_raises_by_default_and_is_skipped_on_request(demo, tmp_path):
    records, _ = read_manifest(corrupt_copy(demo, tmp_path))
    victim = next(i for i, record in enumerate(records) if record["id"] == "train_0001")
    strict = GeometryDataset(records, TINY.model, TINY.data, 17, training=True)
    with pytest.raises(ValueError, match="zero normal"):
        strict[victim]
    tolerant = GeometryDataset(records, TINY.model, replace(TINY.data, bad_records="skip"), 17, training=True)
    sample = tolerant[victim]
    assert sample["id"] == records[(victim + 1) % len(records)]["id"]
    assert tolerant.failed_ids == {"train_0001"}
    assert tolerant[0]["id"] == records[0]["id"]


def test_parallel_audit_equals_sequential_and_filters_bad_records(demo, tmp_path):
    sequential = validate_manifest(demo, workers=1)
    parallel = validate_manifest(demo, workers=2)
    assert sequential == parallel
    assert sequential["failures"] == []
    corrupt = corrupt_copy(demo, tmp_path / "corrupt")
    with pytest.raises(ValueError, match="train_0001"):
        validate_manifest(corrupt)
    report = validate_manifest(corrupt, workers=2, bad_records="skip")
    assert [failure["id"] for failure in report["failures"]] == ["train_0001"]
    assert "train_0001" not in {row["id"] for row in report["shapes"]}
    filtered = write_filtered_manifest(corrupt, report, tmp_path / "clean" / "manifest.json")
    clean_records, _ = read_manifest(filtered)
    assert {record["id"] for record in clean_records} == {row["id"] for row in report["shapes"]}
    clean = validate_manifest(filtered)
    assert clean["shapes"] == report["shapes"]


def test_audit_reuse_requires_the_same_manifest(demo, tmp_path):
    report = validate_manifest(demo)
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps(report))
    assert load_audit(audit, demo)["shapes"] == report["shapes"]
    other = tmp_path / "other.json"
    other.write_text(demo.read_text().replace("train_0000", "renamed"))
    with pytest.raises(ValueError, match="different manifest"):
        load_audit(audit, other)


# --- schedule and resume -----------------------------------------------------------------------


def test_warmup_steps_overrides_warmup_epochs():
    by_epoch = TrainConfig(epochs=10, warmup_epochs=2, learning_rate=1.0)
    by_step = TrainConfig(epochs=10, warmup_epochs=2, warmup_steps=3, learning_rate=1.0)
    assert lr_at_step(TrainConfig(epochs=10, warmup_epochs=2, warmup_steps=0, learning_rate=1.0), 7, 100) == lr_at_step(by_epoch, 7, 100)
    assert lr_at_step(by_step, 0, 100) == pytest.approx(1 / 3)
    assert lr_at_step(by_step, 2, 100) == pytest.approx(1.0)
    assert lr_at_step(by_epoch, 2, 100) == pytest.approx(3 / 200)


def test_resume_accepts_a_checkpoint_written_before_the_new_keys_existed(demo, tmp_path):
    output = tmp_path / "run"
    train(demo, TINY, output, device="cpu", stop_after_epochs=1)
    new_keys = {"augment_rotation", "normal_sign_flip_probability", "bad_records", "warmup_steps", "normal_loss"}
    state = torch.load(output / "last.pt", weights_only=True)
    for section in state["config"].values():
        for key in list(section):
            if key in new_keys:
                del section[key]
    torch.save(state, output / "last.pt")
    config = json.loads((output / "config.json").read_text())
    for section in config.values():
        for key in list(section):
            if key in new_keys:
                del section[key]
    (output / "config.json").write_text(json.dumps(config))
    result = train(demo, ExperimentConfig.load(output / "config.json"), output, device="cpu", resume=output / "last.pt")
    assert result["completed_epochs"] == TINY.train.epochs


def test_training_runs_with_every_option_enabled(demo, tmp_path):
    corrupt = corrupt_copy(demo, tmp_path / "corrupt")
    config = ExperimentConfig(
        TINY.model,
        replace(TINY.data, augment_rotation="so3", normal_sign_flip_probability=0.5, bad_records="skip"),
        replace(TINY.train, normal_loss="unoriented", warmup_steps=2),
    )
    audit = validate_manifest(corrupt, bad_records="skip")
    manifest = write_filtered_manifest(corrupt, audit, tmp_path / "clean" / "manifest.json")
    report = tmp_path / "audit.json"
    report.write_text(json.dumps(validate_manifest(manifest, workers=2)))
    result = train(manifest, config, tmp_path / "run", device="cpu", audit=report)
    assert result["completed_epochs"] == 2 and np.isfinite(result["best_normalized_rmse"])
