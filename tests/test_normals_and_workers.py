"""Normal penalties, the gradient readout, the normal breakdown, and pools that survive dead workers."""

from __future__ import annotations

import math
import os
import signal

import numpy as np
import pytest
import torch

from areal_tokens.config import ExperimentConfig, TrainConfig
from areal_tokens.data import GeometryDataset, read_manifest
from areal_tokens.evaluate import evaluate_model, gradient_normals, normal_breakdown
from areal_tokens.losses import normal_angle, reconstruction_loss
from areal_tokens.model import GeometryAutoencoder
from areal_tokens.parallel import map_jobs
from areal_tokens.prepare import make_demo

TINY = ExperimentConfig.from_dict(
    {
        "model": {"width": 48, "heads": 2, "latents": 16, "encoder_depth": 1, "decoder_depth": 1,
                  "ff_multiplier": 2, "fourier_bands": 3, "query_chunk": 64},
        "data": {"encoder_points": 128, "queries": 64, "mask_fraction": 0.6, "mask_patches": 8},
        "train": {"epochs": 1, "batch_size": 2, "workers": 0, "amp": "none", "normal_loss": "unoriented"},
    }
)


def kill_some(job):
    if job % 7 == 3:
        os.kill(os.getpid(), signal.SIGKILL)  # what the kernel's out-of-memory killer does
    return job * 2


def test_pool_survives_killed_workers_and_reports_them():
    results = map_jobs(kill_some, list(range(20)), 2, on_crash=lambda job: ("died", job))
    assert results == [("died", job) if job % 7 == 3 else job * 2 for job in range(20)]
    with pytest.raises(RuntimeError, match="worker process died"):
        map_jobs(kill_some, [1, 3, 5], 2)


def test_angle_penalty_is_the_angle_and_is_stable_at_zero():
    torch.manual_seed(0)
    a = torch.nn.functional.normalize(torch.randn(500, 3, dtype=torch.float64), dim=-1)
    b = torch.nn.functional.normalize(torch.randn(500, 3, dtype=torch.float64), dim=-1)
    exact = torch.acos((a * b).sum(-1).clamp(-1, 1))
    assert torch.allclose(normal_angle(a, b, unoriented=False), exact, atol=1e-5)
    assert torch.allclose(normal_angle(a, -b, unoriented=True), torch.minimum(exact, math.pi - exact), atol=1e-5)
    same = a.clone().requires_grad_(True)
    normal_angle(same, a, unoriented=True).sum().backward()
    assert torch.isfinite(same.grad).all() and same.grad.abs().max() < 2


def test_angle_penalty_scales_the_loss_as_documented():
    torch.manual_seed(1)
    normals = torch.nn.functional.normalize(torch.randn(1, 64, 3), dim=-1)
    tilt = math.radians(10)
    axis = torch.nn.functional.normalize(torch.linalg.cross(normals, torch.randn(1, 64, 3), dim=-1), dim=-1)
    target = normals * math.cos(tilt) + torch.linalg.cross(axis, normals, dim=-1) * math.sin(tilt)
    batch = dict(udf=torch.zeros(1, 64), scale=torch.ones(1, 1), normal_valid=torch.ones(1, 64, dtype=torch.bool),
                 query_normals=target)
    prediction = dict(normalized_udf=torch.zeros(1, 64), udf=torch.zeros(1, 64), normals=normals)
    cosine = reconstruction_loss(prediction, batch, TrainConfig(normal_loss="unoriented"))["normal"]
    angle = reconstruction_loss(prediction, batch, TrainConfig(normal_loss="unoriented", normal_penalty="angle"))["normal"]
    assert cosine.item() == pytest.approx(1 - math.cos(tilt), rel=1e-3)  # 0.015 at 10 degrees
    assert angle.item() == pytest.approx(tilt, rel=1e-3)  # 0.175 at 10 degrees


def test_gradient_readout_is_the_unit_gradient_of_the_decoded_distance():
    model = GeometryAutoencoder(TINY.model)
    center = torch.tensor([0.3, -0.2, 0.1])

    def decode(packet, queries):  # a known field: distance to a point
        distance = (queries - center).norm(dim=-1)
        return dict(udf=distance, normalized_udf=distance, normals=torch.zeros_like(queries))

    model.decode = decode
    queries = torch.randn(2, 50, 3)
    prediction = gradient_normals(model, None, queries)
    expected = torch.nn.functional.normalize(queries - center, dim=-1)
    assert torch.allclose(prediction["normals"], expected, atol=1e-5)
    assert not prediction["udf"].requires_grad


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    return make_demo(tmp_path_factory.mktemp("demo"), train=2, val=2, test=2, points=256, queries=512)


def test_breakdown_is_additive_and_matches_the_aggregate(demo):
    records, _ = read_manifest(demo)
    torch.manual_seed(0)
    model = GeometryAutoencoder(TINY.model)
    data = GeometryDataset(records, TINY.model, TINY.data, 17, mask_fraction=0.0, fixed_queries=True)
    head = evaluate_model(model, data, TINY, torch.device("cpu"))
    breakdown = head["normal_breakdown"]
    assert breakdown["all"]["mean"] == pytest.approx(head["aggregate"]["normal_degrees"], rel=1e-4)
    by_distance = breakdown["by_distance"].values()
    assert sum(part["queries"] for part in by_distance if part) == breakdown["all"]["queries"]
    assert list(breakdown["by_distance"]) == ["0-0.01", "0.01-0.03", "0.03-0.1"]
    # the gradient readout changes normals only: distances and every distance metric are identical
    gradient = evaluate_model(model, data, TINY, torch.device("cpu"), normal_readout="gradient")
    for key in ("rmse", "normalized_rmse", "mae"):
        assert gradient["aggregate"][key] == pytest.approx(head["aggregate"][key], rel=1e-6)
    assert gradient["normal_readout"] == "gradient"


def test_breakdown_bins_follow_a_narrow_band():
    angles, distances = np.array([5.0, 15.0, 50.0]), np.array([0.005, 0.015, 0.02])
    result = normal_breakdown(angles, distances, np.array([0.0, 0.02, 0.1]), band=0.02)
    assert list(result["by_distance"]) == ["0-0.01", "0.01-0.02"]
    assert result["by_distance"]["0.01-0.02"]["queries"] == 2
    assert result["all"]["under_10"] == pytest.approx(1 / 3)
    assert normal_breakdown(np.zeros(0), np.zeros(0), np.zeros(0), 0.1) is None


# --- the projector normal head (normals as lines) ---------------------------------------------


def test_symmetric_matrix_layout():
    from areal_tokens.model import symmetric_matrix

    matrix = symmetric_matrix(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    assert torch.equal(matrix, torch.tensor([[1.0, 4.0, 5.0], [4.0, 2.0, 6.0], [5.0, 6.0, 3.0]]))


def test_projector_head_reads_the_top_eigenvector_and_its_loss_is_sign_free():
    from dataclasses import replace

    config = ExperimentConfig(replace(TINY.model, normal_head="projector"), TINY.data, TINY.train)
    torch.manual_seed(0)
    model = GeometryAutoencoder(config.model)
    assert model.normal_head.out_features == 6
    assert GeometryAutoencoder(TINY.model).normal_head.out_features == 3  # default unchanged
    hidden = torch.randn(2, 5, config.model.width)
    model.query_features = lambda geometry, queries: hidden
    geometry = type("G", (), dict(scale=torch.ones(2, 1)))()
    prediction = model.decode(geometry, torch.zeros(2, 5, 3))
    values, vectors = torch.linalg.eigh(prediction["normal_tensor"])
    assert torch.allclose(prediction["normals"], vectors[..., -1])
    assert torch.allclose(prediction["normals"].norm(dim=-1), torch.ones(2, 5), atol=1e-5)
    # the loss is zero at n n^T and does not see the sign of the target normal
    n = torch.nn.functional.normalize(torch.randn(1, 8, 3), dim=-1)
    exact = dict(udf=torch.zeros(1, 8), normalized_udf=torch.zeros(1, 8), normals=n,
                 normal_tensor=n[..., :, None] * n[..., None, :])
    batch = dict(udf=torch.zeros(1, 8), scale=torch.ones(1, 1), normal_valid=torch.ones(1, 8, dtype=torch.bool))
    for sign in (1.0, -1.0):
        loss = reconstruction_loss(exact, dict(batch, query_normals=sign * n), config.train)
        assert loss["normal"].item() == pytest.approx(0.0, abs=1e-10)


def test_projector_requires_the_unoriented_loss():
    with pytest.raises(ValueError, match="unoriented"):
        ExperimentConfig.from_dict({"model": {"normal_head": "projector"}})
    ExperimentConfig.from_dict({"model": {"normal_head": "projector"}, "train": {"normal_loss": "unoriented"}})


def test_projector_model_trains_saves_and_reloads(demo, tmp_path):
    from dataclasses import replace

    from areal_tokens.io import load_model
    from areal_tokens.train import train

    config = ExperimentConfig(replace(TINY.model, normal_head="projector"), TINY.data, replace(TINY.train, epochs=1))
    result = train(demo, config, tmp_path / "run", device="cpu")
    model, loaded, _ = load_model(result["checkpoint"])
    assert loaded.model.normal_head == "projector" and model.normal_head.out_features == 6


def test_anchor_prior_recovers_a_plane_regardless_of_normal_signs():
    from areal_tokens.model import LatentGeometry, anchor_normal_prior

    torch.manual_seed(0)
    anchors = torch.cat([torch.rand(1, 64, 2) * 2 - 1, torch.zeros(1, 64, 1)], dim=-1)
    signs = torch.where(torch.rand(1, 64, 1) < 0.5, -1.0, 1.0)  # inconsistent winding
    normals = torch.tensor([0.0, 0.0, 1.0]).expand(1, 64, 3) * signs
    weights = torch.full((1, 64), 1 / 64)
    geometry = LatentGeometry(torch.zeros(1, 64, 8), anchors, normals, weights,
                              torch.zeros(1, 3), torch.ones(1, 1), torch.ones(1, 1))
    queries = torch.randn(1, 100, 3) * 0.3
    prior = anchor_normal_prior(geometry, queries)
    top = torch.linalg.eigh(prior)[1][..., -1]
    assert torch.allclose(top[..., 2].abs(), torch.ones(1, 100), atol=1e-5)
    # padded anchors (zero weight) are ignored
    padded = LatentGeometry(torch.zeros(1, 65, 8), torch.cat([anchors, torch.zeros(1, 1, 3) + 0.01], 1),
                            torch.cat([normals, torch.tensor([[[1.0, 0.0, 0.0]]])], 1),
                            torch.cat([weights, torch.zeros(1, 1)], 1),
                            torch.zeros(1, 3), torch.ones(1, 1), torch.ones(1, 1))
    assert torch.allclose(anchor_normal_prior(padded, queries), prior, atol=1e-6)


def test_anchor_prior_needs_the_projector_head_and_starts_from_the_prior():
    with pytest.raises(ValueError, match="projector"):
        ExperimentConfig.from_dict({"model": {"normal_prior": "anchors"}, "train": {"normal_loss": "unoriented"}})
    config = ExperimentConfig.from_dict(
        {"model": {"normal_head": "projector", "normal_prior": "anchors", "width": 48, "heads": 2, "latents": 16,
                   "encoder_depth": 1, "decoder_depth": 1, "ff_multiplier": 2, "fourier_bands": 3},
         "train": {"normal_loss": "unoriented"}})
    model = GeometryAutoencoder(config.model)
    assert model.normal_head.weight.abs().sum() == 0 and model.normal_head.bias.abs().sum() == 0


def test_surfel_readout_is_exact_on_a_plane_and_flags_uncovered_queries():
    from areal_tokens.model import LatentGeometry, surfel_readout

    grid = torch.stack(torch.meshgrid(torch.linspace(-1, 1, 12), torch.linspace(-1, 1, 12), indexing="ij"), -1)
    anchors = torch.cat([grid.reshape(1, -1, 2), torch.zeros(1, 144, 1)], -1)
    signs = torch.where(torch.rand(1, 144, 1) < 0.5, -1.0, 1.0)
    normals = torch.tensor([0.0, 0.0, 1.0]).expand(1, 144, 3) * signs
    geometry = LatentGeometry(torch.zeros(1, 144, 8), anchors, normals, torch.full((1, 144), 1 / 144),
                              torch.zeros(1, 3), torch.ones(1, 1), torch.ones(1, 1))
    inside = torch.cat([torch.rand(1, 50, 2) * 1.6 - 0.8, torch.rand(1, 50, 1) * 0.2 - 0.1], -1)
    distance, tensor, evidence = surfel_readout(geometry, inside)
    height, spacing = inside[..., 2].abs(), 2 / 11
    # disks of radius spacing/2 on a square grid leave small gaps at the cell corners: the readout is exact
    # wherever the projection lands inside a disk, and overestimates by at most the corner gap elsewhere
    gap = (2 ** 0.5 / 2 - 0.5) * spacing
    assert (distance >= height - 1e-6).all() and (distance <= (height ** 2 + gap ** 2).sqrt() + 1e-6).all()
    covered = evidence[..., 0] < 1e-6
    assert covered.sum() > 25 and torch.allclose(distance[covered], height[covered], atol=1e-5)
    assert torch.allclose(torch.linalg.eigh(tensor)[1][..., -1][..., 2].abs(), torch.ones(1, 50), atol=1e-5)
    assert evidence[..., 0].max() < gap / spacing + 1e-4  # nothing projects further outside than a corner gap
    beyond = torch.tensor([[[3.0, 0.0, 0.0]]])  # far past the edge of the covered plane
    distance, _, evidence = surfel_readout(geometry, beyond)
    assert distance.item() > 1.8 and evidence[0, 0, 0] > 10


def test_surfel_priors_configure_and_start_trusting_the_readout():
    with pytest.raises(ValueError, match="projector"):
        ExperimentConfig.from_dict({"model": {"normal_prior": "surfels"}, "train": {"normal_loss": "unoriented"}})
    small = {"width": 48, "heads": 2, "latents": 16, "encoder_depth": 1, "decoder_depth": 1, "ff_multiplier": 2,
             "fourier_bands": 3}
    both = ExperimentConfig.from_dict({"model": dict(small, normal_head="projector", normal_prior="surfels",
                                                     distance_prior="surfels"), "train": {"normal_loss": "unoriented"}})
    model = GeometryAutoencoder(both.model)
    assert torch.sigmoid(model.prior_gate.bias).allclose(torch.full((2,), 0.8808), atol=1e-4)
    assert model.prior_evidence[-1].weight.abs().sum() == 0
    distance_only = ExperimentConfig.from_dict({"model": dict(small, distance_prior="surfels")})
    assert GeometryAutoencoder(distance_only.model).normal_head.out_features == 3


def test_union_loss_teaches_the_unseen_distance_only_where_the_disks_overestimate():
    config = TrainConfig(normal_loss="unoriented")
    target = torch.tensor([[0.10, 0.10, 0.10]])
    prior = torch.tensor([[0.10, 0.30, 0.10]])  # query 1: the disks overestimate, the nearer surface is unseen
    unseen = torch.tensor([[0.50, 0.10, 0.05]])  # query 0 stays above (fine); query 2 undercuts (penalised)
    tau = 0.01
    stacked = -torch.stack([prior, unseen], -1) / tau
    prediction = dict(normalized_udf=-tau * torch.logsumexp(stacked, -1), udf=target, normals=torch.zeros(1, 3, 3),
                      prior_distance=prior, unseen_distance=unseen.clone().requires_grad_(True))
    batch = dict(udf=target, scale=torch.ones(1, 1), normal_valid=torch.zeros(1, 3, dtype=torch.bool),
                 query_normals=torch.zeros(1, 3, 3))
    loss = reconstruction_loss(prediction, batch, config)["loss"]
    loss.backward()
    grad = prediction["unseen_distance"].grad[0]
    assert grad[0] == 0  # above the target where the disks are right: no push
    assert grad[1] == 0  # matched where the disks overestimate
    assert grad[2] < 0  # undercutting the target is pushed back up
