"""The surfel normal as a noisy hint the decoder denoises (normal_prior="surfels_denoise")."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from areal_tokens.config import ExperimentConfig
from areal_tokens.data import GeometryDataset, collate_clouds, read_manifest
from areal_tokens.losses import normal_angle, reconstruction_loss
from areal_tokens.model import (
    EPS_WEIGHT_CAP,
    HINT_SIGMA,
    GeometryAutoencoder,
    outer6,
    rotate_by,
    surfel_readout,
    symmetric_matrix,
)
from areal_tokens.prepare import make_demo

SMALL = {"width": 48, "heads": 2, "latents": 16, "encoder_depth": 1, "decoder_depth": 1, "ff_multiplier": 2,
         "fourier_bands": 3, "query_chunk": 64}


def config(**model):
    return ExperimentConfig.from_dict(
        {
            "model": dict(SMALL, normal_head="projector", normal_prior="surfels_denoise", **model),
            "data": {"encoder_points": 128, "queries": 64, "mask_fraction": 0.6, "mask_patches": 8},
            "train": {"epochs": 1, "batch_size": 2, "workers": 0, "amp": "none", "normal_loss": "unoriented",
                      "normal_weight": 1.0},
        }
    )


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    return make_demo(tmp_path_factory.mktemp("demo"), train=2, val=2, test=2, points=256, queries=512)


def batch_of(manifest, experiment):
    records, _ = read_manifest(manifest)
    data = GeometryDataset(records[:2], experiment.model, experiment.data, 17, mask_fraction=0.6, fixed_queries=True)
    return collate_clouds([data[0], data[1]])


def test_denoise_options_validate():
    with pytest.raises(ValueError, match="surfels_denoise"):
        ExperimentConfig.from_dict({"model": {"normal_head": "projector", "normal_prior": "surfels",
                                              "normal_target": "eps"}, "train": {"normal_loss": "unoriented"}})
    with pytest.raises(ValueError, match="surfels_denoise"):
        ExperimentConfig.from_dict({"model": {"hint_noise": 0.5}})
    with pytest.raises(ValueError, match="between 0 and 1"):
        config(hint_noise=1.5)
    with pytest.raises(ValueError, match="residual or eps"):
        config(normal_target="x0")
    with pytest.raises(ValueError, match="projector"):
        ExperimentConfig.from_dict({"model": {"normal_prior": "surfels_denoise"}})


def test_rotation_noise_has_the_stated_angle():
    torch.manual_seed(0)
    normals = torch.nn.functional.normalize(torch.randn(500, 3), dim=-1)
    angles = torch.rand(500) * math.pi / 3
    rotated = rotate_by(normals, angles)
    assert torch.allclose(rotated.norm(dim=-1), torch.ones(500), atol=1e-5)
    assert torch.allclose(normal_angle(rotated, normals, False), angles, atol=1e-4)


@pytest.mark.parametrize("target", ["residual", "eps"])
def test_denoiser_starts_at_the_surfel_normal_and_never_reads_targets_at_evaluation(demo, target):
    experiment = config(normal_target=target, hint_noise=0.5)
    batch = batch_of(demo, experiment)
    torch.manual_seed(0)
    model = GeometryAutoencoder(experiment.model).eval()
    with torch.no_grad():
        geometry = model.encoder(batch)
        prediction = model(batch)
        _, tensor, _ = surfel_readout(geometry, batch["queries"], chunk=64)
        scrambled = model(dict(batch, query_normals=torch.randn_like(batch["query_normals"])))
    prior = torch.linalg.eigh(tensor)[1][..., -1]
    assert torch.rad2deg(normal_angle(prediction["normals"], prior, True)).max() < 0.01
    assert torch.equal(prediction["normals"], scrambled["normals"])
    assert torch.equal(prediction["normal_loss_weight"], torch.ones_like(prediction["normal_loss_weight"]))


def test_query_features_are_the_decoder_state_decode_uses(demo):
    experiment = config()  # no distance prior: the distance is the head's alone
    batch = batch_of(demo, experiment)
    torch.manual_seed(0)
    model = GeometryAutoencoder(experiment.model).eval()
    with torch.no_grad():
        geometry = model.encoder(batch)
        hidden = model.query_features(geometry, batch["queries"])
        distance = torch.nn.functional.softplus(model.distance_head(hidden).float().squeeze(-1))
        assert torch.allclose(distance, model.decode(geometry, batch["queries"])["normalized_udf"])


def test_training_hints_use_noised_targets_and_the_eps_loss_is_the_weighted_clean_loss(demo):
    experiment = config(normal_target="eps", hint_noise=1.0)
    batch = batch_of(demo, experiment)
    torch.manual_seed(0)
    model = GeometryAutoencoder(experiment.model).train()
    with torch.no_grad():  # a nonzero head, so the noise estimate matters
        model.normal_head.weight.normal_(0, 0.1)
        model.normal_head.bias.normal_(0, 0.1)
    prediction = model(batch)
    weight, valid = prediction["normal_loss_weight"], batch["normal_valid"]
    assert (weight[valid] < 1).any() and weight.max() <= EPS_WEIGHT_CAP  # noised queries carry their own level
    assert (weight[~valid] == 1).all()  # queries without a target keep the surfel hint
    # || eps_hat - (H - N) / sigma ||^2 * HINT_SIGMA^2 == weight * || clean - N ||^2 when clean = H - sigma eps_hat
    normals = batch["query_normals"].float()
    target = normals[..., :, None] * normals[..., None, :]
    clean_loss = (prediction["normal_tensor"] - target).square().sum(dim=(-1, -2))
    losses = reconstruction_loss(prediction, batch, experiment.train)
    manual = ((clean_loss * weight) * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
    assert torch.allclose(losses["normal"], manual.mean())
    sigma = torch.full((3,), math.radians(20.0))
    hint = torch.nn.functional.normalize(torch.randn(3, 3), dim=-1)
    truth = rotate_by(hint, sigma)
    eps_hat = torch.randn(3, 6)
    H, N = symmetric_matrix(outer6(hint)), symmetric_matrix(outer6(truth))
    clean = H - sigma[:, None, None] * symmetric_matrix(eps_hat)
    eps_loss = (symmetric_matrix(eps_hat) - (H - N) / sigma[:, None, None]).square().sum(dim=(-1, -2))
    w = (HINT_SIGMA / sigma).square()
    assert torch.allclose(eps_loss * HINT_SIGMA**2, w * (clean - N).square().sum(dim=(-1, -2)), rtol=1e-4)


def test_denoise_model_trains_saves_and_reloads(demo, tmp_path):
    from areal_tokens.io import load_model
    from areal_tokens.train import train

    experiment = config(normal_target="eps", hint_noise=0.5, distance_prior="surfels")
    result = train(demo, experiment, tmp_path / "run", device="cpu")
    model, loaded, _ = load_model(result["checkpoint"])
    assert loaded.model.normal_target == "eps" and model.decoder.query_condition.in_features == 10


def test_outer6_matches_the_symmetric_matrix_layout():
    n = torch.nn.functional.normalize(torch.randn(7, 3), dim=-1)
    assert torch.allclose(symmetric_matrix(outer6(n)), n[:, :, None] * n[:, None, :])


def test_existing_priors_are_unchanged_by_the_denoise_options():
    base = ExperimentConfig.from_dict({"model": dict(SMALL, normal_head="projector", normal_prior="surfels",
                                                     distance_prior="surfels"), "train": {"normal_loss": "unoriented"}})
    model = GeometryAutoencoder(base.model)
    assert model.decoder.query_condition is None and not model.denoise
    assert replace(base.model).normal_target == "residual" and base.model.hint_noise == 0.0
