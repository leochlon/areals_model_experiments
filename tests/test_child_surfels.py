"""Denser surfels under the same tokens (model.surfels_per_token)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from areal_tokens.config import ExperimentConfig
from areal_tokens.data import GeometryDataset, collate_clouds, read_manifest
from areal_tokens.geometry import farthest_indices, prepare_cloud
from areal_tokens.io import load_latents
from areal_tokens.model import GeometryAutoencoder, LatentGeometry, surfel_readout
from areal_tokens.prepare import make_demo

SMALL = {"width": 48, "heads": 2, "latents": 16, "encoder_depth": 1, "decoder_depth": 1, "ff_multiplier": 2,
         "fourier_bands": 3, "query_chunk": 64}


def config(per=4, **model):
    return ExperimentConfig.from_dict(
        {
            "model": dict(SMALL, normal_head="projector", normal_prior="surfels", distance_prior="surfels",
                          surfels_per_token=per, **model),
            "data": {"encoder_points": 256, "queries": 64, "mask_fraction": 0.6, "mask_patches": 8},
            "train": {"epochs": 1, "batch_size": 2, "workers": 0, "amp": "none", "normal_loss": "unoriented"},
        }
    )


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    return make_demo(tmp_path_factory.mktemp("demo"), train=2, val=2, test=2, points=512, queries=512)


def test_surfels_per_token_validates():
    with pytest.raises(ValueError, match="positive"):
        config(per=0)
    with pytest.raises(ValueError, match="surfel normal_prior or distance_prior"):
        ExperimentConfig.from_dict({"model": {"surfels_per_token": 4}})
    ExperimentConfig.from_dict({"model": {"surfels_per_token": 4, "distance_prior": "surfels"}})


def test_farthest_point_sampling_is_greedy_so_anchors_prefix_the_surfels():
    rng = np.random.default_rng(0)
    points, weights = rng.normal(size=(600, 3)), rng.random(600) + 0.1
    assert np.array_equal(farthest_indices(points, weights, 200)[:50], farthest_indices(points, weights, 50))


def test_samples_carry_surfels_that_extend_the_anchors(demo):
    records, _ = read_manifest(demo)
    experiment = config()
    data = GeometryDataset(records[:2], experiment.model, experiment.data, 17, mask_fraction=0.6, fixed_queries=True)
    sample = data[0]
    assert len(sample["surfels"]) == min(4 * 16, len(sample["points"]))
    assert np.array_equal(sample["surfels"][:16], sample["anchors"])
    assert np.array_equal(sample["surfel_normals"][:16], sample["anchor_normals"])
    batch = collate_clouds([data[0], data[1]])
    assert batch["surfels"].shape[:2] == batch["surfel_weights"].shape
    plain = GeometryDataset(records[:1], config(per=1).model, experiment.data, 17, mask_fraction=0.6,
                            fixed_queries=True)[0]
    assert "surfels" not in plain
    assert np.array_equal(plain["anchors"], sample["anchors"])  # the anchors do not depend on the surfels


def test_readout_reads_the_surfels_when_the_packet_has_them():
    grid = torch.stack(torch.meshgrid(torch.linspace(-1, 1, 12), torch.linspace(-1, 1, 12), indexing="ij"), -1)
    dense = torch.cat([grid.reshape(-1, 2), torch.zeros(144, 1)], -1)[None]
    up = torch.zeros(1, 144, 3)
    up[..., 2] = 1
    coarse = dense[:, ::9]
    tilted = torch.nn.functional.normalize(torch.tensor([1.0, 0.0, 1.0]), dim=0).expand(1, coarse.shape[1], 3)
    frame = dict(center=torch.zeros(1, 3), scale=torch.ones(1, 1), area=torch.ones(1, 1))
    geometry = LatentGeometry(torch.zeros(1, coarse.shape[1], 4), coarse, tilted.clone(),
                              torch.full((1, coarse.shape[1]), 1.0), **frame, surfel_positions=dense,
                              surfel_normals=up, surfel_weights=torch.full((1, 144), 1.0))
    query = torch.tensor([[[0.05, 0.05, 0.1]]])
    distance, tensor, _ = surfel_readout(geometry, query)
    assert torch.allclose(distance, torch.tensor([[0.1]]), atol=1e-4)
    assert torch.allclose(torch.linalg.eigh(tensor[0, 0])[1][:, -1].abs(), torch.tensor([0.0, 0.0, 1.0]), atol=1e-4)


def test_packets_export_and_reload_their_surfels(demo, tmp_path):
    from areal_tokens.evaluate import encode_checkpoint
    from areal_tokens.train import train

    result = train(demo, config(), tmp_path / "run", device="cpu")
    encode_checkpoint(result["checkpoint"], demo, tmp_path / "packets", "cpu", split="test")
    files = sorted((tmp_path / "packets").glob("shape_*.npz"))
    assert files
    packet = load_latents(files[0])
    assert packet.surfel_positions is not None and packet.surfel_positions.shape[1] > packet.positions.shape[1]
    model = GeometryAutoencoder(config().model).eval()
    with torch.no_grad():
        out = model.decode(packet, packet.world_positions[:, :5])
    assert torch.isfinite(out["udf"]).all()


def test_prepare_cloud_default_is_unchanged():
    from areal_tokens.geometry import canonical_cloud

    rng = np.random.default_rng(1)
    points = rng.normal(size=(300, 3))
    normals = points / np.linalg.norm(points, axis=1, keepdims=True)
    cloud = canonical_cloud(points, normals, np.full(300, 1 / 300))
    a = prepare_cloud(cloud, 32, 256, np.random.default_rng(5))
    b = prepare_cloud(cloud, 32, 256, np.random.default_rng(5), surfel_count=128)
    for key in a:
        assert np.array_equal(a[key], b[key])
    assert "surfels" not in a and len(b["surfels"]) == 128


def test_complete_flag_reaches_the_decoder_and_defaults_to_complete(demo, tmp_path):
    from areal_tokens.evaluate import encode_checkpoint
    from areal_tokens.train import train

    records, _ = read_manifest(demo)
    experiment = config(input_condition="complete")
    masked = GeometryDataset(records[:1], experiment.model, experiment.data, 17, mask_fraction=0.6, fixed_queries=True)
    full = GeometryDataset(records[:1], experiment.model, experiment.data, 17, mask_fraction=0.0, fixed_queries=True)
    assert masked[0]["complete"][0] == 0 and full[0]["complete"][0] == 1
    batch = collate_clouds([full[0]])
    torch.manual_seed(0)
    model = GeometryAutoencoder(experiment.model).eval()
    with torch.no_grad():
        model.decoder.condition.weight.fill_(1.0)
        geometry = model.encoder(batch)
        a = model.decode(geometry, batch["queries"])["udf"]
        b = model.decode(geometry.with_tokens(geometry.tokens), batch["queries"])["udf"]
        from dataclasses import replace

        c = model.decode(replace(geometry, complete=torch.zeros(1, 1)), batch["queries"])["udf"]
        d = model.decode(replace(geometry, complete=None), batch["queries"])["udf"]
    assert torch.equal(a, b) and torch.equal(a, d) and not torch.equal(a, c)
    result = train(demo, experiment, tmp_path / "run", device="cpu")
    encode_checkpoint(result["checkpoint"], demo, tmp_path / "packets", "cpu", split="test")
    packet = load_latents(sorted((tmp_path / "packets").glob("shape_*.npz"))[0])
    assert packet.complete is not None and packet.complete.item() == 1.0
