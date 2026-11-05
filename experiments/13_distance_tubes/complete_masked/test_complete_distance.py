import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from areal_tokens.config import ModelConfig
from areal_tokens.model import GeometryAutoencoder, LatentGeometry, surfel_readout

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('areal_tokens.test_original_model', ROOT/'baseline/areal_tokens/model.py')
legacy = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = legacy
spec.loader.exec_module(legacy)


def pair(distance):
    torch.manual_seed(31)
    config = ModelConfig(width=48, heads=2, latents=16, encoder_depth=1, decoder_depth=1,
                         ff_multiplier=2, fourier_bands=3, query_chunk=64, normal_head='projector',
                         normal_prior='surfels', distance_prior=distance, input_condition='complete')
    model, old = GeometryAutoencoder(config).eval(), legacy.GeometryAutoencoder(config).eval()
    old.load_state_dict(model.state_dict(), strict=True)
    positions = torch.randn(2,16,3)
    geometry = LatentGeometry(tokens=torch.randn(2,16,48), positions=positions,
        normals=torch.nn.functional.normalize(torch.randn(2,16,3),dim=-1), weights=torch.ones(2,16),
        center=torch.tensor([[2.,-1.,4.],[-3.,2.,1.]]), scale=torch.tensor([[2.5],[.7]]),
        area=torch.ones(2,1), complete=torch.tensor([[1.],[0.]]))
    query = torch.randn(2,23,3)*geometry.scale[:,None]+geometry.center[:,None]
    return model,old,geometry,query


@pytest.mark.parametrize('distance',['surfels','union'])
def test_mixed_batch_routes_complete_only_and_preserves_normals(distance):
    model,old,geometry,query = pair(distance)
    with torch.no_grad():
        expected = old.decode(geometry,query)
        actual = model.decode(geometry,query)
        direct = surfel_readout(geometry,query)[0]
    assert torch.equal(actual['normalized_udf'][0],direct[0])
    assert torch.equal(actual['normalized_udf'][1],expected['normalized_udf'][1])
    assert torch.equal(actual['udf'],actual['normalized_udf']*geometry.scale)
    assert torch.equal(actual['normal_tensor'],expected['normal_tensor'])
    assert list(model.state_dict()) == list(old.state_dict())


@pytest.mark.parametrize('distance',['surfels','union'])
def test_legacy_missing_flag_means_complete(distance):
    model,_,geometry,query = pair(distance)
    geometry = replace(geometry,complete=None)
    with torch.no_grad():
        assert torch.equal(model.decode(geometry,query)['normalized_udf'],surfel_readout(geometry,query)[0])


@pytest.mark.parametrize('distance',['surfels','union','none'])
def test_masked_outputs_and_training_gradients_match_original(distance):
    model,old,geometry,query = pair(distance)
    geometry = replace(geometry,complete=torch.zeros(2,1))
    model.train();old.train()
    actual,expected = model.decode(geometry,query),old.decode(geometry,query)
    assert actual.keys() == expected.keys()
    assert all(torch.equal(actual[key],expected[key]) for key in actual)
    actual['normalized_udf'].sum().backward()
    expected['normalized_udf'].sum().backward()
    for (name,p),(_,q) in zip(model.named_parameters(),old.named_parameters()):
        if p.grad is None or q.grad is None:
            assert p.grad is None and q.grad is None, name
        else:
            assert torch.equal(p.grad,q.grad), name


def test_config_without_distance_prior_is_unchanged():
    model,old,geometry,query = pair('none')
    with torch.no_grad():
        a,b = model.decode(geometry,query),old.decode(geometry,query)
    assert all(torch.equal(a[k],b[k]) for k in a)
