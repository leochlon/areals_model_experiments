import json
from pathlib import Path

import torch

import study

ROOT = Path(__file__).resolve().parent


torch.manual_seed(0)
study.configure("measurements")
assert study.REF is None
base, experiment, state = study.fb.load_model(study.BASE, "cpu")
fresh = study.fb.GeometryAutoencoder(experiment.model)
assert set(base.state_dict()) == set(fresh.state_dict())
assert all(base.state_dict()[k].shape == v.shape for k,v in fresh.state_dict().items())
assert any(not torch.equal(base.state_dict()[k], v) for k, v in fresh.state_dict().items())
encoder_difference = max(float((base.encoder.state_dict()[k]-v).abs().max())
    for k,v in fresh.encoder.state_dict().items() if v.is_floating_point())
assert encoder_difference > .01
# The fresh constructor is the one the harness uses for --scratch; run in its own process so it leaves no state.
result = dict(status="PASS", maximum_encoder_weight_difference=encoder_difference)
(ROOT / "initialisation_check.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result),flush=True)
