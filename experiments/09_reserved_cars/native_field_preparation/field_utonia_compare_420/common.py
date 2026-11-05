"""Shared native-package loading and benchmark inputs."""
import dataclasses
import json
from pathlib import Path

import torch
from areals import field
from areals.config import ExperimentConfig
from areals.model import GeometryAutoencoder
from areal_tokens.config import ExperimentConfig as PreviousConfig
from areal_tokens.data import read_manifest

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "config.json").read_text())


def experiment():
    previous = PreviousConfig.from_dict(json.loads((ROOT.parent / "config.json").read_text()))
    expected = json.loads((ROOT.parent / "field_prior_gate_check/compatibility.json").read_text())["removed_config"]
    default, converted = ExperimentConfig(), {}
    for section, values in previous.to_dict().items():
        allowed = {item.name for item in dataclasses.fields(getattr(default, section))}
        assert {key: value for key, value in values.items() if key not in allowed} == expected[section]
        converted[section] = {key: value for key, value in values.items() if key in allowed}
    return ExperimentConfig.from_dict(converted)


def records():
    result, _ = read_manifest(Path(CONFIG["data_root"]) / "manifest.json")
    assert len(result) == 436
    assert sum(record["split"] == "train" for record in result) == 420
    assert sum(record["split"] == "val" for record in result) == 16
    assert {record["split"] for record in result} == {"train", "val"}
    return result


def geometry(device="cpu"):
    state = torch.load(CONFIG["geometry_checkpoint"], map_location="cpu", weights_only=True)
    assert state["epoch"] == CONFIG["geometry_epoch"]
    config = experiment()
    model = GeometryAutoencoder(config.model)
    model.load_state_dict(state["model"], strict=True)
    return model.to(device).eval().requires_grad_(False), config


def field_source():
    return str(Path(field.__file__).resolve())
