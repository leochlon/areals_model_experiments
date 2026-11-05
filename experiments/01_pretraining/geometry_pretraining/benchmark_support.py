"""Pool paths, and a validation hook that also reports the loss of the earlier geometry_latents objective."""

import importlib
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
POOL = Path("/workspace/runs/objaverse_training_pool/versions/20260927T140652Z")
MANIFEST = POOL / "manifest.json"


def install_shared_scoring():
    evaluation = importlib.import_module("areal_tokens.evaluate")
    if getattr(evaluation.evaluate_model, "_shared_scoring", False):
        return
    native_evaluate = evaluation.evaluate_model
    native_loss = evaluation.reconstruction_loss

    def evaluate(*args, **kwargs):
        legacy_losses = []

        def score(prediction, batch, config):
            result = native_loss(prediction, batch, config)
            comparable = {key: prediction[key] for key in ("normals", "udf", "normalized_udf")}
            legacy_config = replace(config, normal_weight=0.1, normal_penalty="cosine",
                                    normal_loss="unoriented")
            legacy_losses.append(float(native_loss(comparable, batch, legacy_config)["loss"]))
            return result

        evaluation.reconstruction_loss = score
        try:
            report = native_evaluate(*args, **kwargs)
        finally:
            evaluation.reconstruction_loss = native_loss
        assert len(legacy_losses) == len(report["shapes"])
        for row, value in zip(report["shapes"], legacy_losses, strict=True):
            row["legacy_loss"] = value
        report["aggregate"]["legacy_loss"] = float(np.mean(legacy_losses))
        report["legacy_loss_definition"] = (
            "Original per-shape normalized-distance SmoothL1(beta=0.05) "
            "+ 0.1 * mean(1-abs(normal cosine)) over the original valid-normal mask."
        )
        return report

    evaluate._shared_scoring = True
    evaluation.evaluate_model = evaluate


