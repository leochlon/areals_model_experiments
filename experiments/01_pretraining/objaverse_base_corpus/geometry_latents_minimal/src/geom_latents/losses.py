"""Losses with explicit units and per-shape normalization."""

import torch
from torch.nn import functional as F

from .config import TrainConfig


def reconstruction_loss(prediction: dict, batch: dict, config: TrainConfig) -> dict:
    target = batch["udf"].float() / batch["scale"].float()
    distance = F.smooth_l1_loss(
        prediction["normalized_udf"].float(), target, beta=config.huber_delta, reduction="none"
    ).mean(dim=1)
    valid = batch["normal_valid"]
    cosine = (prediction["normals"].float() * batch["query_normals"].float()).sum(dim=-1).clamp(-1, 1)
    if config.normal_loss == "unoriented":
        cosine = cosine.abs()  # a flipped target normal is not an error
    normal = ((1 - cosine) * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
    total = distance + config.normal_weight * normal
    return dict(
        loss=total.mean(),
        distance=distance.mean(),
        normal=normal.mean(),
        normal_fraction=valid.float().mean(),
    )


@torch.no_grad()
def reconstruction_metrics(prediction: dict, batch: dict, unoriented: bool = False) -> dict:
    error = prediction["udf"].float() - batch["udf"].float()
    valid = batch["normal_valid"]
    cosine = (prediction["normals"].float() * batch["query_normals"].float()).sum(-1).clamp(-1, 1)
    if unoriented:
        cosine = cosine.abs()  # angle between lines, at most 90 degrees
    angle = torch.rad2deg(torch.acos(cosine))
    return dict(
        mse=error.square().mean(dim=1),
        normalized_mse=(error / batch["scale"].float()).square().mean(dim=1),
        mae=error.abs().mean(dim=1),
        normal_angle_sum=(angle * valid).sum(dim=1),
        normal_count=valid.sum(dim=1),
    )
