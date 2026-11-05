"""Losses with explicit units and per-shape normalization."""

import torch
from torch.nn import functional as F

from .config import TrainConfig


def normal_angle(predicted: torch.Tensor, target: torch.Tensor, unoriented: bool) -> torch.Tensor:
    """The angle between predicted and target normals in radians, stable at zero.

    atan2 of the sine and cosine keeps a finite, bounded gradient at perfect
    alignment, where arccos of the cosine does not. Unoriented angles are
    between lines, so they lie in [0, pi/2].
    """
    dot = (predicted * target).sum(dim=-1)
    sine = (torch.linalg.cross(predicted, target, dim=-1).square().sum(dim=-1) + 1e-12).sqrt()
    return torch.atan2(sine, dot.abs() if unoriented else dot)


def reconstruction_loss(prediction: dict, batch: dict, config: TrainConfig) -> dict:
    target = batch["udf"].float() / batch["scale"].float()
    distance = F.smooth_l1_loss(
        prediction["normalized_udf"].float(), target, beta=config.huber_delta, reduction="none"
    ).mean(dim=1)
    if "unseen_distance" in prediction:
        # Union decoding: teach the head the distance to the unseen part directly. Where the disks
        # overestimate, the nearer surface is unseen and the head must match the target; elsewhere
        # it must only stay above it, so the soft minimum keeps the disks.
        from .model import UNION_MARGIN

        unseen = prediction["unseen_distance"].float()
        nearer = (prediction["prior_distance"].float() - target) > UNION_MARGIN
        match = F.smooth_l1_loss(unseen, target, beta=config.huber_delta, reduction="none")
        above = F.relu(target - unseen).square() / config.huber_delta
        distance = distance + torch.where(nearer, match, above).mean(dim=1)
    valid = batch["normal_valid"]
    if "normal_tensor" in prediction:
        # Projector head: squared Frobenius distance to n n^T, which is sign-free and
        # equals 2 sin^2(angle) once the prediction is itself a unit projector.
        normals = batch["query_normals"].float()
        projector = normals[..., :, None] * normals[..., None, :]
        penalty = (prediction["normal_tensor"].float() - projector).square().sum(dim=(-1, -2))
        if "normal_loss_weight" in prediction:  # the denoiser's per-query noise-level weight
            penalty = penalty * prediction["normal_loss_weight"].float()
    elif config.normal_penalty == "angle":
        penalty = normal_angle(
            prediction["normals"].float(), batch["query_normals"].float(), config.normal_loss == "unoriented"
        )
    else:
        cosine = (prediction["normals"].float() * batch["query_normals"].float()).sum(dim=-1).clamp(-1, 1)
        if config.normal_loss == "unoriented":
            cosine = cosine.abs()  # a flipped target normal is not an error
        penalty = 1 - cosine
    normal = (penalty * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
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
