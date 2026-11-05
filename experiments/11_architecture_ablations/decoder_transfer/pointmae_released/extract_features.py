"""Load the released, coordinates-only Point-MAE encoder without its legacy training extensions."""
import ast
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from timm.layers import DropPath, trunc_normal_
import yaml

ROOT = Path(__file__).resolve().parent


def namespace(value):
    return SimpleNamespace(**{key: namespace(item) if isinstance(item, dict) else item
                              for key, item in value.items()})


def fps(points, count):
    """Batched farthest-point sampling, starting at index zero as in pointnet2."""
    batch, length, _ = points.shape
    indices = torch.empty((batch, count), dtype=torch.long, device=points.device)
    distance = torch.full((batch, length), 1e10, device=points.device)
    selected = torch.zeros(batch, dtype=torch.long, device=points.device)
    rows = torch.arange(batch, device=points.device)
    for index in range(count):
        indices[:, index] = selected
        delta = points - points[rows, selected][:, None]
        distance = torch.minimum(distance, (delta * delta).sum(-1))
        selected = distance.max(-1).indices
    return points[rows[:, None], indices].contiguous()


class KNN:
    def __init__(self, k, transpose_mode=True):
        assert transpose_mode
        self.k = k

    def __call__(self, points, centers):
        distances = torch.cdist(centers, points, compute_mode="donot_use_mm_for_euclid_dist")
        return distances.topk(self.k, dim=-1, largest=False, sorted=True)


def author_encoder():
    path = ROOT / "pointmae_source/models/Point_MAE.py"
    names = {"Encoder", "Group", "Mlp", "Attention", "Block", "TransformerEncoder", "MaskTransformer"}
    tree = ast.parse(path.read_text(), filename=str(path))
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
    assert {node.name for node in classes} == names
    # Execute the author's classes unchanged. Imports used only for geometry
    # pretraining (Chamfer/CUDA extensions) are unnecessary for frozen inference.
    scope = dict(torch=torch, nn=nn, F=F, np=np, random=random, DropPath=DropPath,
                 trunc_normal_=trunc_normal_, KNN=KNN, misc=SimpleNamespace(fps=fps),
                 print_log=lambda message, **kwargs: print(message, flush=True))
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(path), "exec"), scope)
    configuration = namespace(yaml.safe_load((ROOT / "pointmae_source/cfgs/pretrain.yaml").read_text())["model"])
    encoder = scope["MaskTransformer"](configuration)
    checkpoint = ROOT / "weights/pretrain.pth"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    prefix = "module.MAE_encoder."
    tensors = {key[len(prefix):]: value for key, value in state["base_model"].items() if key.startswith(prefix)}
    encoder.load_state_dict(tensors, strict=True)
    assert all(torch.equal(value, tensors[key]) for key, value in encoder.state_dict().items())
    metadata = dict(checkpoint_epoch=state["epoch"], encoder_tensors_loaded=len(tensors), strict_load=True,
                    encoder_parameters=sum(parameter.numel() for parameter in encoder.parameters()),
                    author_encoder_classes_unmodified=True, legacy_ops="PyTorch FPS and exact kNN")
    return encoder.cuda().eval().requires_grad_(False), scope["Group"](64, 32).cuda(), metadata

