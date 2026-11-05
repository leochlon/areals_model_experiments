"""Frozen released geometry features projected into AeroJEPA's context width."""
from pathlib import Path

import numpy as np
import torch
from torch import nn
from physicsnemo.experimental.models.aerojepa.encoders.base import BaseContextEncoder
from physicsnemo.experimental.models.aerojepa.layers import EncoderOutput, TokenSet
from drivaerml import DrivAerMLDataset


class FeatureDataset(DrivAerMLDataset):
    def __init__(self, root_dir, split, *, source, seed=0, **kwargs):
        super().__init__(root_dir, split, seed=seed)
        cache = Path(root_dir) / "backbone_tokens" / source
        assert (cache / "features_ready.json").exists(), f"run extract_backbone_tokens.py --source {source} first"
        self.features = []
        for row in self.records:
            path = cache / (row["id"] + ".npy")
            self.features.append(torch.from_numpy(np.load(path, allow_pickle=False)))

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        features = self.features[index]
        # AeroJEPA's inference API requires N feature rows for N input points.
        # Keep the cached token packet first and pad the unused rows with zero area.
        padding = features.new_zeros(len(sample["context_pos"])-len(features), features.shape[1])
        sample["context_feat"] = torch.cat([features, padding], dim=0)
        return sample


class FeatureContextEncoder(BaseContextEncoder):
    supports_batched_forward = True

    def __init__(self, feature_dim, token_dim=128):
        super().__init__()
        self.projection = nn.Linear(feature_dim, token_dim)

    def forward(self, *, context_pos, context_feat):
        context_feat = context_feat[context_feat[:, 3] > 0]
        coords, weights = context_feat[:, :3], context_feat[:, 3]
        features = self.projection(context_feat[:, 4:].detach())
        global_token = (features*weights[:, None]).sum(0)/weights.sum().clamp_min(1e-12)
        tokens = TokenSet(features=features, coords=coords,
                          mask=torch.ones(len(features), dtype=torch.bool, device=features.device),
                          global_token=global_token, aux=dict(area_weights=weights))
        return EncoderOutput(tokens=tokens, global_token=global_token,
                             aux=dict(point_token_count=len(features)))

    def forward_batched(self, *, context_pos, context_feat, context_pos_n):
        outputs = [self.forward(context_pos=p[:int(n)], context_feat=f)
                   for p, f, n in zip(context_pos, context_feat, context_pos_n, strict=True)]
        tokens = TokenSet(features=torch.stack([x.tokens.features for x in outputs]),
                          coords=torch.stack([x.tokens.coords for x in outputs]),
                          mask=torch.stack([x.tokens.mask for x in outputs]),
                          global_token=torch.stack([x.global_token for x in outputs]))
        return EncoderOutput(tokens=tokens, global_token=tokens.global_token)
