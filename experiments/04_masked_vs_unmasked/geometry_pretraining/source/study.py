import json
import os
from pathlib import Path

import numpy as np
import torch

import areal_tokens.data as data_module
from areal_tokens.config import ExperimentConfig
from areal_tokens.data import GeometryDataset, collate_clouds, load_cloud_file
from areal_tokens.geometry import frame
from areal_tokens.model import GeometryAutoencoder
from areal_tokens.train import optimizer_for, seed_everything

ROOT = Path(os.environ['STUDY_ROOT'])
ARMS = ['masked_native', 'unmasked_native', 'masked_common_frame']
SEEDS = [17, 18, 19]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def configuration(seed, arm):
    source = json.loads((ROOT / 'production_config.json').read_text())
    source['train']['seed'] = seed
    source['data']['mask_fraction'] = 0. if arm == 'unmasked_native' else .6
    return ExperimentConfig.from_dict(source)


def build(seed, arm, device=None, rank_seed_offset=0):
    config = configuration(seed, arm)
    seed_everything(seed + rank_seed_offset)
    model = GeometryAutoencoder(config.model)
    if device is not None:
        model.to(device)
    optimizer = optimizer_for(model, config.train)
    return model, optimizer, config


class CommonFrameDataset(GeometryDataset):
    def _sample(self, record):
        cloud, targets = load_cloud_file(record['path'])
        if self.training:
            cloud, _ = self._augment(record, cloud, targets)
        center, scale, _ = frame(cloud)
        original = data_module.prepare_cloud

        def prepare(*args, **kwargs):
            result = original(*args, **kwargs)
            old_center, old_scale = result['center'].astype(np.float64), float(result['scale'][0])
            for key in ['points', 'anchors', 'surfels']:
                if key in result:
                    result[key] = ((result[key].astype(np.float64) * old_scale + old_center - center) / scale).astype(np.float32)
            result['center'] = np.asarray(center, dtype=np.float32)
            result['scale'] = np.asarray([scale], dtype=np.float32)
            return result

        # DataLoader workers process samples serially in separate processes. The native
        # loader subsequently computes symmetry candidates and loss eligibility in this frame.
        data_module.prepare_cloud = prepare
        try:
            return super()._sample(record)
        finally:
            data_module.prepare_cloud = original


def dataset(records, config, seed, arm, *, training=True):
    cls = CommonFrameDataset if arm == 'masked_common_frame' else GeometryDataset
    return cls(records, config.model, config.data, seed, training=training)


def collate_fixed(samples):
    batch = collate_clouds(samples)
    for keys, size in [(('points', 'normals', 'point_weights'), 4096),
                       (('anchors', 'anchor_normals', 'anchor_weights'), 256),
                       (('surfels', 'surfel_normals', 'surfel_weights'), 2048)]:
        for key in keys:
            if key not in batch:
                continue
            value = batch[key]
            missing = size - value.shape[1]
            assert missing >= 0
            if missing:
                batch[key] = torch.cat([value, value.new_zeros(value.shape[0], missing, *value.shape[2:])], dim=1)
    if 'symmetry_local' in batch:
        value = batch['symmetry_local']
        missing = 2048 - value.shape[2]
        assert missing >= 0
        if missing:
            batch['symmetry_local'] = torch.cat([value, value.new_zeros(*value.shape[:2], missing)], dim=2)
    return batch


