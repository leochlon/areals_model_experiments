"""DrivAerML surface pressure in the native AeroJEPA dataset interface."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class DrivAerMLDataset(Dataset):
    def __init__(self, root_dir, split, *, target_encoder_points=2048, query_points=2048, seed=0, **unused):
        self.root, self.split, self.seed = Path(root_dir), split, seed
        self.metadata = json.loads((self.root / "data_ready.json").read_text())
        self.records = [x for x in self.metadata["records"] if x["split"] == split]
        self.target_encoder_points, self.query_points, self.epoch = target_encoder_points, query_points, 0
        self.prepared = []
        for row in self.records:
            path = self.root / "data" / (row["id"] + ".npz")
            with np.load(path, allow_pickle=False) as z:
                self.prepared.append({key: z[key] for key in z.files})
        self.mean, self.std = (self.metadata["normalization"][x] for x in ("mean", "std"))

    def __len__(self):
        return len(self.records)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        name, p = self.records[index]["id"], self.prepared[index]
        epoch = self.epoch if self.split == "train" else 0
        seed = int.from_bytes(hashlib.sha256(f"{name}:{self.seed}:{epoch}".encode()).digest()[:8], "little")
        rng = np.random.default_rng(seed)
        weights = p["context_feat"][:, 3].astype(np.float64)
        weights /= weights.sum()
        target_idx = rng.choice(len(weights), min(len(weights), self.target_encoder_points), replace=False, p=weights)
        target = ((p["pressure"] - self.mean) / self.std).astype(np.float32)[:, None]
        if self.split == "train":
            query_idx = rng.choice(len(weights), self.query_points, replace=True, p=weights)
            queries, query_target = p["context_pos"][query_idx], target[query_idx]
        else:
            queries = p["fixed_query_pos"]
            query_target = ((p["fixed_target"] - self.mean) / self.std).astype(np.float32)
        gp = (p["gen_params_raw"] - np.array(self.metadata["gen_params_mean"])) / np.array(self.metadata["gen_params_std"])
        arrays = dict(context_pos=p["context_pos"], context_feat=p["context_feat"],
                      target_surface_pos=p["context_pos"][target_idx],
                      target_surface_main_feat=np.concatenate([p["context_pos"][target_idx], target[target_idx]], -1),
                      target_volume_pos=np.zeros((0, 3), np.float32), target_volume_feat=np.zeros((0, 4), np.float32),
                      query_pos=queries, query_sdf=np.zeros((len(queries), 1), np.float32), query_target=query_target,
                      gen_params=gp.astype(np.float32))
        return dict(case_id=name, **{key: torch.from_numpy(value) for key, value in arrays.items()})
