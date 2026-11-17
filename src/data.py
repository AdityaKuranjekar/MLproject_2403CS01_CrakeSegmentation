"""Dataset over the cached 256x256 arrays. Augmentation (train only, identical
for every model): random horizontal/vertical flip and 90-degree rotation, which
keep thin cracks pixel-exact (no interpolation)."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def load_split(data_dir="data"):
    d = Path(data_dir)
    arr = np.load(d / "cfd256.npz")
    split = json.loads((d / "split.json").read_text())
    return arr["images"], arr["masks"], arr["stems"], split


class CrackDataset(Dataset):
    def __init__(self, images, masks, idx, augment=False, seed=0):
        self.images, self.masks, self.idx = images, masks, list(idx)
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        img = (self.images[self.idx[i]].astype(np.float32) / 255 - MEAN) / STD
        m = self.masks[self.idx[i]].astype(np.float32)
        if self.augment:
            if self.rng.random() < 0.5:
                img, m = img[:, ::-1], m[:, ::-1]
            if self.rng.random() < 0.5:
                img, m = img[::-1], m[::-1]
            k = int(self.rng.integers(4))
            img, m = np.rot90(img, k), np.rot90(m, k)
        img = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
        m = torch.from_numpy(np.ascontiguousarray(m))[None]
        return img, m
