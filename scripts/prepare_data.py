"""Build the 256x256 CrackForest (CFD) dataset and a fixed train/val/test split.

CFD ships 156 images but only 118 ground-truth masks (`groundTruth/NNN.mat`);
only image/mask pairs are used. Each .mat holds `groundTruth.Segmentation`
with label 1 = pavement and 2 = crack. 22 masks also carry label 3 (whole
pavement blocks enclosed by a crack network, e.g. 093/097/101, up to 45k px)
and 5 of them a rare label 4 (~1.3k px in total over the dataset). Neither is a
crack line, so crack = (label == 2), the crack-line annotation.

Resizing: images use bilinear interpolation. Masks are resized with *area*
averaging and then thresholded at 0.25, not nearest-neighbour: CFD cracks are
1-3 px wide, and shrinking 480x320 -> 256x256 with nearest-neighbour drops
whole pixel columns and breaks thin cracks into dashes.

Split (seed 42, 70/15/15 of 118): 82 train / 18 val / 18 test, saved to
data/split.json and used for every model.

Usage: python scripts/prepare_data.py [--cfd data/raw/CFD] [--out data]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat

SIZE = 256
MASK_THRESHOLD = 0.25
SPLIT_SEED = 42


def load_mask(mat_path: Path) -> np.ndarray:
    gt = loadmat(mat_path)["groundTruth"]
    seg = gt[0, 0]["Segmentation"]
    return (seg == 2).astype(np.float32)  # 1 = crack


def resize_mask(mask: np.ndarray) -> np.ndarray:
    img = Image.fromarray(mask).resize((SIZE, SIZE), Image.BOX)  # area average
    return (np.asarray(img) >= MASK_THRESHOLD).astype(np.uint8)


def make_split(n: int, seed: int = SPLIT_SEED):
    idx = np.random.default_rng(seed).permutation(n)
    n_val = n_test = round(0.15 * n)
    test, val, train = idx[:n_test], idx[n_test:n_test + n_val], idx[n_test + n_val:]
    return sorted(train.tolist()), sorted(val.tolist()), sorted(test.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfd", default="data/raw/CFD")
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    cfd, out = Path(args.cfd), Path(args.out)

    stems = sorted(p.stem for p in (cfd / "groundTruth").glob("*.mat"))
    stems = [s for s in stems if (cfd / "image" / f"{s}.jpg").exists()]
    images, masks, raw_fg = [], [], []
    for s in stems:
        img = Image.open(cfd / "image" / f"{s}.jpg").convert("RGB")
        m = load_mask(cfd / "groundTruth" / f"{s}.mat")
        assert m.shape == (img.height, img.width), (s, m.shape, img.size)
        raw_fg.append(m.mean())
        images.append(np.asarray(img.resize((SIZE, SIZE), Image.BILINEAR), dtype=np.uint8))
        masks.append(resize_mask(m))
    images, masks = np.stack(images), np.stack(masks)

    train, val, test = make_split(len(stems))
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "cfd256.npz", images=images, masks=masks, stems=np.array(stems))
    split = {"seed": SPLIT_SEED, "train": train, "val": val, "test": test,
             "stems": {k: [stems[i] for i in v] for k, v in
                       [("train", train), ("val", val), ("test", test)]}}
    (out / "split.json").write_text(json.dumps(split, indent=1))

    stats = {
        "n_pairs": len(stems),
        "n_train/val/test": [len(train), len(val), len(test)],
        "crack_pixel_fraction_raw": float(np.mean(raw_fg)),
        "crack_pixel_fraction_256": float(masks.mean()),
        "neg_to_pos_ratio_train": float((1 - masks[train].mean()) / masks[train].mean()),
    }
    (out / "dataset_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
