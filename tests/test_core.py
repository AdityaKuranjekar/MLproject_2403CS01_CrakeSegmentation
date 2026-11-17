"""Run: python -m pytest -q tests   (data tests skip until prepare_data has run)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.losses import weighted_bce, soft_dice  # noqa: E402
from src.metrics import evaluate, scores, counts  # noqa: E402
from src.model import UNet  # noqa: E402


def test_weighted_bce_matches_written_formula():
    torch.manual_seed(0)
    lo, y = torch.randn(2, 1, 8, 8), (torch.rand(2, 1, 8, 8) > 0.8).float()
    p = torch.sigmoid(lo)
    for w in (1.0, 7.0):
        manual = -(w * y * torch.log(p) + (1 - y) * torch.log(1 - p)).mean()
        assert torch.allclose(weighted_bce(lo, y, w), manual, atol=1e-5)


def test_w1_is_ordinary_cross_entropy():
    lo, y = torch.randn(1, 1, 4, 4), (torch.rand(1, 1, 4, 4) > 0.5).float()
    two_class = torch.cat([torch.zeros_like(lo), lo], 1)  # softmax([0,z]) == sigmoid(z)
    ce = torch.nn.functional.cross_entropy(two_class, y[:, 0].long())
    assert torch.allclose(weighted_bce(lo, y, 1.0), ce, atol=1e-5)


def test_dice_perfect_and_empty():
    y = torch.zeros(1, 1, 4, 4); y[..., 1, :] = 1
    assert soft_dice(y * 40 - 20, y) < 1e-3
    assert soft_dice(-20 * torch.ones_like(y), y) > 0.7


def test_metrics_on_toy_masks():
    gt = np.zeros((1, 4, 4), int); gt[0, 0, :] = 1          # 4 crack px
    pred = np.zeros_like(gt); pred[0, 0, :2] = 1; pred[0, 1, 0] = 1  # 2 TP, 1 FP, 2 FN
    assert counts(pred, gt) == (2, 1, 2)
    s = evaluate(pred, gt)
    assert s["precision"] == pytest.approx(2 / 3) and s["recall"] == pytest.approx(0.5)
    assert s["f1"] == pytest.approx(4 / 7) and s["iou"] == pytest.approx(0.4)
    assert s["f1_tol2"] >= s["f1"]


def test_scores_zero_safe():
    assert scores(0, 0, 0) == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "iou": 0.0}


def test_unet_shape():
    assert UNet()(torch.zeros(1, 3, 256, 256)).shape == (1, 1, 256, 256)


def test_split_and_resize():
    split_file, npz = ROOT / "data/split.json", ROOT / "data/cfd256.npz"
    if not split_file.exists():
        pytest.skip("run scripts/prepare_data.py first")
    s = json.loads(split_file.read_text())
    tr, va, te = map(set, (s["train"], s["val"], s["test"]))
    assert not (tr & va or tr & te or va & te)
    assert len(tr | va | te) == 118 and (len(tr), len(va), len(te)) == (82, 18, 18)
    d = np.load(npz)
    assert d["images"].shape == (118, 256, 256, 3) and d["masks"].shape == (118, 256, 256)
    assert set(np.unique(d["masks"])) <= {0, 1} and (d["masks"].reshape(118, -1).sum(1) > 0).all()
    from prepare_data import make_split
    assert make_split(118) == (s["train"], s["val"], s["test"])  # reproducible
