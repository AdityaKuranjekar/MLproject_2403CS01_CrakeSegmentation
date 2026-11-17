"""Train one U-Net and evaluate it on the fixed val/test split.

Every model shares the same split, architecture, optimiser, schedule, epochs,
augmentation, checkpoint rule (best val F1 at threshold 0.5) and seeds; only
the loss changes.

    python -m src.train --w 1  --seed 0   # baseline: ordinary cross-entropy
    python -m src.train --w 10 --seed 0   # weighted cross-entropy
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data import CrackDataset, load_split
from src.losses import make_loss
from src.metrics import evaluate
from src.model import UNet


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@torch.no_grad()
def predict(model, ds, device, tta=False):
    """Returns P(crack) for every image of `ds`, shape (N,H,W), float32."""
    model.eval()
    out = []
    for img, _ in DataLoader(ds, batch_size=8):
        img = img.to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            if not tta:
                prob = torch.sigmoid(model(img).float())
            else:  # average over the 8 dihedral transforms, each mapped back
                prob = 0
                for k in range(4):
                    for flip in (False, True):
                        x = torch.rot90(img, k, (2, 3))
                        x = torch.flip(x, (3,)) if flip else x
                        y = torch.sigmoid(model(x).float())
                        y = torch.flip(y, (3,)) if flip else y
                        prob = prob + torch.rot90(y, -k, (2, 3))
                prob = prob / 8
        out.append(prob[:, 0].cpu().numpy())
    return np.concatenate(out)


def run(w=1.0, seed=0, loss="wce", epochs=150, lr=1e-3, batch=8, data="data",
        out="results/runs", tag=None, save_probs=True):
    tag = tag or f"{loss}_w{w:g}_s{seed}"
    run_dir = Path(out) / tag
    if (run_dir / "metrics.json").exists():  # resumable sweeps
        return json.loads((run_dir / "metrics.json").read_text())
    run_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images, masks, _, split = load_split(data)
    train_ds = CrackDataset(images, masks, split["train"], augment=True, seed=seed)
    val_ds = CrackDataset(images, masks, split["val"])
    test_ds = CrackDataset(images, masks, split["test"])
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_ds, batch_size=batch, shuffle=True, drop_last=True, generator=g)

    model = UNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(loader))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    criterion = make_loss(loss, w)
    val_gt = masks[split["val"]]

    best, best_state, history, t0 = -1.0, None, [], time.time()
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for img, m in loader:
            img, m = img.to(device), m.to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(img)
            l = criterion(logits.float(), m)
            opt.zero_grad(set_to_none=True)
            scaler.scale(l).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += l.item()
        vf1 = evaluate(predict(model, val_ds, device) >= 0.5, val_gt)["f1"]
        history.append({"epoch": ep + 1, "loss": tot / len(loader), "val_f1": vf1})
        if vf1 > best:
            best, best_state = vf1, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    result = {"tag": tag, "loss": loss, "w": w, "seed": seed, "epochs": epochs,
              "best_epoch": int(np.argmax([h["val_f1"] for h in history]) + 1),
              "train_seconds": round(time.time() - t0, 1)}
    probs = {}
    for name, ds in [("val", val_ds), ("test", test_ds)]:
        gt = masks[split[name]]
        p, p_tta = predict(model, ds, device), predict(model, ds, device, tta=True)
        result[name] = evaluate(p >= 0.5, gt)
        result[name + "_tta"] = evaluate(p_tta >= 0.5, gt)
        probs[name], probs[name + "_tta"] = p.astype(np.float16), p_tta.astype(np.float16)
    if save_probs:
        np.savez_compressed(run_dir / "probs.npz", **probs)
    (run_dir / "history.json").write_text(json.dumps(history))
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=1))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", type=float, default=1.0, help="crack-pixel weight (1 = ordinary CE)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--loss", default="wce", choices=["wce", "wce_dice"])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()
    r = run(a.w, a.seed, a.loss, a.epochs, a.lr)
    print(json.dumps({k: r[k] for k in ("tag", "best_epoch", "val", "test")}, indent=1))
