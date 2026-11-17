"""Full protocol. Resumable: finished runs are skipped.

1. Weight sweep on the VALIDATION split: w in W_GRID x 3 seeds.
2. w* = argmax of mean val F1 (test scores are never looked at here).
3. Test comparison: w = 1 (ordinary CE = baseline = ablation) vs w = w*.
4. Add-on (beyond the assignment): weighted CE + Dice at w = 1 and w*.

Usage: python scripts/run_experiments.py [epochs]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.train import run  # noqa: E402

W_GRID = [1, 2, 5, 10, 20, 50]
SEEDS = [0, 1, 2]
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 150


def main():
    sweep = {}
    for w in W_GRID:
        rs = [run(w=w, seed=s, epochs=EPOCHS) for s in SEEDS]
        sweep[w] = float(np.mean([r["val"]["f1"] for r in rs]))
        print(f"w={w:<3} mean val F1={sweep[w]:.4f}", flush=True)
    w_star = max(sweep, key=sweep.get)
    print("selected w* =", w_star, flush=True)
    for w in sorted({1, w_star}):
        for s in SEEDS:
            run(w=w, seed=s, loss="wce_dice", epochs=EPOCHS)
    Path("results/selection.json").write_text(json.dumps(
        {"w_grid": W_GRID, "seeds": SEEDS, "epochs": EPOCHS,
         "mean_val_f1": {str(k): v for k, v in sweep.items()}, "w_star": w_star}, indent=1))


if __name__ == "__main__":
    main()
