"""Tables (results/results.md, results/summary.json) and figures (results/figures/*.png)."""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_split  # noqa: E402
from src.metrics import evaluate  # noqa: E402

RUNS, FIG = Path("results/runs"), Path("results/figures")
KEYS = ["precision", "recall", "f1", "iou", "f1_tol2"]
HEADER = ["| Model | Precision | Recall | F1 | IoU | F1 (2-px tol.) |", "|---|---|---|---|---|---|"]
BASE_C, PROP_C = "#8a8f98", "#2a6fdb"


def load(tag):
    return json.loads((RUNS / tag / "metrics.json").read_text())


def tuned_threshold(tag, val_gt, test_gt):
    """Add-on: threshold chosen on val probs (grid 0.05..0.95), applied to test."""
    pr = np.load(RUNS / tag / "probs.npz")
    grid = np.arange(0.05, 0.96, 0.05)
    t = grid[int(np.argmax([evaluate(pr["val"] >= t, val_gt)["f1"] for t in grid]))]
    return evaluate(pr["test"] >= t, test_gt)


def agg(rows):
    return {k: (float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows]))) for k in KEYS}


def fmt(a):
    return " | ".join(f"{a[k][0]:.3f} ± {a[k][1]:.3f}" for k in KEYS)


def main():
    sel = json.loads(Path("results/selection.json").read_text())
    W, S, ws = sel["w_grid"], sel["seeds"], sel["w_star"]
    images, masks, stems, split = load_split()
    val_gt, test_gt = masks[split["val"]], masks[split["test"]]
    FIG.mkdir(parents=True, exist_ok=True)
    md = []

    # 1. validation sweep
    md += [f"## Weight selection on the validation split (mean ± std over seeds {S})", "",
           HEADER[0].replace("Model", "w"), HEADER[1]]
    sweep = {w: agg([load(f"wce_w{w}_s{s}")["val"] for s in S]) for w in W}
    for w in W:
        star = " **(w\\*)**" if w == ws else ""
        md.append(f"| {w}{star} | {fmt(sweep[w])} |")

    # 2. test comparison (the assignment's required table)
    base_name, prop_name = "Baseline: ordinary CE (w = 1)", f"Proposed: weighted CE (w = {ws})"
    rows = {base_name: [load(f"wce_w1_s{s}")["test"] for s in S],
            prop_name: [load(f"wce_w{ws}_s{s}")["test"] for s in S]}
    md += ["", "## Test split: baseline vs weighted cross-entropy (ablation = weight set back to 1)", "", *HEADER]
    md += [f"| {k} | {fmt(agg(v))} |" for k, v in rows.items()]
    md += ["", "Per-seed test scores:", ""]
    for s in S:
        b, p = load(f"wce_w1_s{s}")["test"], load(f"wce_w{ws}_s{s}")["test"]
        md.append(f"- seed {s}: w=1 F1 {b['f1']:.3f} / IoU {b['iou']:.3f} -> "
                  f"w={ws} F1 {p['f1']:.3f} / IoU {p['iou']:.3f}")

    # 3. add-ons
    extra = {
        "Ordinary CE + threshold tuned on val": [tuned_threshold(f"wce_w1_s{s}", val_gt, test_gt) for s in S],
        f"Weighted CE (w = {ws}) + threshold tuned on val":
            [tuned_threshold(f"wce_w{ws}_s{s}", val_gt, test_gt) for s in S],
        f"Weighted CE (w = {ws}) + 8x flip/rot TTA": [load(f"wce_w{ws}_s{s}")["test_tta"] for s in S],
        "CE + Dice (w = 1)": [load(f"wce_dice_w1_s{s}")["test"] for s in S],
        f"Weighted CE + Dice (w = {ws})": [load(f"wce_dice_w{ws}_s{s}")["test"] for s in S],
        f"Weighted CE + Dice (w = {ws}) + TTA": [load(f"wce_dice_w{ws}_s{s}")["test_tta"] for s in S],
    }
    md += ["", "## Add-on experiments (beyond the assignment; test split, threshold 0.5 unless stated)", "", *HEADER]
    md += [f"| {k} | {fmt(agg(v))} |" for k, v in {**rows, **extra}.items()]
    Path("results/results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    Path("results/summary.json").write_text(json.dumps(
        {k: agg(v) for k, v in {**rows, **extra}.items()}, indent=1))

    # Figure 1: test metrics bar chart
    a, b = agg(rows[base_name]), agg(rows[prop_name])
    x = np.arange(4)
    fig, ax = plt.subplots(figsize=(7, 4))
    for off, d, c, lab in [(-0.2, a, BASE_C, "Ordinary CE (w=1)"), (0.2, b, PROP_C, f"Weighted CE (w={ws})")]:
        bars = ax.bar(x + off, [d[k][0] for k in KEYS[:4]], 0.4, yerr=[d[k][1] for k in KEYS[:4]],
                      color=c, label=lab, capsize=3)
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
    ax.set_xticks(x, ["Precision", "Recall", "F1", "IoU"])
    ax.set_ylim(0, 1)
    ax.set_ylabel(f"Test score (mean ± std, {len(S)} seeds)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    ax.set_title("CrackForest test split")
    fig.tight_layout()
    fig.savefig(FIG / "test_metrics.png", dpi=150)
    plt.close(fig)

    # Figure 2: validation sweep over w
    fig, ax = plt.subplots(figsize=(7, 4))
    for k, c, lab in [("precision", "#d9822b", "Precision"), ("recall", "#2a9d8f", "Recall"),
                      ("f1", PROP_C, "F1"), ("iou", "#6b4fbb", "IoU")]:
        ax.errorbar(W, [sweep[w][k][0] for w in W], yerr=[sweep[w][k][1] for w in W],
                    marker="o", color=c, label=lab, capsize=3)
    ax.axvline(ws, ls="--", color="#555", lw=1)
    ax.text(ws, 0.03, f" w*={ws}", fontsize=9)
    ax.set_xscale("log")
    ax.set_xticks(W, [str(w) for w in W])
    ax.set_ylim(0, 1)
    ax.set_xlabel("crack weight w (log scale)")
    ax.set_ylabel("Validation score")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="lower right")
    ax.set_title("Choosing w on the validation split")
    fig.tight_layout()
    fig.savefig(FIG / "val_weight_sweep.png", dpi=150)
    plt.close(fig)

    # Figure 3: validation F1 during training (seed 0)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for tag, c, lab in [("wce_w1_s0", BASE_C, "w=1"), (f"wce_w{ws}_s0", PROP_C, f"w={ws}")]:
        h = json.loads((RUNS / tag / "history.json").read_text())
        ax.plot([e["epoch"] for e in h], [e["val_f1"] for e in h], color=c, label=lab)
    ax.set_xlabel("epoch")
    ax.set_ylabel("val F1")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "val_f1_curves.png", dpi=150)
    plt.close(fig)

    # Figure 4: test masks (seed 0), first 5 test images by file name -> not cherry-picked
    pb = np.load(RUNS / "wce_w1_s0" / "probs.npz")["test"] >= 0.5
    pw = np.load(RUNS / f"wce_w{ws}_s0" / "probs.npz")["test"] >= 0.5
    n = 5
    fig, axs = plt.subplots(n, 4, figsize=(10, 2.6 * n))
    for r in range(n):
        gi = split["test"][r]
        g = masks[gi].astype(bool)
        panels = [(images[gi], f"{stems[gi]}.jpg"), (g, "ground truth")]
        for p, lab in [(pb[r], "ordinary CE (w=1)"), (pw[r], f"weighted CE (w={ws})")]:
            ov = np.full((*g.shape, 3), 0.12)
            ov[p & g] = (0.2, 0.85, 0.3)
            ov[p & ~g] = (0.95, 0.25, 0.25)
            ov[~p & g] = (0.3, 0.55, 1.0)
            s = evaluate(p[None], g[None])
            panels.append((ov, f"{lab}\nF1 {s['f1']:.2f}  IoU {s['iou']:.2f}"))
        for c, (im, t) in enumerate(panels):
            axs[r, c].imshow(im, cmap="gray" if im.ndim == 2 else None)
            axs[r, c].set_title(t, fontsize=8)
            axs[r, c].axis("off")
    fig.suptitle("Test masks: green = correct crack, red = false crack, blue = missed crack", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "test_masks.png", dpi=120)
    plt.close(fig)
    print("\n".join(md))


if __name__ == "__main__":
    main()
