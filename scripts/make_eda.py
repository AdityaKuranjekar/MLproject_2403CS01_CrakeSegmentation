"""EDA figures + fishbone (cause-and-effect) diagram -> results/figures/.

Usage: python scripts/make_eda.py
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import label  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_split  # noqa: E402

FIG = Path("results/figures")


def eda():
    images, masks, stems, split = load_split()
    frac = masks.reshape(len(masks), -1).mean(1) * 100
    comps = np.array([label(m, structure=np.ones((3, 3)))[1] for m in masks])
    stats = {
        "crack_percent_mean": float(frac.mean()), "crack_percent_min": float(frac.min()),
        "crack_percent_max": float(frac.max()),
        "components_per_mask_median": float(np.median(comps)),
        "per_split_crack_percent": {k: float(masks[split[k]].mean() * 100) for k in ("train", "val", "test")},
        "mean_rgb": images.reshape(-1, 3).mean(0).round(1).tolist(),
    }
    Path("results/eda.json").write_text(json.dumps(stats, indent=1))

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
    axs[0].hist(frac, bins=25, color="#2a6fdb")
    axs[0].axvline(frac.mean(), color="#333", ls="--", lw=1)
    axs[0].set_xlabel("crack pixels per image (%)")
    axs[0].set_ylabel("images")
    axs[0].set_title(f"Class imbalance: mean {frac.mean():.2f}% crack")
    for k, c in [("train", "#2a6fdb"), ("val", "#2a9d8f"), ("test", "#d9822b")]:
        axs[1].hist(frac[split[k]], bins=15, alpha=0.6, color=c, label=f"{k} ({len(split[k])})")
    axs[1].set_xlabel("crack pixels per image (%)")
    pct = {k: masks[split[k]].mean() * 100 for k in ("train", "val", "test")}
    axs[1].set_title("Per split: " + ", ".join(f"{k} {v:.2f}%" for k, v in pct.items()), fontsize=10)
    axs[1].legend(frameon=False)
    for a in axs:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "eda_crack_fraction.png", dpi=150)
    plt.close(fig)

    order = np.argsort(frac)
    pick = [order[5], order[len(order) // 3], order[2 * len(order) // 3], order[-3]]
    fig, axs = plt.subplots(2, 4, figsize=(11, 5.6))
    for c, i in enumerate(pick):
        axs[0, c].imshow(images[i])
        axs[0, c].set_title(f"{stems[i]}.jpg  ({frac[i]:.1f}% crack)", fontsize=9)
        axs[1, c].imshow(masks[i], cmap="gray")
        for r in (0, 1):
            axs[r, c].axis("off")
    fig.suptitle("CrackForest samples at 256x256 (top) and crack masks (bottom)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "eda_samples.png", dpi=120)
    plt.close(fig)
    print(json.dumps(stats, indent=1))


def fishbone():
    """Ishikawa diagram: why plain CE gives broken / thick cracks."""
    causes = {
        "Data": ["~2% crack pixels (1:48)", "1-3 px wide cracks", "only 118 labelled images"],
        "Loss": ["CE averages over all pixels", "pavement dominates gradient", "no overlap (region) term"],
        "Model": ["pooling blurs thin lines", "random init, no pre-training", "fixed 0.5 threshold"],
        "Preprocessing": ["480x320 -> 256x256 shrink", "nearest resize breaks lines", "shadows / oil stains / texture"],
    }
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.annotate("", xy=(9.6, 3), xytext=(0.4, 3), arrowprops=dict(arrowstyle="-|>", lw=2.2, color="#333"))
    ax.text(9.7, 3, "Broken or\ntoo-thick\ncrack masks", va="center", fontsize=11, weight="bold",
            bbox=dict(boxstyle="round,pad=0.4", fc="#ffe9d6", ec="#d9822b"))
    xs = [2.6, 6.4]
    for j, (name, items) in enumerate(causes.items()):
        top = j % 2 == 0
        x = xs[j // 2]
        y_end = 5.4 if top else 0.6
        ax.plot([x - 1.4, x], [y_end, 3], color="#2a6fdb", lw=1.8)
        ax.text(x - 1.4, y_end + (0.25 if top else -0.25), name, ha="center", va="center",
                fontsize=11, weight="bold", color="#2a6fdb")
        for k, it in enumerate(items):
            t = (k + 1) / (len(items) + 1)
            px, py = x - 1.4 * (1 - t), y_end + (3 - y_end) * t
            ax.plot([px - 0.35, px], [py, py], color="#888", lw=1)
            ax.text(px - 0.4, py, it, ha="right", va="center", fontsize=8.5)
    ax.set_title("Fishbone: causes of poor thin-crack segmentation; weighted CE targets the 'Loss' branch",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "fishbone.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    eda()
    fishbone()
