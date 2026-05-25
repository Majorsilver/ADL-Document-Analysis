import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

DATASET_ROOT = Path(__file__).parent / "cross_out_dataset_v2"
OUTPUT_JSON = Path(__file__).parent / "dataset_dimensions.json"
OUTPUT_PLOT = Path(__file__).parent / "dataset_dimensions.png"
BIN_WIDTH = 12  # px per histogram bucket


def collect_dimensions(root: Path) -> dict:
    widths = Counter()
    heights = Counter()
    pairs = Counter()
    total = 0
    errors = []

    png_files = list(root.rglob("*.png"))
    print(f"Found {len(png_files)} PNG files under {root}")

    for i, path in enumerate(png_files, 1):
        try:
            with Image.open(path) as img:
                w, h = img.size
            widths[w] += 1
            heights[h] += 1
            pairs[(w, h)] += 1
            total += 1
        except Exception as e:
            errors.append((str(path), str(e)))

        if i % 5000 == 0:
            print(f"  processed {i}/{len(png_files)}")

    return {
        "total_images": total,
        "widths": dict(widths),
        "heights": dict(heights),
        "width_height_pairs": {f"{w}x{h}": c for (w, h), c in pairs.items()},
        "errors": errors,
    }


def summarize(counts: dict) -> dict:
    total = sum(counts.values())
    mean = sum(int(k) * v for k, v in counts.items()) / total
    items = sorted((int(k), v) for k, v in counts.items())

    half = total / 2
    cum = 0
    median = None
    for idx, (val, c) in enumerate(items):
        cum += c
        if cum >= half:
            if total % 2 == 1 or cum > half:
                median = float(val)
            else:
                median = (val + items[idx + 1][0]) / 2
            break

    return {
        "count": total,
        "mean": mean,
        "median": median,
        "min": items[0][0],
        "max": items[-1][0],
    }


def save_json(stats: dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(stats, f, indent=2, sort_keys=True)
    print(f"Wrote {path}")


def _binned(counts: dict, bin_width: int):
    vals = np.array([int(k) for k in counts], dtype=int)
    freqs = np.array([counts[k] for k in counts], dtype=int)
    lo = (vals.min() // bin_width) * bin_width
    hi = ((vals.max() // bin_width) + 1) * bin_width
    edges = np.arange(lo, hi + bin_width, bin_width)
    bin_idx = (vals - lo) // bin_width
    totals = np.zeros(len(edges) - 1, dtype=int)
    np.add.at(totals, bin_idx, freqs)
    centers = edges[:-1] + bin_width / 2
    return centers, totals, edges


def plot_histograms(stats: dict, summary: dict, path: Path) -> None:
    widths = {int(k): v for k, v in stats["widths"].items()}
    heights = {int(k): v for k, v in stats["heights"].items()}

    fig, axes = plt.subplots(1, 2, figsize=(7, 2.5))

    for ax, data, label, color, stat in (
        (axes[0], widths, "width", "steelblue", summary["width"]),
        (axes[1], heights, "height", "indianred", summary["height"]),
    ):
        centers, totals, _ = _binned(data, BIN_WIDTH)
        ax.bar(centers, totals, width=BIN_WIDTH * 0.95, color=color, align="center")
        ax.axvline(stat["mean"], color="black", linestyle="--", linewidth=1.5,
                   label=f"mean {stat['mean']:.1f}")
        ax.axvline(stat["median"], color="black", linestyle=":", linewidth=1.5,
                   label=f"median {stat['median']:.1f}")
        ax.set_title(f"{label.capitalize()} frequency (bin = {BIN_WIDTH} px)")
        ax.set_xlabel(f"{label} (px)")
        ax.set_ylabel("count")
        ax.legend()

    fig.suptitle(f"Image dimension frequencies ({stats['total_images']} images)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"Wrote {path}")
    plt.show()


def main() -> None:
    if OUTPUT_JSON.exists():
        print(f"Loading cached stats from {OUTPUT_JSON}")
        with open(OUTPUT_JSON) as f:
            stats = json.load(f)
    else:
        stats = collect_dimensions(DATASET_ROOT)

    stats["summary"] = {
        "width": summarize(stats["widths"]),
        "height": summarize(stats["heights"]),
    }
    print("width  summary:", stats["summary"]["width"])
    print("height summary:", stats["summary"]["height"])

    save_json(stats, OUTPUT_JSON)
    plot_histograms(stats, stats["summary"], OUTPUT_PLOT)


if __name__ == "__main__":
    main()
