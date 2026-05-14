import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


METHOD_ORDER = [
    "bicubic",
    "lanczos",
    "srcnn",
    "temporal",
    "real_esrgan",
    "basicvsr",
    "basicvsrpp",
    "adaptive_hybrid",
]

METHOD_LABELS = {
    "bicubic": "Bicubic",
    "lanczos": "Lanczos",
    "srcnn": "SRCNN",
    "temporal": "Temporal Avg.",
    "real_esrgan": "Real-ESRGAN",
    "basicvsr": "BasicVSR",
    "basicvsrpp": "BasicVSR++",
    "adaptive_hybrid": "Adaptive Hybrid",
}

METRIC_FILES = [
    "results/part1/metrics_part1.csv",
    "results/part2/metrics_part2_all.csv",
    "results/part3/metrics_part3.csv",
]


def load_metrics(metric_files):
    frames = []
    for path in metric_files:
        path = Path(path)
        if not path.exists():
            print(f"[warn] missing metric file: {path}")
            continue
        frame = pd.read_csv(path)
        frame["source_file"] = str(path)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("No metric CSV files were found.")

    data = pd.concat(frames, ignore_index=True)
    data["method"] = pd.Categorical(data["method"], categories=METHOD_ORDER, ordered=True)
    data = data.sort_values(["sequence", "method"]).reset_index(drop=True)
    return data


def style_axis(ax, ylabel):
    ax.set_xlabel("Method")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.tick_params(axis="x", rotation=35)
    ax.legend(title="Dataset")


def plot_metric(data, metric, output_path):
    fig, ax = plt.subplots(figsize=(12, 5.8))
    x_labels = [METHOD_LABELS[m] for m in METHOD_ORDER]
    x_positions = list(range(len(METHOD_ORDER)))

    for sequence, group in data.groupby("sequence", sort=True):
        values = []
        for method in METHOD_ORDER:
            row = group[group["method"] == method]
            values.append(float(row[metric].iloc[0]) if not row.empty else None)
        ax.plot(x_positions, values, marker="o", linewidth=2.2, label=sequence)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, ha="right")
    ax.set_title(f"{metric.upper()} comparison across VSR methods")
    style_axis(ax, metric.upper())
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_combined(data, output_path):
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    x_labels = [METHOD_LABELS[m] for m in METHOD_ORDER]
    x_positions = list(range(len(METHOD_ORDER)))

    for ax, metric in zip(axes, ["psnr", "ssim"]):
        for sequence, group in data.groupby("sequence", sort=True):
            values = []
            for method in METHOD_ORDER:
                row = group[group["method"] == method]
                values.append(float(row[metric].iloc[0]) if not row.empty else None)
            ax.plot(x_positions, values, marker="o", linewidth=2.2, label=sequence)
        ax.set_title(f"{metric.upper()} comparison")
        style_axis(ax, metric.upper())

    axes[-1].set_xticks(x_positions)
    axes[-1].set_xticklabels(x_labels, ha="right")
    fig.suptitle("Video Super-Resolution Metric Trends", fontsize=15, y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_summary_table(data, output_path):
    table = data.copy()
    table["method_label"] = table["method"].map(METHOD_LABELS)
    table = table[["sequence", "method", "method_label", "frames", "psnr", "ssim", "source_file"]]
    table.to_csv(output_path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Generate report-ready metric line charts.")
    parser.add_argument("--output", default="results/figures")
    parser.add_argument("--metrics", nargs="*", default=METRIC_FILES)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_metrics(args.metrics)
    save_summary_table(data, output_dir / "metrics_all_methods.csv")
    plot_metric(data, "psnr", output_dir / "psnr_line_comparison.png")
    plot_metric(data, "ssim", output_dir / "ssim_line_comparison.png")
    plot_combined(data, output_dir / "metric_line_comparison.png")

    print(f"[done] Saved plots and summary CSV to {output_dir}")


if __name__ == "__main__":
    main()
