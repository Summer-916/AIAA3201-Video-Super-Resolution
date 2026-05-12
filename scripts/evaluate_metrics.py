import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def list_images(frame_dir):
    return sorted(
        p for p in Path(frame_dir).iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS and not p.name.startswith("._")
    )


def load_rgb(path):
    return np.array(Image.open(path).convert("RGB"))


def calculate_metrics(gt_dir, pred_dir):
    gt_dir = Path(gt_dir)
    pred_dir = Path(pred_dir)
    psnr_values = []
    ssim_values = []

    for gt_path in list_images(gt_dir):
        pred_path = pred_dir / gt_path.name
        if not pred_path.exists():
            continue
        gt = load_rgb(gt_path)
        pred = load_rgb(pred_path)
        if gt.shape != pred.shape:
            continue
        psnr_values.append(peak_signal_noise_ratio(gt, pred, data_range=255))
        ssim_values.append(structural_similarity(gt, pred, data_range=255, channel_axis=-1))

    if not psnr_values:
        return None
    return {
        "frames": len(psnr_values),
        "psnr": float(np.mean(psnr_values)),
        "ssim": float(np.mean(ssim_values)),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Part 1 outputs with PSNR and SSIM.")
    parser.add_argument("--results", default="results/part1")
    parser.add_argument("--output", default="results/part1/metrics_part1.csv")
    args = parser.parse_args()

    ground_truth = {
        "REDS_002": "data/sample/REDS-sample/REDS-sample/002",
        "Vimeo_00018_0043": "data/sample/vimeo-RL/vimeo-RL/00018/0043",
    }
    methods = ["bicubic", "lanczos", "srcnn", "temporal"]

    rows = []
    for sequence, gt_dir in ground_truth.items():
        for method in methods:
            pred_dir = Path(args.results) / sequence / method
            if not pred_dir.exists():
                continue
            metrics = calculate_metrics(gt_dir, pred_dir)
            if metrics is None:
                continue
            rows.append({
                "sequence": sequence,
                "method": method,
                "frames": metrics["frames"],
                "psnr": f"{metrics['psnr']:.4f}",
                "ssim": f"{metrics['ssim']:.6f}",
            })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sequence", "method", "frames", "psnr", "ssim"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} metric rows to {output}")
    for row in rows:
        print(f"{row['sequence']:18s} {row['method']:9s} PSNR={row['psnr']} SSIM={row['ssim']}")


if __name__ == "__main__":
    main()
