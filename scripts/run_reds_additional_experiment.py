import argparse
import csv
import os
import time
from pathlib import Path

import cv2
import lpips
import matplotlib
import numpy as np
import torch
from PIL import Image, ImageDraw
from pytorch_fid.fid_score import calculate_fid_given_paths
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_part1 import load_srcnn, process_sequence as run_part1_sequence
from run_part2_basicvsrpp import load_basicvsrpp
from run_part2_basicvsr import process_synthetic_sequence as run_vsr_sequence
from run_part2_realesrgan import build_upsampler, process_synthetic_sequence as run_realesrgan_sequence
from run_part3_adaptive_hybrid import process_sequence as run_hybrid_sequence


REDS_ROOT = Path("data/sample/REDS-sample/REDS-sample")
METHOD_ORDER = ["bicubic", "lanczos", "srcnn", "temporal", "real_esrgan", "basicvsrpp", "adaptive_hybrid"]
METHOD_LABELS = {
    "bicubic": "Bicubic",
    "lanczos": "Lanczos",
    "srcnn": "SRCNN",
    "temporal": "Temporal Avg.",
    "real_esrgan": "Real-ESRGAN",
    "basicvsrpp": "BasicVSR++",
    "adaptive_hybrid": "Adaptive Hybrid",
}


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def list_images(frame_dir):
    return sorted(p for p in Path(frame_dir).iterdir() if p.suffix.lower() == ".png" and not p.name.startswith("._"))


def load_rgb(path):
    return Image.open(path).convert("RGB")


def pil_to_bgr(image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def save_video_from_dir(frame_dir, output_path, fps=24):
    frames = [pil_to_bgr(load_rgb(path)) for path in list_images(frame_dir)]
    if not frames:
        return
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()


def tensor_for_lpips(image, device):
    arr = np.array(image).astype(np.float32) / 127.5 - 1.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def calculate_psnr_ssim(gt_dir, pred_dir):
    psnr_values, ssim_values = [], []
    for gt_path in list_images(gt_dir):
        pred_path = Path(pred_dir) / gt_path.name
        if not pred_path.exists():
            continue
        gt = np.array(load_rgb(gt_path))
        pred = np.array(load_rgb(pred_path).resize((gt.shape[1], gt.shape[0]), Image.BICUBIC))
        psnr_values.append(peak_signal_noise_ratio(gt, pred, data_range=255))
        ssim_values.append(structural_similarity(gt, pred, data_range=255, channel_axis=-1))
    if not psnr_values:
        return None
    return float(np.mean(psnr_values)), float(np.mean(ssim_values)), len(psnr_values)


def calculate_lpips_and_tlpips(gt_dir, pred_dir, loss_fn, device, max_metric_frames=None):
    gt_paths = list_images(gt_dir)
    if max_metric_frames:
        gt_paths = gt_paths[:max_metric_frames]

    lpips_values = []
    pred_pair_values = []
    gt_pair_values = []
    previous_pred = None
    previous_gt = None

    for gt_path in gt_paths:
        pred_path = Path(pred_dir) / gt_path.name
        if not pred_path.exists():
            continue
        gt_img = load_rgb(gt_path)
        pred_img = load_rgb(pred_path).resize(gt_img.size, Image.BICUBIC)
        gt_tensor = tensor_for_lpips(gt_img, device)
        pred_tensor = tensor_for_lpips(pred_img, device)

        with torch.no_grad():
            lpips_values.append(float(loss_fn(pred_tensor, gt_tensor).item()))
            if previous_pred is not None and previous_gt is not None:
                pred_pair_values.append(float(loss_fn(pred_tensor, previous_pred).item()))
                gt_pair_values.append(float(loss_fn(gt_tensor, previous_gt).item()))

        previous_pred = pred_tensor
        previous_gt = gt_tensor

    # Warpless tLPIPS proxy: compare how much consecutive-frame perceptual change
    # differs from the GT sequence. Lower means less extra flicker beyond true motion.
    if pred_pair_values and gt_pair_values:
        tlpips = float(np.mean(np.abs(np.array(pred_pair_values) - np.array(gt_pair_values))))
    else:
        tlpips = None
    return float(np.mean(lpips_values)) if lpips_values else None, tlpips


def calculate_fid(gt_dir, pred_dir, device, max_fid_frames=None):
    # FID is computed between GT frames and restored frames. It is more stable with
    # many images, but still useful here as a supplementary perceptual indicator.
    def run_with_retry(paths):
        last_error = None
        for attempt in range(3):
            try:
                return float(calculate_fid_given_paths(paths, batch_size=16, device=device, dims=2048, num_workers=0))
            except Exception as exc:
                last_error = exc
                print(f"[warn] FID failed on attempt {attempt + 1}/3: {exc}")
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"FID calculation failed after retries: {last_error}") from last_error

    if max_fid_frames is None:
        return run_with_retry([str(gt_dir), str(pred_dir)])

    tmp_root = Path("results/reds_additional/_fid_tmp")
    gt_tmp = tmp_root / "gt"
    pred_tmp = tmp_root / "pred"
    if gt_tmp.exists():
        for p in gt_tmp.glob("*.png"):
            p.unlink()
    if pred_tmp.exists():
        for p in pred_tmp.glob("*.png"):
            p.unlink()
    ensure_dir(gt_tmp)
    ensure_dir(pred_tmp)
    for gt_path in list_images(gt_dir)[:max_fid_frames]:
        pred_path = Path(pred_dir) / gt_path.name
        if not pred_path.exists():
            continue
        load_rgb(gt_path).save(gt_tmp / gt_path.name)
        load_rgb(pred_path).resize(load_rgb(gt_path).size, Image.BICUBIC).save(pred_tmp / gt_path.name)
    return run_with_retry([str(gt_tmp), str(pred_tmp)])


def metric_rows_for_sequence(sequence_name, gt_dir, method_dirs, loss_fn, device, max_metric_frames, max_fid_frames):
    rows = []
    for method, pred_dir in method_dirs.items():
        psnr_ssim = calculate_psnr_ssim(gt_dir, pred_dir)
        if psnr_ssim is None:
            continue
        psnr, ssim, frames = psnr_ssim
        lpips_value, tlpips_value = calculate_lpips_and_tlpips(gt_dir, pred_dir, loss_fn, device, max_metric_frames)
        fid_value = calculate_fid(gt_dir, pred_dir, device, max_fid_frames)
        rows.append({
            "sequence": sequence_name,
            "method": method,
            "method_label": METHOD_LABELS[method],
            "frames": frames,
            "psnr": psnr,
            "ssim": ssim,
            "lpips": lpips_value,
            "fid": fid_value,
            "tlpips": tlpips_value,
        })
    return rows


def write_metrics(rows, output_path):
    ensure_dir(Path(output_path).parent)
    fields = ["sequence", "method", "method_label", "frames", "psnr", "ssim", "lpips", "fid", "tlpips"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            formatted = row.copy()
            for key in ["psnr", "ssim", "lpips", "fid", "tlpips"]:
                if formatted[key] is not None:
                    formatted[key] = f"{formatted[key]:.6f}"
            writer.writerow(formatted)


def plot_metric(rows, metric, output_path):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    sequences = sorted(set(row["sequence"] for row in rows))
    xs = np.arange(len(METHOD_ORDER))
    for sequence in sequences:
        vals = []
        for method in METHOD_ORDER:
            match = [r for r in rows if r["sequence"] == sequence and r["method"] == method]
            vals.append(match[0][metric] if match else np.nan)
        ax.plot(xs, vals, marker="o", linewidth=2.0, label=sequence)
    ax.set_xticks(xs)
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHOD_ORDER], rotation=35, ha="right")
    ax.set_title(f"REDS additional {metric.upper()} comparison")
    ax.set_ylabel(metric.upper())
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(title="Sequence")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def draw_flowchart(output_path):
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.axis("off")
    boxes = [
        ("REDS HR frames", 0.05, 0.55),
        ("Synthetic x4 LR", 0.23, 0.55),
        ("Part 1\nBicubic/Lanczos/SRCNN/Temporal", 0.41, 0.72),
        ("Part 2\nReal-ESRGAN/BasicVSR++", 0.41, 0.38),
        ("Part 3\nAdaptive Hybrid", 0.64, 0.55),
        ("Metrics + Figures\nPSNR/SSIM/LPIPS/FID/tLPIPS", 0.84, 0.55),
    ]
    for text, x, y in boxes:
        ax.text(x, y, text, ha="center", va="center", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.45", fc="#f3f6fb", ec="#38598a", lw=1.4))
    arrows = [((0.13, 0.55), (0.18, 0.55)), ((0.31, 0.55), (0.36, 0.70)),
              ((0.31, 0.55), (0.36, 0.40)), ((0.52, 0.72), (0.59, 0.57)),
              ((0.52, 0.38), (0.59, 0.53)), ((0.72, 0.55), (0.78, 0.55))]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=1.6, color="#38598a"))
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def make_zoom_patch_figure(sequence_name, gt_dir, method_dirs, output_path, frame_name=None, patch_size=56):
    image_names = [p.name for p in list_images(gt_dir)]
    frame_name = frame_name or image_names[len(image_names) // 2]
    gt = load_rgb(Path(gt_dir) / frame_name)

    # Choose an informative crop around the strongest GT edge region.
    gray = cv2.cvtColor(np.array(gt), cv2.COLOR_RGB2GRAY)
    edge = cv2.Sobel(gray, cv2.CV_32F, 1, 1, ksize=3)
    y, x = np.unravel_index(np.argmax(np.abs(edge)), edge.shape)
    x = int(np.clip(x - patch_size // 2, 0, gt.width - patch_size))
    y = int(np.clip(y - patch_size // 2, 0, gt.height - patch_size))
    crop_box = (x, y, x + patch_size, y + patch_size)

    panels = [("GT", gt)]
    for method in METHOD_ORDER:
        pred_dir = method_dirs.get(method)
        if pred_dir is None:
            continue
        pred = load_rgb(Path(pred_dir) / frame_name).resize(gt.size, Image.BICUBIC)
        panels.append((METHOD_LABELS[method], pred))

    zoom_scale = 3
    tile_w = patch_size * zoom_scale
    title_h = 28
    canvas = Image.new("RGB", (tile_w * len(panels), tile_w + title_h), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (label, image) in enumerate(panels):
        patch = image.crop(crop_box).resize((tile_w, tile_w), Image.NEAREST)
        canvas.paste(patch, (idx * tile_w, title_h))
        draw.text((idx * tile_w + 6, 7), label, fill=(0, 0, 0))
    ensure_dir(Path(output_path).parent)
    canvas.save(output_path)


def make_rendering_comparison(sequence_name, gt_dir, method_dirs, output_path, frame_name=None):
    image_names = [p.name for p in list_images(gt_dir)]
    frame_name = frame_name or image_names[len(image_names) // 2]
    gt = load_rgb(Path(gt_dir) / frame_name)
    labels = ["GT"]
    images = [gt]
    for method in METHOD_ORDER:
        pred_dir = method_dirs.get(method)
        if pred_dir is None:
            continue
        images.append(load_rgb(Path(pred_dir) / frame_name).resize(gt.size, Image.BICUBIC))
        labels.append(METHOD_LABELS[method])

    title_h = 28
    w, h = gt.size
    canvas = Image.new("RGB", (w * len(images), h + title_h), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (label, image) in enumerate(zip(labels, images)):
        canvas.paste(image, (idx * w, title_h))
        draw.text((idx * w + 6, 7), label, fill=(0, 0, 0))
    ensure_dir(Path(output_path).parent)
    canvas.save(output_path)


def build_method_dirs(output_root, sequence_name):
    return {
        "bicubic": Path(output_root) / "part1" / sequence_name / "bicubic",
        "lanczos": Path(output_root) / "part1" / sequence_name / "lanczos",
        "srcnn": Path(output_root) / "part1" / sequence_name / "srcnn",
        "temporal": Path(output_root) / "part1" / sequence_name / "temporal",
        "real_esrgan": Path(output_root) / "part2" / "real_esrgan" / sequence_name / "frames",
        "basicvsrpp": Path(output_root) / "part2" / "basicvsrpp" / sequence_name / "frames",
        "adaptive_hybrid": Path(output_root) / "part3" / sequence_name / "frames",
    }


def main():
    parser = argparse.ArgumentParser(description="Run additional REDS experiments and report visualizations.")
    parser.add_argument("--sequences", nargs="+", default=["007", "010", "012"])
    parser.add_argument("--output", default="results/reds_additional")
    parser.add_argument("--max-metric-frames", type=int, default=30)
    parser.add_argument("--max-fid-frames", type=int, default=50)
    args = parser.parse_args()

    output_root = Path(args.output)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    srcnn = load_srcnn("srcnn_weights.pth", device)
    basicvsrpp = load_basicvsrpp("weights/basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth", device)
    realesrgan = build_upsampler(
        "weights/RealESRGAN_x4plus.pth",
        tile=256,
        half=device.type == "cuda",
        device=device,
    )
    lpips_loss = lpips.LPIPS(net="alex").to(device).eval()

    rows = []
    for seq in args.sequences:
        sequence_name = f"REDS_{seq}"
        gt_dir = REDS_ROOT / seq
        if not gt_dir.exists():
            print(f"[skip] missing REDS sequence: {gt_dir}")
            continue

        run_part1_sequence(sequence_name, gt_dir, output_root / "part1", 4, srcnn, device, real_input=False)
        run_realesrgan_sequence(sequence_name, gt_dir, output_root / "part2", 4, realesrgan)
        run_vsr_sequence(
            sequence_name,
            gt_dir,
            output_root / "part2",
            4,
            basicvsrpp,
            device,
            method_dir="basicvsrpp",
            method_label="BasicVSR++",
        )
        run_hybrid_sequence(
            sequence_name,
            output_root / "part2" / "basicvsrpp" / sequence_name / "frames",
            output_root / "part2" / "real_esrgan" / sequence_name / "frames",
            output_root / "part3",
            gt_dir=gt_dir,
        )

        method_dirs = build_method_dirs(output_root, sequence_name)
        rows.extend(metric_rows_for_sequence(sequence_name, gt_dir, method_dirs, lpips_loss, device, args.max_metric_frames, args.max_fid_frames))

        for method, pred_dir in method_dirs.items():
            save_video_from_dir(pred_dir, output_root / "videos" / sequence_name / f"{method}.mp4")

        make_rendering_comparison(sequence_name, gt_dir, method_dirs, output_root / "figures" / f"{sequence_name}_rendering_comparison.png")
        make_zoom_patch_figure(sequence_name, gt_dir, method_dirs, output_root / "figures" / f"{sequence_name}_zoom_patches.png")

    write_metrics(rows, output_root / "tables" / "reds_additional_metrics.csv")
    for metric in ["psnr", "ssim", "lpips", "fid", "tlpips"]:
        plot_metric(rows, metric, output_root / "figures" / f"reds_additional_{metric}.png")
    draw_flowchart(output_root / "figures" / "pipeline_flowchart.png")
    print(f"[done] REDS additional results saved to {output_root}")


if __name__ == "__main__":
    main()
