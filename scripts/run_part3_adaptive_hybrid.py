import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def list_images(frame_dir):
    return sorted(
        p for p in Path(frame_dir).iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS and not p.name.startswith("._")
    )


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_rgb(path):
    return Image.open(path).convert("RGB")


def pil_to_bgr(image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def bgr_to_pil(image):
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def save_video(frames, output_path, fps=24):
    if not frames:
        return
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()


def label_strip(images, labels):
    font_h = 28
    widths, heights = zip(*(img.size for img in images))
    canvas = Image.new("RGB", (sum(widths), max(heights) + font_h), "white")
    draw = ImageDraw.Draw(canvas)
    x = 0
    for img, label in zip(images, labels):
        canvas.paste(img, (x, font_h))
        draw.text((x + 8, 6), label, fill=(0, 0, 0))
        x += img.width
    return canvas


def normalize_map(value, percentile=95):
    scale = np.percentile(value, percentile)
    if scale < 1e-6:
        return np.zeros_like(value, dtype=np.float32)
    return np.clip(value / scale, 0.0, 1.0).astype(np.float32)


def edge_strength(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(grad_x, grad_y)


def frame_difference(frames, idx):
    current = frames[idx].astype(np.float32)
    diffs = []
    if idx > 0:
        diffs.append(np.mean(np.abs(current - frames[idx - 1].astype(np.float32)), axis=2))
    if idx < len(frames) - 1:
        diffs.append(np.mean(np.abs(current - frames[idx + 1].astype(np.float32)), axis=2))
    if not diffs:
        return np.zeros(frames[idx].shape[:2], dtype=np.float32)
    return np.mean(diffs, axis=0).astype(np.float32)


def adaptive_mask(stable_frames, detail_frames, idx, base_weight, max_weight):
    stable = stable_frames[idx].astype(np.float32)
    detail = detail_frames[idx].astype(np.float32)

    # Texture encourages Real-ESRGAN contribution; large disagreement discourages it.
    texture = normalize_map(edge_strength(stable_frames[idx]))
    disagreement = normalize_map(np.mean(np.abs(detail - stable), axis=2))
    temporal_motion = normalize_map(frame_difference(stable_frames, idx))

    mask = base_weight
    mask += 0.35 * texture
    mask -= 0.45 * disagreement
    mask -= 0.25 * temporal_motion
    mask = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 1.2)
    return np.clip(mask, 0.0, max_weight).astype(np.float32)


def fuse_frame(stable, detail, mask):
    stable = stable.astype(np.float32)
    detail = detail.astype(np.float32)
    mask3 = mask[:, :, None]
    fused = stable * (1.0 - mask3) + detail * mask3
    return np.clip(fused, 0, 255).astype(np.uint8)


def calculate_metrics(gt_dir, pred_dir):
    psnr_values = []
    ssim_values = []
    for gt_path in list_images(gt_dir):
        pred_path = Path(pred_dir) / gt_path.name
        if not pred_path.exists():
            continue
        gt = np.array(load_rgb(gt_path))
        pred = np.array(load_rgb(pred_path))
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


def common_paths(stable_dir, detail_dir, max_frames=None):
    stable = {p.name: p for p in list_images(stable_dir)}
    detail = {p.name: p for p in list_images(detail_dir)}
    names = sorted(set(stable).intersection(detail))
    if max_frames:
        names = names[:max_frames]
    return [(name, stable[name], detail[name]) for name in names]


def process_sequence(
    name,
    stable_dir,
    detail_dir,
    output_root,
    gt_dir=None,
    max_frames=None,
    base_weight=0.18,
    max_weight=0.65,
):
    pairs = common_paths(stable_dir, detail_dir, max_frames=max_frames)
    if not pairs:
        print(f"[skip] {name}: no common frames")
        return None

    seq_root = Path(output_root) / name
    for subdir in ["frames", "masks", "comparison", "videos"]:
        ensure_dir(seq_root / subdir)

    stable_frames = [np.array(load_rgb(stable_path)) for _, stable_path, _ in pairs]
    detail_frames = [np.array(load_rgb(detail_path).resize((stable_frames[i].shape[1], stable_frames[i].shape[0]), Image.BICUBIC))
                     for i, (_, _, detail_path) in enumerate(pairs)]

    fused_frames = []
    print(f"[run] Part 3 adaptive hybrid {name}: {len(pairs)} frames")
    for idx, (filename, _, _) in enumerate(pairs):
        mask = adaptive_mask(stable_frames, detail_frames, idx, base_weight, max_weight)
        fused = fuse_frame(stable_frames[idx], detail_frames[idx], mask)
        fused_frames.append(fused)

        bgr_to_pil(cv2.applyColorMap((mask * 255).astype(np.uint8), cv2.COLORMAP_TURBO)).save(seq_root / "masks" / filename)
        Image.fromarray(fused).save(seq_root / "frames" / filename)

    sample_indices = sorted(set([0, len(pairs) // 2, len(pairs) - 1]))
    for idx in sample_indices:
        filename = pairs[idx][0]
        tiles = [
            Image.fromarray(stable_frames[idx]),
            Image.fromarray(detail_frames[idx]),
            load_rgb(seq_root / "masks" / filename),
            Image.fromarray(fused_frames[idx]),
        ]
        labels = ["BasicVSR++", "Real-ESRGAN", "Adaptive mask", "Hybrid"]
        if gt_dir is not None:
            gt_path = Path(gt_dir) / filename
            if gt_path.exists():
                tiles.append(load_rgb(gt_path))
                labels.append("GT")
        label_strip(tiles, labels).save(seq_root / "comparison" / filename)

    save_video([pil_to_bgr(Image.fromarray(frame)) for frame in fused_frames], seq_root / "videos" / "adaptive_hybrid.mp4")

    if gt_dir is None:
        return None
    metrics = calculate_metrics(gt_dir, seq_root / "frames")
    if metrics:
        metrics.update({"sequence": name, "method": "adaptive_hybrid"})
    return metrics


def write_metrics(rows, output_path):
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sequence", "method", "frames", "psnr", "ssim"])
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "sequence": row["sequence"],
                "method": row["method"],
                "frames": row["frames"],
                "psnr": f"{row['psnr']:.4f}",
                "ssim": f"{row['ssim']:.6f}",
            })


def main():
    parser = argparse.ArgumentParser(description="Run Part 3 adaptive hybrid refinement.")
    parser.add_argument("--output", default="results/part3")
    parser.add_argument("--part2-root", default="results/part2")
    parser.add_argument("--base-weight", type=float, default=0.18)
    parser.add_argument("--max-weight", type=float, default=0.65)
    parser.add_argument("--max-wild-frames", type=int, default=None)
    args = parser.parse_args()

    part2_root = Path(args.part2_root)
    rows = []

    synthetic = {
        "REDS_002": "data/sample/REDS-sample/REDS-sample/002",
        "Vimeo_00018_0043": "data/sample/vimeo-RL/vimeo-RL/00018/0043",
    }
    for name, gt_dir in synthetic.items():
        metrics = process_sequence(
            name=name,
            stable_dir=part2_root / "basicvsrpp" / name / "frames",
            detail_dir=part2_root / "real_esrgan" / name / "frames",
            output_root=args.output,
            gt_dir=gt_dir,
            base_weight=args.base_weight,
            max_weight=args.max_weight,
        )
        if metrics:
            rows.append(metrics)

    process_sequence(
        name="wild",
        stable_dir=part2_root / "basicvsrpp" / "wild" / "frames",
        detail_dir=part2_root / "real_esrgan" / "wild" / "frames",
        output_root=args.output,
        gt_dir=None,
        max_frames=args.max_wild_frames,
        base_weight=args.base_weight,
        max_weight=args.max_weight,
    )

    metrics_path = Path(args.output) / "metrics_part3.csv"
    write_metrics(rows, metrics_path)
    print(f"[done] Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
