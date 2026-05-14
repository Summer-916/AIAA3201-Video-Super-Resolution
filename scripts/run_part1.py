import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchvision import transforms
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.srcnn import SRCNN


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def list_images(frame_dir):
    """Return valid image frames in temporal order.

    REDS/Vimeo frame names sort lexicographically in the same order as time.
    Hidden macOS resource-fork files are ignored because they are not images.
    """
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


def resize_pil(image, size, resample):
    return image.resize(size, resample)


def make_lr_and_reference(image, scale, real_input):
    # Synthetic datasets have HR ground truth, so we create LR inputs by
    # downsampling and keep the original HR frame as reference. Wild frames are
    # already treated as real LR inputs and have no GT reference.
    if real_input:
        return image, None
    lr_size = (max(1, image.width // scale), max(1, image.height // scale))
    lr = resize_pil(image, lr_size, Image.BICUBIC)
    return lr, image


def upscale_lr(lr_image, scale, target_size=None, method=Image.BICUBIC):
    # target_size is used on synthetic datasets to exactly match GT dimensions.
    # For real LR input, the requested scale determines the output size.
    size = target_size or (lr_image.width * scale, lr_image.height * scale)
    return resize_pil(lr_image, size, method)


def run_srcnn(model, device, image):
    # SRCNN expects a bicubic-upsampled image and predicts a refined HR image.
    # It is frame-independent and therefore cannot use temporal information.
    tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor).clamp(0.0, 1.0)
    return transforms.ToPILImage()(output.squeeze(0).cpu())


def temporal_average(frames, radius=1, weights=None):
    # A simple no-alignment temporal baseline: average neighboring upsampled frames.
    # This is intentionally naive so its failure under motion can motivate BasicVSR.
    if weights is None:
        weights = np.ones(2 * radius + 1, dtype=np.float32)
    weights = weights / weights.sum()

    output = []
    for idx in range(len(frames)):
        # Boundary frames reuse the nearest valid neighbor. This keeps the
        # output length identical to the input length.
        acc = np.zeros_like(frames[idx], dtype=np.float32)
        norm = 0.0
        for offset in range(-radius, radius + 1):
            src_idx = min(max(idx + offset, 0), len(frames) - 1)
            weight = float(weights[offset + radius])
            acc += frames[src_idx].astype(np.float32) * weight
            norm += weight
        output.append(np.clip(acc / norm, 0, 255).astype(np.uint8))
    return output


def unsharp_mask_bgr(image, amount=0.65, sigma=1.0):
    # Temporal averaging can look overly smooth. A mild unsharp mask makes this
    # baseline easier to inspect without introducing another learned model.
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def save_video(frames, output_path, fps=24):
    # MP4 outputs are for quick viewing and submission. PNG frames are kept as
    # the preferred source for report figures because they avoid compression.
    if not frames:
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()


def label_strip(images, labels):
    # Build compact side-by-side figures used directly in the report.
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


def calculate_metrics(gt_dir, pred_dir):
    # Only synthetic sequences have GT, so PSNR/SSIM is computed for matching files.
    gt_files = list_images(gt_dir)
    pred_dir = Path(pred_dir)
    psnr_values = []
    ssim_values = []
    for gt_path in gt_files:
        # Compare only matching frame names with identical shapes. This prevents
        # accidental metrics on real-input outputs or incomplete folders.
        pred_path = pred_dir / gt_path.name
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


def load_srcnn(weights_path, device):
    # If weights are missing, Part 1 still produces the classical baselines; the
    # SRCNN branch falls back to bicubic output rather than stopping the script.
    model = SRCNN().to(device)
    if not Path(weights_path).exists():
        print(f"[warn] SRCNN weights not found: {weights_path}. Skipping SRCNN output.")
        return None
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model


def process_sequence(name, input_dir, output_root, scale, model, device, real_input, max_frames=None):
    """Run all Part 1 baselines on one sequence.

    Synthetic sequences are degraded from HR to LR inside this function and can
    be scored with PSNR/SSIM. Real sequences are upscaled directly and are used
    only for qualitative figures/videos.
    """
    image_paths = list_images(input_dir)
    if max_frames:
        image_paths = image_paths[:max_frames]
    if not image_paths:
        print(f"[skip] {name}: no frames found in {input_dir}")
        return None

    seq_root = Path(output_root) / name
    for method in ["lr", "bicubic", "lanczos", "srcnn", "temporal", "comparison"]:
        ensure_dir(seq_root / method)

    bicubic_bgr_frames = []
    lanczos_frames = []
    srcnn_frames = []
    references = []
    lr_frames = []

    print(f"[run] {name}: {len(image_paths)} frames")
    for path in tqdm(image_paths, desc=name):
        frame = load_rgb(path)
        lr, reference = make_lr_and_reference(frame, scale=scale, real_input=real_input)
        target_size = None if real_input else reference.size

        # Part 1 compares classical interpolation, shallow CNN SR, and temporal
        # fusion under the same LR input.
        bicubic = upscale_lr(lr, scale=scale, target_size=target_size, method=Image.BICUBIC)
        lanczos = upscale_lr(lr, scale=scale, target_size=target_size, method=Image.LANCZOS)
        srcnn = run_srcnn(model, device, bicubic) if model is not None else bicubic

        lr.save(seq_root / "lr" / path.name)
        bicubic.save(seq_root / "bicubic" / path.name)
        lanczos.save(seq_root / "lanczos" / path.name)
        srcnn.save(seq_root / "srcnn" / path.name)

        bicubic_bgr_frames.append(pil_to_bgr(bicubic))
        lanczos_frames.append(pil_to_bgr(lanczos))
        srcnn_frames.append(pil_to_bgr(srcnn))
        references.append(reference)
        lr_frames.append(lr)

    # Temporal baseline is computed after bicubic upsampling, matching the PDF roadmap.
    temporal_frames = temporal_average(bicubic_bgr_frames, radius=1, weights=np.array([0.25, 0.5, 0.25]))
    temporal_frames = [unsharp_mask_bgr(frame) for frame in temporal_frames]
    for path, frame in zip(image_paths, temporal_frames):
        bgr_to_pil(frame).save(seq_root / "temporal" / path.name)

    sample_indices = sorted(set([0, len(image_paths) // 2, len(image_paths) - 1]))
    for idx in sample_indices:
        # Save first/middle/last strips so the report can show both early and
        # later motion/texture behavior without including every frame.
        path = image_paths[idx]
        tiles = [
            upscale_lr(lr_frames[idx], scale=scale, target_size=(bicubic_bgr_frames[idx].shape[1], bicubic_bgr_frames[idx].shape[0])),
            bgr_to_pil(bicubic_bgr_frames[idx]),
            bgr_to_pil(lanczos_frames[idx]),
            bgr_to_pil(srcnn_frames[idx]),
            bgr_to_pil(temporal_frames[idx]),
        ]
        labels = ["LR input", "Bicubic", "Lanczos", "SRCNN", "Temporal"]
        if references[idx] is not None:
            tiles.append(references[idx])
            labels.append("GT")
        label_strip(tiles, labels).save(seq_root / "comparison" / path.name)

    videos_dir = seq_root / "videos"
    ensure_dir(videos_dir)
    save_video(bicubic_bgr_frames, videos_dir / "bicubic.mp4")
    save_video(lanczos_frames, videos_dir / "lanczos.mp4")
    save_video(srcnn_frames, videos_dir / "srcnn.mp4")
    save_video(temporal_frames, videos_dir / "temporal.mp4")

    metrics = []
    if not real_input:
        # Wild videos do not have GT, so they are used only for qualitative videos/figures.
        for method in ["bicubic", "lanczos", "srcnn", "temporal"]:
            result = calculate_metrics(input_dir, seq_root / method)
            if result:
                result.update({"sequence": name, "method": method})
                metrics.append(result)
    return metrics


def write_metrics(metrics, output_path):
    # Simple CSV output makes manual checking and report table creation easy.
    ensure_dir(Path(output_path).parent)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sequence", "method", "frames", "psnr", "ssim"])
        writer.writeheader()
        for row in metrics:
            writer.writerow({
                "sequence": row["sequence"],
                "method": row["method"],
                "frames": row["frames"],
                "psnr": f"{row['psnr']:.4f}",
                "ssim": f"{row['ssim']:.6f}",
            })


def main():
    parser = argparse.ArgumentParser(description="Run Part 1 VSR baselines.")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--real-scale", type=int, default=2)
    parser.add_argument("--output", default="results/part1")
    parser.add_argument("--weights", default="srcnn_weights.pth")
    parser.add_argument("--max-wild-frames", type=int, default=80)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_srcnn(args.weights, device)

    # Synthetic sequences are downsampled in-script and evaluated against original HR.
    synthetic_sequences = {
        "REDS_002": "data/sample/REDS-sample/REDS-sample/002",
        "Vimeo_00018_0043": "data/sample/vimeo-RL/vimeo-RL/00018/0043",
    }
    # Real sequences are upscaled directly and used for qualitative inspection.
    real_sequences = {
        "wild": "data/wild/extracted_frames",
    }

    all_metrics = []
    for name, path in synthetic_sequences.items():
        if Path(path).exists():
            metrics = process_sequence(name, path, args.output, args.scale, model, device, real_input=False)
            all_metrics.extend(metrics or [])
        else:
            print(f"[skip] missing synthetic sequence: {path}")

    for name, path in real_sequences.items():
        if Path(path).exists():
            process_sequence(
                name,
                path,
                args.output,
                args.real_scale,
                model,
                device,
                real_input=True,
                max_frames=args.max_wild_frames,
            )
        else:
            print(f"[skip] missing real sequence: {path}")

    metrics_path = Path(args.output) / "metrics_part1.csv"
    write_metrics(all_metrics, metrics_path)
    print(f"[done] Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
