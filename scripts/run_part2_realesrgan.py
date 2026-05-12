import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from PIL import Image, ImageDraw
from realesrgan import RealESRGANer
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm


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


def make_lr(image, scale):
    lr_size = (max(1, image.width // scale), max(1, image.height // scale))
    return image.resize(lr_size, Image.BICUBIC)


def resize_to(image, target_size):
    if image.size == target_size:
        return image
    return image.resize(target_size, Image.BICUBIC)


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


def save_video(frames, output_path, fps=24):
    if not frames:
        return
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()


def build_upsampler(weights_path, tile, half, device):
    # RealESRGAN_x4plus uses an RRDBNet backbone; tiling avoids VRAM spikes on big frames.
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4,
    )
    return RealESRGANer(
        scale=4,
        model_path=str(weights_path),
        model=model,
        tile=tile,
        tile_pad=10,
        pre_pad=0,
        half=half,
        device=device,
    )


def enhance_pil(upsampler, image, outscale):
    # Real-ESRGAN works with OpenCV BGR arrays, while the rest of this repo uses PIL RGB.
    output, _ = upsampler.enhance(pil_to_bgr(image), outscale=outscale)
    return bgr_to_pil(output)


def calculate_metrics(gt_dir, pred_dir):
    # GAN/perceptual SR may look sharper even when PSNR/SSIM is lower.
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


def process_synthetic_sequence(name, input_dir, output_root, scale, upsampler, max_frames=None):
    paths = list_images(input_dir)
    if max_frames:
        paths = paths[:max_frames]
    if not paths:
        print(f"[skip] {name}: no frames in {input_dir}")
        return None

    seq_root = Path(output_root) / "real_esrgan" / name
    for subdir in ["lr", "bicubic", "frames", "comparison", "videos"]:
        ensure_dir(seq_root / subdir)

    video_frames = []
    print(f"[run] Real-ESRGAN synthetic {name}: {len(paths)} frames")
    for path in tqdm(paths, desc=f"part2-{name}"):
        gt = load_rgb(path)
        # Match Part 1: create LR by synthetic bicubic downsampling, then recover HR.
        lr = make_lr(gt, scale)
        bicubic = lr.resize(gt.size, Image.BICUBIC)
        sr = enhance_pil(upsampler, lr, outscale=scale)
        sr = resize_to(sr, gt.size)

        lr.save(seq_root / "lr" / path.name)
        bicubic.save(seq_root / "bicubic" / path.name)
        sr.save(seq_root / "frames" / path.name)
        video_frames.append(pil_to_bgr(sr))

    sample_indices = sorted(set([0, len(paths) // 2, len(paths) - 1]))
    for idx in sample_indices:
        path = paths[idx]
        gt = load_rgb(path)
        lr = load_rgb(seq_root / "lr" / path.name).resize(gt.size, Image.NEAREST)
        bicubic = load_rgb(seq_root / "bicubic" / path.name)
        sr = load_rgb(seq_root / "frames" / path.name)
        label_strip(
            [lr, bicubic, sr, gt],
            ["LR input", "Bicubic", "Real-ESRGAN", "GT"],
        ).save(seq_root / "comparison" / path.name)

    save_video(video_frames, seq_root / "videos" / "real_esrgan.mp4")
    metrics = calculate_metrics(input_dir, seq_root / "frames")
    if metrics:
        metrics.update({"sequence": name, "method": "real_esrgan"})
    return metrics


def process_real_sequence(name, input_dir, output_root, real_scale, upsampler, max_frames):
    paths = list_images(input_dir)
    if max_frames == 0:
        print(f"[skip] {name}: --max-wild-frames 0")
        return
    if max_frames:
        paths = paths[:max_frames]
    if not paths:
        print(f"[skip] {name}: no frames in {input_dir}")
        return

    seq_root = Path(output_root) / "real_esrgan" / name
    for subdir in ["frames", "comparison", "videos"]:
        ensure_dir(seq_root / subdir)

    video_frames = []
    print(f"[run] Real-ESRGAN real {name}: {len(paths)} frames")
    for path in tqdm(paths, desc=f"part2-{name}"):
        # Wild frames have no GT, so these outputs are for visual comparison/video demos.
        frame = load_rgb(path)
        sr = enhance_pil(upsampler, frame, outscale=real_scale)
        sr.save(seq_root / "frames" / path.name)
        video_frames.append(pil_to_bgr(sr))

    sample_indices = sorted(set([0, len(paths) // 2, len(paths) - 1]))
    for idx in sample_indices:
        path = paths[idx]
        frame = load_rgb(path)
        sr = load_rgb(seq_root / "frames" / path.name)
        input_up = frame.resize(sr.size, Image.BICUBIC)
        label_strip(
            [input_up, sr],
            ["Bicubic input", "Real-ESRGAN"],
        ).save(seq_root / "comparison" / path.name)

    save_video(video_frames, seq_root / "videos" / "real_esrgan.mp4")


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
    parser = argparse.ArgumentParser(description="Run Part 2 pretrained Real-ESRGAN reproduction.")
    parser.add_argument("--weights", default="weights/RealESRGAN_x4plus.pth")
    parser.add_argument("--output", default="results/part2")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--real-scale", type=int, default=2)
    parser.add_argument("--tile", type=int, default=256)
    parser.add_argument("--max-wild-frames", type=int, default=12)
    parser.add_argument("--max-sample-frames", type=int, default=None)
    parser.add_argument("--fp32", action="store_true")
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing Real-ESRGAN weights: {weights_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Half precision is faster on GPU; --fp32 is available if numerical/debug issues appear.
    half = device.type == "cuda" and not args.fp32
    upsampler = build_upsampler(weights_path, tile=args.tile, half=half, device=device)

    synthetic_sequences = {
        "REDS_002": "data/sample/REDS-sample/REDS-sample/002",
        "Vimeo_00018_0043": "data/sample/vimeo-RL/vimeo-RL/00018/0043",
    }

    rows = []
    for name, path in synthetic_sequences.items():
        if Path(path).exists():
            metrics = process_synthetic_sequence(
                name,
                path,
                args.output,
                args.scale,
                upsampler,
                max_frames=args.max_sample_frames,
            )
            if metrics:
                rows.append(metrics)
        else:
            print(f"[skip] missing sequence: {path}")

    wild_dir = "data/wild/extracted_frames"
    if Path(wild_dir).exists():
        process_real_sequence("wild", wild_dir, args.output, args.real_scale, upsampler, args.max_wild_frames)

    metrics_path = Path(args.output) / "metrics_part2.csv"
    write_metrics(rows, metrics_path)
    print(f"[done] Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
