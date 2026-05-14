import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from basicsr.archs.basicvsr_arch import BasicVSR
from PIL import Image, ImageDraw
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchvision import transforms


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def list_images(frame_dir):
    """List frames in deterministic temporal order."""
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


def tensor_to_pil(tensor):
    tensor = tensor.detach().cpu().clamp(0.0, 1.0)
    return transforms.ToPILImage()(tensor)


def make_lr(image, scale):
    lr_size = (max(1, image.width // scale), max(1, image.height // scale))
    return image.resize(lr_size, Image.BICUBIC)


def resize_to(image, target_size):
    if image.size == target_size:
        return image
    return image.resize(target_size, Image.BICUBIC)


def load_sequence_tensor(images, device):
    # BasicVSR consumes a whole video clip as (B, T, C, H, W). The batch size is
    # one because each REDS/Vimeo sequence is evaluated independently.
    tensors = [transforms.ToTensor()(image) for image in images]
    return torch.stack(tensors, dim=0).unsqueeze(0).to(device)


def label_strip(images, labels):
    # Labeled comparison strip used for report-ready qualitative examples.
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


def calculate_metrics(gt_dir, pred_dir):
    # Distortion metrics are reported only for synthetic LR/HR pairs.
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


def load_basicvsr(weights_path, device, num_block):
    # The downloaded REDS4 checkpoint was trained with 30 residual blocks. The
    # value is configurable for debugging, but the default matches the weights.
    model = BasicVSR(num_block=num_block).to(device)
    checkpoint = torch.load(weights_path, map_location=device)
    params = checkpoint.get("params", checkpoint)
    model.load_state_dict(params, strict=True)
    model.eval()
    return model


def pad_sequence_tensor(tensor, min_size=64):
    # SPyNet builds an image pyramid; tiny LR inputs such as Vimeo 28x16 need padding
    # to avoid zero-sized feature maps during optical-flow estimation.
    _, _, _, h, w = tensor.shape
    pad_h = max(0, min_size - h)
    pad_w = max(0, min_size - w)
    if pad_h == 0 and pad_w == 0:
        return tensor, (h, w)
    b, n, c, h, w = tensor.shape
    flat = tensor.view(b * n, c, h, w)
    flat = F.pad(flat, (0, pad_w, 0, pad_h), mode="replicate")
    return flat.view(b, n, c, h + pad_h, w + pad_w), (h, w)


def run_model(model, lr_images, device):
    # BasicVSR takes the whole sequence at once: (batch, time, channel, height, width).
    input_tensor = load_sequence_tensor(lr_images, device)
    input_tensor, original_lr_size = pad_sequence_tensor(input_tensor)
    with torch.no_grad():
        # Inference is done in evaluation mode without gradients to reduce VRAM
        # and keep outputs deterministic.
        output = model(input_tensor).squeeze(0)
    original_h, original_w = original_lr_size
    # Remove any padding after the 4x model output.
    output = output[:, :, :original_h * 4, :original_w * 4]
    return [tensor_to_pil(frame) for frame in output]


def process_synthetic_sequence(
    name,
    input_dir,
    output_root,
    scale,
    model,
    device,
    max_frames=None,
    method_dir="basicvsr",
    method_label="BasicVSR",
):
    """Run BasicVSR/BasicVSR++ style model on one synthetic sequence."""
    paths = list_images(input_dir)
    if max_frames:
        paths = paths[:max_frames]
    if not paths:
        print(f"[skip] {name}: no frames in {input_dir}")
        return None

    seq_root = Path(output_root) / method_dir / name
    for subdir in ["lr", "bicubic", "frames", "comparison", "videos"]:
        ensure_dir(seq_root / subdir)

    gt_images = [load_rgb(path) for path in paths]
    # For benchmark-like evaluation, HR frames are degraded to LR and
    # reconstructed. The original HR frames remain GT for PSNR/SSIM.
    lr_images = [make_lr(image, scale) for image in gt_images]
    print(f"[run] {method_label} synthetic {name}: {len(paths)} frames, LR size={lr_images[0].size}")
    sr_images = run_model(model, lr_images, device)

    video_frames = []
    for path, gt, lr, sr in zip(paths, gt_images, lr_images, sr_images):
        # Some models/padding paths can produce dimensions that differ by one or
        # two pixels; resize back to GT for fair metric calculation.
        sr = resize_to(sr, gt.size)
        bicubic = lr.resize(gt.size, Image.BICUBIC)
        lr.save(seq_root / "lr" / path.name)
        bicubic.save(seq_root / "bicubic" / path.name)
        sr.save(seq_root / "frames" / path.name)
        video_frames.append(pil_to_bgr(sr))

    sample_indices = sorted(set([0, len(paths) // 2, len(paths) - 1]))
    for idx in sample_indices:
        path = paths[idx]
        gt = gt_images[idx]
        lr = lr_images[idx].resize(gt.size, Image.NEAREST)
        bicubic = load_rgb(seq_root / "bicubic" / path.name)
        sr = load_rgb(seq_root / "frames" / path.name)
        label_strip(
            [lr, bicubic, sr, gt],
            ["LR input", "Bicubic", method_label, "GT"],
        ).save(seq_root / "comparison" / path.name)

    save_video(video_frames, seq_root / "videos" / f"{method_dir}.mp4")
    metrics = calculate_metrics(input_dir, seq_root / "frames")
    if metrics:
        metrics.update({"sequence": name, "method": method_dir})
    return metrics


def process_real_sequence(
    name,
    input_dir,
    output_root,
    model,
    device,
    max_frames,
    input_resize,
    method_dir="basicvsr",
    method_label="BasicVSR",
):
    """Run video SR on real LR frames.

    Wild frames have no GT, so this path produces only frames, videos, and
    qualitative comparison strips.
    """
    if max_frames == 0:
        print(f"[skip] {name}: --max-wild-frames 0")
        return
    paths = list_images(input_dir)
    if max_frames:
        paths = paths[:max_frames]
    if not paths:
        print(f"[skip] {name}: no frames in {input_dir}")
        return

    seq_root = Path(output_root) / method_dir / name
    for subdir in ["lr", "frames", "comparison", "videos"]:
        ensure_dir(seq_root / subdir)

    raw_images = [load_rgb(path) for path in paths]
    lr_images = []
    for image in raw_images:
        if input_resize != 1.0:
            # Wild frames are large; resizing before 4x BasicVSR keeps VRAM/output size practical.
            size = (max(1, int(image.width * input_resize)), max(1, int(image.height * input_resize)))
            image = image.resize(size, Image.BICUBIC)
        lr_images.append(image)

    print(f"[run] {method_label} real {name}: {len(paths)} frames, input size={lr_images[0].size}")
    sr_images = run_model(model, lr_images, device)

    video_frames = []
    for path, lr, sr in zip(paths, lr_images, sr_images):
        lr.save(seq_root / "lr" / path.name)
        sr.save(seq_root / "frames" / path.name)
        video_frames.append(pil_to_bgr(sr))

    sample_indices = sorted(set([0, len(paths) // 2, len(paths) - 1]))
    for idx in sample_indices:
        path = paths[idx]
        lr = lr_images[idx]
        sr = load_rgb(seq_root / "frames" / path.name)
        input_up = lr.resize(sr.size, Image.BICUBIC)
        label_strip([input_up, sr], ["Bicubic input", method_label]).save(seq_root / "comparison" / path.name)

    save_video(video_frames, seq_root / "videos" / f"{method_dir}.mp4")


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
    parser = argparse.ArgumentParser(description="Run Part 2 pretrained BasicVSR reproduction.")
    parser.add_argument("--weights", default="weights/BasicVSR_REDS4-543c8261.pth")
    parser.add_argument("--output", default="results/part2")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--max-sample-frames", type=int, default=None)
    parser.add_argument("--max-wild-frames", type=int, default=6)
    parser.add_argument("--wild-input-resize", type=float, default=0.5)
    parser.add_argument("--num-block", type=int, default=30)
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing BasicVSR weights: {weights_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_basicvsr(weights_path, device, args.num_block)

    # REDS/Vimeo are quantitative synthetic tests; wild is qualitative only.
    synthetic_sequences = {
        "REDS_002": "data/sample/REDS-sample/REDS-sample/002",
        "Vimeo_00018_0043": "data/sample/vimeo-RL/vimeo-RL/00018/0043",
    }

    # Synthetic clips are short enough to run as full sequences. This preserves
    # BasicVSR's temporal propagation behavior and makes PSNR/SSIM comparable.
    rows = []
    for name, path in synthetic_sequences.items():
        if Path(path).exists():
            metrics = process_synthetic_sequence(
                name,
                path,
                args.output,
                args.scale,
                model,
                device,
                max_frames=args.max_sample_frames,
            )
            if metrics:
                rows.append(metrics)
        else:
            print(f"[skip] missing sequence: {path}")

    wild_dir = "data/wild/extracted_frames"
    if Path(wild_dir).exists():
        process_real_sequence("wild", wild_dir, args.output, model, device, args.max_wild_frames, args.wild_input_resize)

    metrics_path = Path(args.output) / "metrics_basicvsr.csv"
    write_metrics(rows, metrics_path)
    print(f"[done] Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
