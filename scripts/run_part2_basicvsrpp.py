import argparse
import csv
import re
from pathlib import Path

import torch
from basicsr.archs.basicvsrpp_arch import BasicVSRPlusPlus

from run_part2_basicvsr import (
    ensure_dir,
    process_real_sequence,
    process_synthetic_sequence,
)


def map_mmedit_basicvsrpp_state_dict(state_dict):
    # OpenMMLab and BasicSR use slightly different module names for the same model.
    # This remaps the official MMagic checkpoint into the BasicSR architecture.
    mapped = {}
    for key, value in state_dict.items():
        if key == "step_counter":
            continue
        if key.startswith("generator."):
            key = key[len("generator."):]
        key = key.replace("upsample1.upsample_conv", "upconv1")
        key = key.replace("upsample2.upsample_conv", "upconv2")

        # MMagic names SPyNet conv layers as 0..4.conv; BasicSR stores them as
        # 0,2,4,6,8 because ReLU layers occupy the odd Sequential positions.
        match = re.match(r"(spynet\.basic_module\.\d+\.basic_module\.)(\d+)\.conv\.(weight|bias)$", key)
        if match:
            key = f"{match.group(1)}{int(match.group(2)) * 2}.{match.group(3)}"
        mapped[key] = value
    return mapped


def load_basicvsrpp(weights_path, device):
    # Load the official REDS4 BasicVSR++ checkpoint after key-name conversion.
    model = BasicVSRPlusPlus().to(device)
    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint.get("params", checkpoint))
    state_dict = map_mmedit_basicvsrpp_state_dict(state_dict)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def write_metrics(rows, output_path):
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sequence", "method", "frames", "psnr", "ssim"])
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "sequence": row["sequence"],
                "method": "basicvsrpp",
                "frames": row["frames"],
                "psnr": f"{row['psnr']:.4f}",
                "ssim": f"{row['ssim']:.6f}",
            })


def main():
    parser = argparse.ArgumentParser(description="Run Part 2 pretrained BasicVSR++ reproduction.")
    parser.add_argument(
        "--weights",
        default="weights/basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth",
    )
    parser.add_argument("--output", default="results/part2")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--max-sample-frames", type=int, default=None)
    parser.add_argument("--max-wild-frames", type=int, default=6)
    parser.add_argument("--wild-input-resize", type=float, default=0.5)
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing BasicVSR++ weights: {weights_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_basicvsrpp(weights_path, device)

    # Reuse the same evaluation protocol as BasicVSR for a clean comparison.
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
                model,
                device,
                max_frames=args.max_sample_frames,
                method_dir="basicvsrpp",
                method_label="BasicVSR++",
            )
            if metrics:
                metrics["method"] = "basicvsrpp"
                rows.append(metrics)
        else:
            print(f"[skip] missing sequence: {path}")

    wild_dir = "data/wild/extracted_frames"
    if Path(wild_dir).exists():
        process_real_sequence(
            "wild",
            wild_dir,
            args.output,
            model,
            device,
            args.max_wild_frames,
            args.wild_input_resize,
            method_dir="basicvsrpp",
            method_label="BasicVSR++",
        )

    metrics_path = Path(args.output) / "metrics_basicvsrpp.csv"
    write_metrics(rows, metrics_path)
    print(f"[done] Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
