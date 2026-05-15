# AIAA 3201 Project: Video Super-Resolution

This repository implements the AIAA 3201 term project pipeline for reconstructing high-resolution video frames from low-resolution inputs.

## Current Status

### Part 1: Baseline

Implemented and runnable:

- Bicubic interpolation
- Lanczos interpolation
- SRCNN inference with `srcnn_weights.pth`
- Multi-frame temporal averaging with unsharp masking
- PSNR / SSIM evaluation on sample datasets
- Side-by-side comparison images
- MP4 outputs for each baseline method

Generated outputs are written to `results/part1/`.

### Part 2: SOTA Reproduction

Implemented:

- Real-ESRGAN x4plus pretrained inference
- BasicVSR pretrained inference
- BasicVSR++ pretrained inference
- Synthetic 4x evaluation on REDS and Vimeo samples
- Real-video enhancement on wild frames
- Comparison figures, MP4 videos, and PSNR/SSIM metrics

### Part 3: Exploration

Implemented:

- Adaptive hybrid enhancement based on local uncertainty / texture / temporal difference
- BasicVSR++ and Real-ESRGAN fusion with a generated mask
- PSNR / SSIM evaluation on sample datasets
- Comparison figures, mask visualizations, and MP4 videos
- No large model training required
- Designed to preserve BasicVSR++ stability while adding controlled perceptual detail

## Setup

```bash
pip install -r requirements.txt
```

The current tested environment uses PyTorch 2.1.2 with CUDA 12.1 and NumPy 1.26.x. Keep NumPy below 2.0 for compatibility with this PyTorch build.

## Data Layout

Expected sample data:

```text
data/sample/REDS-sample/REDS-sample/002
data/sample/vimeo-RL/vimeo-RL/00018/0043
data/wild/extracted_frames
```

The sample datasets are evaluated by synthetically downsampling each HR frame by 4x and reconstructing it back to the original resolution.

The wild video frames are treated as real low-resolution inputs and upscaled by 2x by default to keep output videos practical in size.

## Run Part 1

```bash
python scripts/run_part1.py
```

Useful options:

```bash
python scripts/run_part1.py --scale 4 --real-scale 2 --max-wild-frames 80
```

Outputs:

```text
results/part1/REDS_002/
results/part1/Vimeo_00018_0043/
results/part1/wild/
results/part1/metrics_part1.csv
```

Each sequence folder contains:

```text
lr/
bicubic/
lanczos/
srcnn/
temporal/
comparison/
videos/
```

## Evaluate Part 1

```bash
python scripts/evaluate_metrics.py
```

This writes:

```text
results/part1/metrics_part1.csv
```

Current sample metrics after running Part 1:

| Sequence | Method | PSNR | SSIM |
|---|---:|---:|---:|
| REDS_002 | Bicubic | 21.6300 | 0.675386 |
| REDS_002 | Lanczos | 21.8382 | 0.687649 |
| REDS_002 | SRCNN | 20.5023 | 0.652510 |
| REDS_002 | Temporal | 20.3740 | 0.595532 |
| Vimeo_00018_0043 | Bicubic | 27.5722 | 0.751740 |
| Vimeo_00018_0043 | Lanczos | 27.7183 | 0.759160 |
| Vimeo_00018_0043 | SRCNN | 24.2808 | 0.690225 |
| Vimeo_00018_0043 | Temporal | 27.6853 | 0.758837 |

The REDS temporal baseline performs worse because naive frame averaging is not motion-aligned. This is a useful failure case for motivating BasicVSR in Part 2.

## Run Part 2

Download or place the pretrained weights at:

```text
weights/RealESRGAN_x4plus.pth
weights/BasicVSR_REDS4-543c8261.pth
weights/basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth
```

Run the pretrained Real-ESRGAN reproduction:

```bash
python scripts/run_part2_realesrgan.py --max-wild-frames 12
```

Run BasicVSR:

```bash
python scripts/run_part2_basicvsr.py --max-wild-frames 6
```

Run BasicVSR++:

```bash
python scripts/run_part2_basicvsrpp.py --max-wild-frames 6
```

Outputs:

```text
results/part2/real_esrgan/REDS_002/
results/part2/real_esrgan/Vimeo_00018_0043/
results/part2/real_esrgan/wild/
results/part2/basicvsr/REDS_002/
results/part2/basicvsr/Vimeo_00018_0043/
results/part2/basicvsr/wild/
results/part2/basicvsrpp/REDS_002/
results/part2/basicvsrpp/Vimeo_00018_0043/
results/part2/basicvsrpp/wild/
results/part2/metrics_part2_all.csv
results/part2/metrics_part2.csv
results/part2/metrics_basicvsr.csv
results/part2/metrics_basicvsrpp.csv
```

Current Part 2 sample metrics:

| Sequence | Method | PSNR | SSIM |
|---|---:|---:|---:|
| REDS_002 | Real-ESRGAN | 19.8416 | 0.632718 |
| REDS_002 | BasicVSR | 26.1518 | 0.883018 |
| REDS_002 | BasicVSR++ | 26.8504 | 0.900157 |
| Vimeo_00018_0043 | Real-ESRGAN | 24.9929 | 0.667951 |
| Vimeo_00018_0043 | BasicVSR | 29.1144 | 0.830463 |
| Vimeo_00018_0043 | BasicVSR++ | 29.3056 | 0.834918 |

BasicVSR++ is the strongest quantitative method. Real-ESRGAN is perceptually sharper in some textured regions but scores lower under PSNR/SSIM, which is expected for GAN-based perceptual restoration.

## Run Part 3

Run adaptive hybrid refinement:

```bash
python scripts/run_part3_adaptive_hybrid.py --max-wild-frames 6
```

Outputs:

```text
results/part3/REDS_002/
results/part3/Vimeo_00018_0043/
results/part3/wild/
results/part3/metrics_part3.csv
```

Current Part 3 sample metrics:

| Sequence | Method | PSNR | SSIM |
|---|---:|---:|---:|
| REDS_002 | Adaptive Hybrid | 26.7978 | 0.898219 |
| Vimeo_00018_0043 | Adaptive Hybrid | 29.2463 | 0.833274 |

The hybrid result stays close to BasicVSR++ under PSNR/SSIM while selectively blending Real-ESRGAN details through an adaptive mask.

## Plot Metrics

Generate report-ready line charts for PSNR and SSIM across all implemented Part 1/2/3 methods:

```bash
python scripts/plot_metrics.py
```

Outputs:

```text
results/figures/metrics_all_methods.csv
results/figures/psnr_line_comparison.png
results/figures/ssim_line_comparison.png
results/figures/metric_line_comparison.png
```

## REDS Additional Experiment

The additional REDS evaluation uses three extra REDS sample clips (`007`, `010`, `012`) and reports PSNR, SSIM, LPIPS, FID, and tLPIPS, plus qualitative figures and processed videos.

```bash
python scripts/run_reds_additional_experiment.py --sequences 007 010 012 --max-metric-frames 30 --max-fid-frames 50
```

Outputs:

```text
results/reds_additional/tables/reds_additional_metrics.csv
results/reds_additional/figures/pipeline_flowchart.png
results/reds_additional/figures/REDS_007_rendering_comparison.png
results/reds_additional/figures/REDS_007_zoom_patches.png
results/reds_additional/figures/reds_additional_psnr.png
results/reds_additional/figures/reds_additional_ssim.png
results/reds_additional/figures/reds_additional_lpips.png
results/reds_additional/figures/reds_additional_fid.png
results/reds_additional/figures/reds_additional_tlpips.png
results/reds_additional/part*/**/videos/*.mp4
```

Average additional REDS results across clips 007/010/012:

| Method | PSNR | SSIM | LPIPS | FID | tLPIPS |
|---|---:|---:|---:|---:|---:|
| BasicVSR++ | 28.9653 | 0.887440 | 0.144580 | 127.5387 | 0.004899 |
| Adaptive Hybrid | 28.8943 | 0.884806 | 0.144591 | 125.3311 | 0.005136 |
| Lanczos | 24.2426 | 0.707970 | 0.508716 | 234.4695 | 0.018727 |
| Bicubic | 24.0223 | 0.697719 | 0.496769 | 254.9038 | 0.019029 |
| Temporal Avg. | 22.4859 | 0.620048 | 0.543455 | 276.9944 | 0.036659 |
| Real-ESRGAN | 22.2057 | 0.643936 | 0.187831 | 156.0711 | 0.022322 |
| SRCNN | 22.1047 | 0.678610 | 0.560812 | 275.4520 | 0.008223 |

For the full official REDS validation benchmark, download and extract REDS into `data/benchmark/REDS`. The benchmark script expects HR validation clips under `data/benchmark/REDS/val/val_sharp/<clip_id>` and creates synthetic x4 LR frames internally.

To download the public REDS VSR benchmark into `data/benchmark/REDS`, use:

```bash
python scripts/download_reds_benchmark.py --root data/benchmark/REDS --proxy http://127.0.0.1:18080
```

The downloader uses Google Drive IDs from the official REDS page, resumes partial files with `gdown --continue`, and extracts each archive after it finishes. It downloads validation files first so the full validation benchmark becomes available before the large training HR archive completes.

After `val_sharp.zip` is extracted, run the full REDS validation benchmark with:

```bash
python scripts/run_reds_additional_experiment.py \
  --reds-root data/benchmark/REDS/val/val_sharp \
  --all-sequences \
  --output results/reds_benchmark_val \
  --max-metric-frames 30 \
  --max-fid-frames 50
```

This runs the same Part 1/2/3 pipeline on every validation clip and writes tables, metric figures, rendering comparisons, zoom-in patches, and mp4 outputs under `results/reds_benchmark_val`.

## Train SRCNN

```bash
python train_srcnn.py
```

The default training script uses `data/sample/REDS-sample/REDS-sample/002` and saves weights to `srcnn_weights.pth`.

## Repository Notes

`data/` and `results/` are ignored by Git because they contain datasets and generated outputs. Re-run the pipeline commands above to regenerate outputs locally.
