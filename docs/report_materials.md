# Report Materials and Progress Notes

This file collects the current experiment evidence for writing the final CVPR-style report.

## Project Goal

Given a low-resolution video sequence, the project aims to reconstruct high-resolution frames while improving spatial detail and maintaining temporal consistency.

The implementation is organized into three parts:

- Part 1: classical and early CNN baselines
- Part 2: pretrained SOTA reproduction
- Part 3: lightweight optimization / extension

## Current Progress Summary

| Part | Requirement | Current Status | Report Use |
|---|---|---|---|
| Part 1 | Bicubic/Lanczos, SRCNN, multi-frame averaging, PSNR/SSIM, visual comparison | Completed and run | Main baseline section |
| Part 2 | BasicVSR / BasicVSR++ / Real-ESRGAN SOTA reproduction | Completed and run | SOTA reproduction section |
| Part 3 | Optimization or extension | Completed and run | Exploration section |

## Part 1: Baseline

### Completed Tasks

- Implemented Bicubic interpolation.
- Implemented Lanczos interpolation.
- Implemented a 3-layer SRCNN model.
- Loaded and ran `srcnn_weights.pth`.
- Implemented temporal average fusion with unsharp masking.
- Built a unified Part 1 inference script: `scripts/run_part1.py`.
- Generated per-method output frames.
- Generated side-by-side comparison figures.
- Generated MP4 videos for each method.
- Calculated PSNR and SSIM on sample datasets.

### Reproducible Commands

Run all Part 1 baselines:

```bash
python scripts/run_part1.py --scale 4 --real-scale 2 --max-wild-frames 80
```

Recalculate PSNR/SSIM:

```bash
python scripts/evaluate_metrics.py
```

### Output Locations

Main output directory:

```text
results/part1/
```

Metrics table:

```text
results/part1/metrics_part1.csv
```

Comparison figures:

```text
results/part1/REDS_002/comparison/00000000.png
results/part1/REDS_002/comparison/00000050.png
results/part1/REDS_002/comparison/00000099.png
results/part1/Vimeo_00018_0043/comparison/im1.png
results/part1/Vimeo_00018_0043/comparison/im4.png
results/part1/Vimeo_00018_0043/comparison/im7.png
results/part1/wild/comparison/0000.png
results/part1/wild/comparison/0040.png
results/part1/wild/comparison/0079.png
```

Videos:

```text
results/part1/REDS_002/videos/bicubic.mp4
results/part1/REDS_002/videos/lanczos.mp4
results/part1/REDS_002/videos/srcnn.mp4
results/part1/REDS_002/videos/temporal.mp4
results/part1/Vimeo_00018_0043/videos/bicubic.mp4
results/part1/Vimeo_00018_0043/videos/lanczos.mp4
results/part1/Vimeo_00018_0043/videos/srcnn.mp4
results/part1/Vimeo_00018_0043/videos/temporal.mp4
results/part1/wild/videos/bicubic.mp4
results/part1/wild/videos/lanczos.mp4
results/part1/wild/videos/srcnn.mp4
results/part1/wild/videos/temporal.mp4
```

### Quantitative Results

| Dataset | Method | Frames | PSNR | SSIM |
|---|---:|---:|---:|---:|
| REDS_002 | Bicubic | 100 | 21.6300 | 0.675386 |
| REDS_002 | Lanczos | 100 | 21.8382 | 0.687649 |
| REDS_002 | SRCNN | 100 | 20.5023 | 0.652510 |
| REDS_002 | Temporal | 100 | 20.3740 | 0.595532 |
| Vimeo_00018_0043 | Bicubic | 7 | 27.5722 | 0.751740 |
| Vimeo_00018_0043 | Lanczos | 7 | 27.7183 | 0.759160 |
| Vimeo_00018_0043 | SRCNN | 7 | 24.2808 | 0.690225 |
| Vimeo_00018_0043 | Temporal | 7 | 27.6853 | 0.758837 |

### How to Present Part 1 in the Report

Use the table above as the first quantitative baseline table. It shows that Lanczos slightly outperforms Bicubic on both sample datasets under PSNR/SSIM.

Use one REDS comparison figure, preferably:

```text
results/part1/REDS_002/comparison/00000050.png
```

This frame makes the blur gap between LR/interpolation and GT visually clear.

Use one wild-video comparison figure, preferably:

```text
results/part1/wild/comparison/0040.png
```

This figure demonstrates generalization to real low-resolution input where no GT is available.

Use the MP4 files for the demo submission. For Part 1, include at least:

```text
results/part1/wild/videos/bicubic.mp4
results/part1/wild/videos/srcnn.mp4
results/part1/wild/videos/temporal.mp4
```

### Interpretation for Report Text

The interpolation baselines are stable and fast, but they cannot recover missing high-frequency details. Lanczos gives slightly better PSNR/SSIM than Bicubic in the current synthetic evaluation, but both remain visibly blurry.

SRCNN is a simple early deep-learning baseline. In the current run, SRCNN underperforms Bicubic/Lanczos quantitatively. This is likely because the included weights are trained quickly on a small subset, so the model does not generalize as well as classical interpolation on the current evaluation frames. This is useful to mention as a limitation of shallow CNNs and limited training data.

The temporal averaging baseline does not use motion alignment. On REDS, it performs worse because moving objects and camera motion cause neighboring frames to be misaligned before averaging. This failure case directly motivates Part 2 methods such as BasicVSR, which explicitly perform temporal propagation and alignment.

### Suggested Figures and Captions

Figure: Part 1 baseline comparison on REDS.

Caption draft:

> Qualitative comparison of Part 1 baselines on REDS_002. Bicubic and Lanczos produce stable but blurry outputs. SRCNN gives a learned reconstruction but remains limited by shallow architecture and small-scale training. Naive temporal averaging can introduce motion blur when frames are not aligned.

Figure: Part 1 baseline comparison on wild video.

Caption draft:

> Qualitative comparison on a real low-resolution video. Since ground truth is unavailable, this example is used to inspect visual sharpness and temporal behavior. The result highlights the need for stronger pretrained restoration models in Part 2.

Table: Part 1 PSNR/SSIM.

Caption draft:

> Quantitative evaluation of Part 1 baselines under synthetic 4x degradation. Lanczos performs best among the simple baselines, while naive temporal averaging can degrade scores when frame motion is not compensated.

## Part 2: SOTA Reproduction

### Completed Tasks

- Installed Real-ESRGAN and its restoration dependencies.
- Downloaded the official `RealESRGAN_x4plus.pth` pretrained weights.
- Implemented a reproducible Part 2 inference script: `scripts/run_part2_realesrgan.py`.
- Ran Real-ESRGAN on REDS_002 and Vimeo_00018_0043 under synthetic 4x degradation.
- Ran Real-ESRGAN on 12 wild-video frames as real low-resolution input.
- Downloaded official BasicSR/OpenMMLab pretrained weights for BasicVSR and BasicVSR++.
- Implemented reproducible BasicVSR and BasicVSR++ inference scripts:
  - `scripts/run_part2_basicvsr.py`
  - `scripts/run_part2_basicvsrpp.py`
- Ran BasicVSR and BasicVSR++ on REDS_002 and Vimeo_00018_0043 under synthetic 4x degradation.
- Ran BasicVSR and BasicVSR++ on 6 wild-video frames as real low-resolution input.
- Generated Part 2 comparison figures, MP4 videos, and PSNR/SSIM metrics.

### Reproducible Commands

Run Part 2 Real-ESRGAN:

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

Quick smoke tests can use:

```bash
python scripts/run_part2_basicvsr.py --max-sample-frames 2 --max-wild-frames 0 --output results/part2_smoke
python scripts/run_part2_basicvsrpp.py --max-sample-frames 2 --max-wild-frames 0 --output results/part2_smoke
```

### Output Locations

Main output directory:

```text
results/part2/
```

Metrics table:

```text
results/part2/metrics_part2_all.csv
results/part2/metrics_part2.csv
results/part2/metrics_basicvsr.csv
results/part2/metrics_basicvsrpp.csv
```

Comparison figures:

```text
results/part2/real_esrgan/REDS_002/comparison/00000000.png
results/part2/real_esrgan/REDS_002/comparison/00000050.png
results/part2/real_esrgan/REDS_002/comparison/00000099.png
results/part2/real_esrgan/Vimeo_00018_0043/comparison/im1.png
results/part2/real_esrgan/Vimeo_00018_0043/comparison/im4.png
results/part2/real_esrgan/Vimeo_00018_0043/comparison/im7.png
results/part2/real_esrgan/wild/comparison/0000.png
results/part2/real_esrgan/wild/comparison/0006.png
results/part2/real_esrgan/wild/comparison/0011.png
results/part2/basicvsr/REDS_002/comparison/00000050.png
results/part2/basicvsr/Vimeo_00018_0043/comparison/im4.png
results/part2/basicvsr/wild/comparison/0003.png
results/part2/basicvsrpp/REDS_002/comparison/00000050.png
results/part2/basicvsrpp/Vimeo_00018_0043/comparison/im4.png
results/part2/basicvsrpp/wild/comparison/0003.png
```

Videos:

```text
results/part2/real_esrgan/REDS_002/videos/real_esrgan.mp4
results/part2/real_esrgan/Vimeo_00018_0043/videos/real_esrgan.mp4
results/part2/real_esrgan/wild/videos/real_esrgan.mp4
results/part2/basicvsr/REDS_002/videos/basicvsr.mp4
results/part2/basicvsr/Vimeo_00018_0043/videos/basicvsr.mp4
results/part2/basicvsr/wild/videos/basicvsr.mp4
results/part2/basicvsrpp/REDS_002/videos/basicvsrpp.mp4
results/part2/basicvsrpp/Vimeo_00018_0043/videos/basicvsrpp.mp4
results/part2/basicvsrpp/wild/videos/basicvsrpp.mp4
```

### Quantitative Results

| Dataset | Method | Frames | PSNR | SSIM |
|---|---:|---:|---:|---:|
| REDS_002 | Real-ESRGAN | 100 | 19.8416 | 0.632718 |
| REDS_002 | BasicVSR | 100 | 26.1518 | 0.883018 |
| REDS_002 | BasicVSR++ | 100 | 26.8504 | 0.900157 |
| Vimeo_00018_0043 | Real-ESRGAN | 7 | 24.9929 | 0.667951 |
| Vimeo_00018_0043 | BasicVSR | 7 | 29.1144 | 0.830463 |
| Vimeo_00018_0043 | BasicVSR++ | 7 | 29.3056 | 0.834918 |

### How to Present Part 2 in the Report

Use the REDS comparison figure:

```text
results/part2/basicvsrpp/REDS_002/comparison/00000050.png
```

This figure is the best main Part 2 result because BasicVSR++ has the highest PSNR/SSIM and explicitly uses temporal propagation/alignment.

Use Real-ESRGAN as a perceptual contrast figure:

```text
results/part2/real_esrgan/REDS_002/comparison/00000050.png
```

This figure shows the difference between perceptual enhancement and distortion-oriented video SR.

Use the wild comparison figure:

```text
results/part2/real_esrgan/wild/comparison/0006.png
results/part2/basicvsrpp/wild/comparison/0003.png
```

This figure is useful for qualitative discussion because no GT exists for wild video.

Use the video:

```text
results/part2/real_esrgan/wild/videos/real_esrgan.mp4
results/part2/basicvsr/wild/videos/basicvsr.mp4
results/part2/basicvsrpp/wild/videos/basicvsrpp.mp4
```

This should be compared against:

```text
results/part1/wild/videos/bicubic.mp4
results/part1/wild/videos/srcnn.mp4
results/part1/wild/videos/temporal.mp4
```

### Interpretation for Report Text

BasicVSR and BasicVSR++ significantly outperform the Part 1 baselines in PSNR/SSIM. This confirms the value of temporal propagation and optical-flow-based alignment for video super-resolution. BasicVSR++ is the strongest quantitative method in the current project, reaching 26.8504 PSNR / 0.900157 SSIM on REDS_002 and 29.3056 PSNR / 0.834918 SSIM on the Vimeo sample.

Real-ESRGAN produces visibly sharper and more realistic local details than interpolation-based methods. However, its PSNR/SSIM is lower than Lanczos and much lower than BasicVSR/BasicVSR++ on the synthetic datasets. This is expected for GAN/perceptual super-resolution methods because they optimize perceptual realism rather than pixel-wise fidelity.

Together, these results support an important report point: BasicVSR/BasicVSR++ are better for distortion-based video SR and temporal consistency, while Real-ESRGAN is better framed as perceptual enhancement. PSNR/SSIM alone cannot fully represent visual quality, but they clearly show the advantage of dedicated VSR models over naive baselines.

This motivates Part 3: combining the temporal stability and high PSNR/SSIM of BasicVSR++ with the sharper perceptual details of Real-ESRGAN through an adaptive hybrid strategy.

### Suggested Figures and Captions

Figure: Real-ESRGAN reproduction on REDS.

Caption draft:

> Real-ESRGAN produces sharper perceptual details than Bicubic interpolation. Although PSNR/SSIM may decrease, visual quality improves in textured regions, illustrating the difference between distortion-based and perception-based super-resolution objectives.

Figure: BasicVSR++ reproduction on REDS.

Caption draft:

> BasicVSR++ uses temporal propagation and alignment to reconstruct sharper and more faithful frames than classical interpolation. It achieves the best PSNR/SSIM among the evaluated Part 2 methods, demonstrating the importance of video-specific temporal modeling.

Figure: Real-ESRGAN on wild video.

Caption draft:

> Real-ESRGAN applied to real low-resolution video frames. The model enhances local texture and edge contrast, but frame-independent processing may introduce temporal inconsistency, motivating the adaptive hybrid refinement in Part 3.

### Original Planned Task

Use pretrained SOTA models to improve perceptual quality and temporal handling without training from scratch.

Implemented targets:

- Real-ESRGAN for perceptual enhancement on individual frames.
- BasicVSR for video super-resolution with temporal propagation.
- BasicVSR++ for stronger propagation/alignment and best quantitative scores.

### Report Angle

Part 2 should be framed as a reproduction and comparison against Part 1. The main questions:

- Does pretrained SOTA produce sharper and more realistic textures?
- Does the output improve visual quality even when PSNR does not always increase?
- Are there artifacts such as hallucinated textures or temporal flicker?

## Part 3: Exploration

### Completed Direction

The implemented direction is an adaptive hybrid enhancement pipeline:

```text
Hybrid = (1 - mask) * BasicVSR++ + mask * Real-ESRGAN
```

BasicVSR++ is used as the stable, distortion-oriented video SR backbone. Real-ESRGAN is used as a perceptual detail source. The adaptive mask controls where Real-ESRGAN can contribute.

The system will compute simple uncertainty or risk maps from:

- local edge strength
- difference between BasicVSR++ and Real-ESRGAN outputs
- temporal frame difference from neighboring BasicVSR++ frames

The implementation is in:

```text
scripts/run_part3_adaptive_hybrid.py
```

Run command:

```bash
python scripts/run_part3_adaptive_hybrid.py --max-wild-frames 6
```

### Output Locations

Main output directory:

```text
results/part3/
```

Metrics table:

```text
results/part3/metrics_part3.csv
```

Comparison figures:

```text
results/part3/REDS_002/comparison/00000050.png
results/part3/Vimeo_00018_0043/comparison/im4.png
results/part3/wild/comparison/0003.png
```

Mask visualizations:

```text
results/part3/REDS_002/masks/00000050.png
results/part3/wild/masks/0003.png
```

Videos:

```text
results/part3/REDS_002/videos/adaptive_hybrid.mp4
results/part3/Vimeo_00018_0043/videos/adaptive_hybrid.mp4
results/part3/wild/videos/adaptive_hybrid.mp4
```

### Quantitative Results

| Dataset | Method | Frames | PSNR | SSIM |
|---|---:|---:|---:|---:|
| REDS_002 | Adaptive Hybrid | 100 | 26.7978 | 0.898219 |
| Vimeo_00018_0043 | Adaptive Hybrid | 7 | 29.2463 | 0.833274 |

For comparison, BasicVSR++ reaches 26.8504 / 0.900157 on REDS_002 and 29.3056 / 0.834918 on the Vimeo sample. The adaptive hybrid remains close to BasicVSR++ in PSNR/SSIM while introducing a controlled amount of perceptual detail from Real-ESRGAN.

### Why This Direction

This direction does not require training diffusion models, LoRA adapters, or flow matching models. It is computationally cheap and was implemented with classical image processing and already generated outputs.

### Report Angle

Part 3 can be presented as an attempt to balance:

- sharpness from perceptual enhancement
- stability from BasicVSR++ temporal propagation
- reduced temporal flicker

### How to Present Part 3 in the Report

Use this figure as the main Part 3 qualitative result:

```text
results/part3/REDS_002/comparison/00000050.png
```

It shows BasicVSR++, Real-ESRGAN, the adaptive mask, the hybrid result, and GT in one row.

Use this wild-video figure for qualitative inspection:

```text
results/part3/wild/comparison/0003.png
```

Suggested caption:

> Adaptive hybrid refinement combines the stable reconstruction of BasicVSR++ with selected perceptual details from Real-ESRGAN. The mask suppresses Real-ESRGAN in uncertain or high-motion regions while allowing limited enhancement in textured areas.

### Interpretation for Report Text

The adaptive hybrid result is not designed to beat BasicVSR++ on PSNR/SSIM. Instead, it aims to preserve most of BasicVSR++'s distortion-based fidelity while selectively incorporating Real-ESRGAN details. The quantitative result confirms that the fusion remains close to BasicVSR++: the REDS_002 PSNR drops only from 26.8504 to 26.7978, and SSIM drops from 0.900157 to 0.898219.

This makes Part 3 a low-risk extension: it demonstrates an uncertainty-aware refinement idea without costly training, while giving the report a clear discussion about the trade-off between fidelity, perceptual detail, and temporal stability.

## Report Structure Draft

1. Abstract
   - One paragraph summarizing the pipeline and key findings.
   - Include the GitHub repository link at the end.

2. Introduction
   - Define video super-resolution.
   - Explain spatial detail vs temporal consistency.
   - Summarize the three-part pipeline.

3. Related Work
   - SRCNN and classical image SR.
   - BasicVSR / BasicVSR++ for temporal propagation.
   - Real-ESRGAN for perceptual restoration.
   - LPIPS/tLPIPS for perceptual and temporal evaluation.

4. Method
   - Part 1 baseline methods.
   - Part 2 pretrained SOTA reproduction.
   - Part 3 adaptive hybrid extension.

5. Experiments
   - Datasets: REDS sample, Vimeo sample, wild video.
   - Metrics: PSNR, SSIM, qualitative comparisons.
   - Tables and figures from this file.

6. Discussion
   - Why interpolation is stable but blurry.
   - Why naive temporal averaging fails under motion.
   - Why SOTA/perceptual methods may improve visual quality but risk artifacts.

7. Conclusion
   - Summarize completed pipeline, limitations, and future improvements.

## Metric Line Charts

A plotting script has been added to generate intuitive metric trend figures across all implemented methods:

```bash
python scripts/plot_metrics.py
```

Generated files:

```text
results/figures/metrics_all_methods.csv
results/figures/psnr_line_comparison.png
results/figures/ssim_line_comparison.png
results/figures/metric_line_comparison.png
```

Recommended report figure:

```text
results/figures/metric_line_comparison.png
```

Suggested caption:

> PSNR and SSIM trends across all implemented methods. Classical baselines are stable but limited, BasicVSR and BasicVSR++ significantly improve distortion metrics through temporal propagation, Real-ESRGAN trades pixel fidelity for perceptual detail, and the adaptive hybrid remains close to BasicVSR++ while incorporating controlled perceptual enhancement.

Use `psnr_line_comparison.png` or `ssim_line_comparison.png` if the report needs separate single-metric plots.

## Current Limitations to Mention

- The SRCNN model is shallow and trained only lightly.
- The temporal average baseline does not include motion alignment.
- Wild-video evaluation lacks ground truth, so it must be judged qualitatively.
- PSNR/SSIM may not fully reflect perceptual quality.
- Part 2 and Part 3 results should include visual analysis, not just metrics.
