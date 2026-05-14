# Processed Video Manifest

This file lists the processed video outputs that should be used for report demos or supplementary submission.

## Full REDS Validation Benchmark

Root directory:

```text
results/reds_benchmark_val/
```

Dataset coverage:

- `REDS_000` through `REDS_029`
- 30 validation sequences
- 100 frames per sequence
- 7 output methods per sequence
- 210 processed videos total

Video path templates:

```text
results/reds_benchmark_val/part1/<sequence>/videos/bicubic.mp4
results/reds_benchmark_val/part1/<sequence>/videos/lanczos.mp4
results/reds_benchmark_val/part1/<sequence>/videos/srcnn.mp4
results/reds_benchmark_val/part1/<sequence>/videos/temporal.mp4
results/reds_benchmark_val/part2/real_esrgan/<sequence>/videos/real_esrgan.mp4
results/reds_benchmark_val/part2/basicvsrpp/<sequence>/videos/basicvsrpp.mp4
results/reds_benchmark_val/part3/<sequence>/videos/adaptive_hybrid.mp4
```

Recommended compact demo subset:

```text
results/reds_benchmark_val/part1/REDS_000/videos/bicubic.mp4
results/reds_benchmark_val/part1/REDS_000/videos/lanczos.mp4
results/reds_benchmark_val/part1/REDS_000/videos/srcnn.mp4
results/reds_benchmark_val/part1/REDS_000/videos/temporal.mp4
results/reds_benchmark_val/part2/real_esrgan/REDS_000/videos/real_esrgan.mp4
results/reds_benchmark_val/part2/basicvsrpp/REDS_000/videos/basicvsrpp.mp4
results/reds_benchmark_val/part3/REDS_000/videos/adaptive_hybrid.mp4
```

For a stronger qualitative appendix, also include the same seven videos for:

```text
REDS_007
REDS_029
```

## Original Mandatory Sample Outputs

Part 1 videos:

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

Part 2 videos:

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

Part 3 videos:

```text
results/part3/REDS_002/videos/adaptive_hybrid.mp4
results/part3/Vimeo_00018_0043/videos/adaptive_hybrid.mp4
results/part3/wild/videos/adaptive_hybrid.mp4
```

## Notes

- The MP4 files are convenient for submission and demos.
- PNG frames in each method's `frames/` directory are better for report figures and zoom-in inspection because they avoid MP4 compression artifacts.
- The full REDS benchmark videos are ignored by git because they are generated result files. Keep them in the local `results/` directory or upload them separately if the submission portal requires processed videos.
