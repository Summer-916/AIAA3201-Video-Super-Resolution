# AIAA 3201 Project: Video Super-Resolution

This repository contains the official implementation for the AIAA 3201 Introduction to Computer Vision term project: Video Super-Resolution. 

## 📖 Project Overview
The goal of this project is to design a pipeline to reconstruct a high-resolution (HR) video from a low-resolution (LR) input. The system aims to enhance spatial details and maintain temporal consistency across video sequences.

## 🛠️ Environment Setup
- Python 3.x
- PyTorch
- OpenCV
- numpy

### Installation
Clone this repository and enter the directory:

> git clone [https://github.com/yourusername/AIAA3201-Video-Super-Resolution.git](https://github.com/yourusername/AIAA3201-Video-Super-Resolution.git)
> 
> cd AIAA3201-Video-Super-Resolution


## 🚀 Implementation Roadmap

### Part 1: The Baseline (Hand-crafted Approach)
- [x] **Bicubic & Lanczos Interpolation** (Spatial upsampling base)
- [x] **SRCNN** (Basic 3-layer CNN architecture designed)
- [x] **Multi-frame Averaging** (Temporal fusion with unsharp masking)

*How to run Part 1:*

1. Extract frames from your raw video:
> python utils/data_loader.py

2. Run the baseline pipeline (Bicubic and Temporal Fusion):
> python run_baseline.py


### Part 2: SOTA Reproduction (AI-driven Pipeline)
- [ ] **BasicVSR** (Feature Alignment & Reconstruction)
- [ ] **Real-ESRGAN** (Perceptual Enhancement)

### Part 3: Exploration (Optimization & Extension)
- [ ] To be determined (e.g., Generative VSR / Consistent Enhancement)

## 📁 Project Structure

* `models/` - Contains the SRCNN neural network architecture.
* `scripts/` - Core execution scripts for inference and evaluation.
* `utils/` - Helper functions for frame extraction and interpolation.
* `train_srcnn.py` - Script to train the SRCNN model.
* `data/` - (Ignored by Git) Put your raw videos and datasets here.
* `results/` - (Ignored by Git) Generated output images will be saved here.