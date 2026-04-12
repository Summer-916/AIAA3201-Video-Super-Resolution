import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from PIL import Image
from tqdm import tqdm

def calculate_metrics(gt_dir, test_dir):
    """Calculate average PSNR and SSIM for corresponding images in two directories"""
    if not os.path.exists(test_dir):
        return 0.0, 0.0
        
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith('.png')])
    test_files = sorted([f for f in os.listdir(test_dir) if f.endswith('.png')])
    
    # Ensure the files match up
    common_files = set(gt_files).intersection(test_files)
    if not common_files:
        return 0.0, 0.0

    total_psnr, total_ssim = 0.0, 0.0
    count = 0

    for filename in common_files:
        gt_img = np.array(Image.open(os.path.join(gt_dir, filename)).convert('RGB'))
        test_img = np.array(Image.open(os.path.join(test_dir, filename)).convert('RGB'))
        
        # Ensure dimensions match before calculating metrics
        if gt_img.shape == test_img.shape:
            # Calculate PSNR
            total_psnr += psnr(gt_img, test_img, data_range=255)
            # Calculate SSIM (channel_axis=-1 indicates RGB images)
            total_ssim += ssim(gt_img, test_img, data_range=255, channel_axis=-1)
            count += 1

    return (total_psnr / count), (total_ssim / count) if count > 0 else (0, 0)

def main():
    print("Starting evaluation metrics calculation (PSNR & SSIM)...")
    
    # Using Sequence 002 from REDS-sample as an example for evaluation
    # Note: Please ensure the images in these paths have already been generated!
    gt_path = "data/sample/REDS-sample/REDS-sample/002"
    bicubic_path = "results/part1/sample/REDS/002/bicubic"  # Assuming your Bicubic results are stored here
    srcnn_path = "results/part1/sample/REDS/002/srcnn"      # Assuming your SRCNN results are stored here

    print("\n--- REDS-sample (Sequence 002) Evaluation Results ---")
    
    # 1. Calculate Bicubic scores
    bicubic_psnr, bicubic_ssim = calculate_metrics(gt_path, bicubic_path)
    print(f"Bicubic -> PSNR: {bicubic_psnr:.2f} dB, SSIM: {bicubic_ssim:.4f}")
    
    # 2. Calculate SRCNN scores
    srcnn_psnr, srcnn_ssim = calculate_metrics(gt_path, srcnn_path)
    print(f"SRCNN   -> PSNR: {srcnn_psnr:.2f} dB, SSIM: {srcnn_ssim:.4f}")
    
    print("\nYou can plug these numbers directly into your PPT tables!")

if __name__ == "__main__":
    main()