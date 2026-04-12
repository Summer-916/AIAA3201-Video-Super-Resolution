import cv2
import numpy as np

def temporal_average_fusion(frames, weights=None):
    """
    Perform weighted average fusion on multiple frames.
    frames: List of consecutive frames upscaled using Bicubic interpolation (List of numpy arrays)
    weights: List of weights, e.g., [0.2, 0.6, 0.2] gives higher weight to the center frame
    """
    if weights is None:
        weights = [1.0 / len(frames)] * len(frames)
    
    # Convert images to float32 for calculation to avoid overflow
    fused_frame = np.zeros_like(frames[0], dtype=np.float32)
    for frame, weight in zip(frames, weights):
        fused_frame += frame.astype(np.float32) * weight
        
    return np.clip(fused_frame, 0, 255).astype(np.uint8)

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Unsharp masking, used to enhance high-frequency edges"""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened