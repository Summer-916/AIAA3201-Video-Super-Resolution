import cv2
import numpy as np

def temporal_average_fusion(frames, weights=None):
    """
    对多帧进行加权平均 。
    frames: 经过 Bicubic 放大的连续帧列表 (List of numpy arrays)
    weights: 权重列表，例如 [0.2, 0.6, 0.2] 给中间帧更高的权重
    """
    if weights is None:
        weights = [1.0 / len(frames)] * len(frames)
    
    # 将图像转换为 float32 进行计算以避免溢出
    fused_frame = np.zeros_like(frames[0], dtype=np.float32)
    for frame, weight in zip(frames, weights):
        fused_frame += frame.astype(np.float32) * weight
        
    return np.clip(fused_frame, 0, 255).astype(np.uint8)

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """反锐化掩膜，用于增强高频边缘 """
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened