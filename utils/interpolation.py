import cv2

def upscale_bicubic(image, scale_factor=4):
    """Upscale image using Bicubic interpolation"""
    h, w = image.shape[:2]
    return cv2.resize(image, (w * scale_factor, h * scale_factor), interpolation=cv2.INTER_CUBIC)

def upscale_lanczos(image, scale_factor=4):
    """Upscale image using Lanczos interpolation"""
    h, w = image.shape[:2]
    return cv2.resize(image, (w * scale_factor, h * scale_factor), interpolation=cv2.INTER_LANCZOS4)