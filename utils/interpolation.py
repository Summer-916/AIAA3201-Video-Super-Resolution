import cv2

def upscale_bicubic(image, scale_factor=4):
    """使用双三次插值 (Bicubic) 放大图像 """
    h, w = image.shape[:2]
    return cv2.resize(image, (w * scale_factor, h * scale_factor), interpolation=cv2.INTER_CUBIC)

def upscale_lanczos(image, scale_factor=4):
    """使用 Lanczos 插值放大图像 """
    h, w = image.shape[:2]
    return cv2.resize(image, (w * scale_factor, h * scale_factor), interpolation=cv2.INTER_LANCZOS4)
