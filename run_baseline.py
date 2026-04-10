import os
import cv2
from utils.data_loader import load_frames
from utils.interpolation import upscale_bicubic
from utils.temporal_baseline import temporal_average_fusion, unsharp_mask

def run_interpolation_baseline(input_dir, output_dir, scale_factor=4):
    """运行基础的空间插值 (Bicubic)"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print("🚀 开始运行 Bicubic 空间上采样基线...")
    frames = load_frames(input_dir)
    if not frames:
        print("❌ 未找到输入帧，请检查路径！")
        return []

    hr_frames = []
    for i, frame in enumerate(frames):
        # 执行双三次插值放大
        hr_frame = upscale_bicubic(frame, scale_factor)
        hr_frames.append(hr_frame)
        
        # 保存结果
        out_path = os.path.join(output_dir, f"{i:04d}_bicubic.png")
        cv2.imwrite(out_path, hr_frame)
        
    print(f"✅ Bicubic 处理完成！已保存至 {output_dir}")
    return hr_frames

def run_temporal_baseline(bicubic_frames, output_dir):
    """运行时间基线：多帧平均 + 反锐化掩膜"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print("🚀 开始运行多帧平均时间基线...")
    # 假设我们用一个简单的 3 帧滑动窗口 (前一帧，当前帧，后一帧)
    # 权重分配：中间帧权重最高，相邻帧权重点缀以去噪
    weights = [0.2, 0.6, 0.2] 
    
    num_frames = len(bicubic_frames)
    for i in range(num_frames):
        # 处理边界情况 (第一帧和最后一帧没有前/后帧)
        if i == 0 or i == num_frames - 1:
            fused = bicubic_frames[i] 
        else:
            # 提取 3 帧窗口
            window = [bicubic_frames[i-1], bicubic_frames[i], bicubic_frames[i+1]]
            # 多帧加权平均
            fused = temporal_average_fusion(window, weights)
            
        # 进阶：使用反锐化掩膜增强高频边缘
        fused_sharpened = unsharp_mask(fused, amount=1.5)
            
        out_path = os.path.join(output_dir, f"{i:04d}_temporal.png")
        cv2.imwrite(out_path, fused_sharpened)
        
    print(f"✅ 时间序列融合处理完成！已保存至 {output_dir}")

if __name__ == "__main__":
    # 1. 定义你的路径 (请确保这些路径下有你刚刚提取出来的帧)
    INPUT_FRAMES_DIR = "data/wild/extracted_frames"
    BICUBIC_OUT_DIR = "results/part1/bicubic"
    TEMPORAL_OUT_DIR = "results/part1/temporal"
    
    # 2. 运行空间上采样 (假设把视频放大 4 倍)
    hr_bicubic_frames = run_interpolation_baseline(INPUT_FRAMES_DIR, BICUBIC_OUT_DIR, scale_factor=4)
    
    # 3. 运行时间序列融合
    if hr_bicubic_frames:
        run_temporal_baseline(hr_bicubic_frames, TEMPORAL_OUT_DIR)