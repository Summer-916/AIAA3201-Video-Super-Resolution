import cv2
import os
import glob

def extract_frames(video_path, output_dir):
    """
    将视频按帧提取为图片并保存到指定目录。
    
    参数:
        video_path (str): 视频文件的路径 (例如 'data/wild/my_test_video.mp4')
        output_dir (str): 提取出的帧要保存的文件夹路径
    返回:
        list: 包含所有已保存图片路径的列表
    """
    # 如果输出目录不存在，则创建它
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 使用 OpenCV 读取视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 错误：无法打开视频文件 {video_path}，请检查路径是否正确！")
        return []

    frame_paths = []
    frame_idx = 0
    print(f"🎬 开始拆解视频: {video_path}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break # 视频读取完毕
            
        # 格式化保存的文件名，例如 0000.png, 0001.png
        # 使用 PNG 格式可以避免 JPEG 压缩带来的二次画质损失
        out_path = os.path.join(output_dir, f"{frame_idx:04d}.png")
        cv2.imwrite(out_path, frame)
        frame_paths.append(out_path)
        
        # 打印进度，防止处理大视频时以为程序卡死了
        if frame_idx % 50 == 0 and frame_idx > 0:
            print(f"  已提取 {frame_idx} 帧...")
            
        frame_idx += 1

    cap.release()
    print(f"✅ 提取完成！共计 {frame_idx} 帧，已保存至: {output_dir}")
    return frame_paths

def load_frames(frame_dir):
    """
    读取已经提取好的帧序列。
    """
    # 按照文件名顺序排序读取，确保时序正确
    frame_paths = sorted(glob.glob(os.path.join(frame_dir, "*.png")))
    frames = [cv2.imread(p) for p in frame_paths]
    return frames

# ==========================================
# 测试代码块：当你直接运行这个脚本时会执行这里
# ==========================================
if __name__ == "__main__":
    test_video = "data/wild/test_video.mp4" 
    output_frames_dir = "data/wild/extracted_frames"
    
    # 运行提取逻辑 (你可以把 test_video 换成你实际的文件路径来测试)
    # 如果文件存在，它就会跑起来
    if os.path.exists(test_video):
        extract_frames(test_video, output_frames_dir)
    else:
        print(f"⚠️ 测试视频 {test_video} 不存在。请先放入一个视频文件进行测试。")