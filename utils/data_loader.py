import cv2
import os
import glob

def extract_frames(video_path, output_dir):
    """
    Extract frames from a video and save them as images in the specified directory.
    
    Args:
        video_path (str): Path to the video file (e.g., 'data/wild/my_test_video.mp4')
        output_dir (str): Path to the folder where extracted frames will be saved
    Returns:
        list: A list containing the paths of all saved images
    """
    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read the video using OpenCV
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}. Please check if the path is correct.")
        return []

    frame_paths = []
    frame_idx = 0
    print(f"Starting to extract frames from video: {video_path}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break # Finished reading video
            
        # Format the saved filename, e.g., 0000.png, 0001.png
        # Using PNG format avoids secondary quality loss from JPEG compression
        out_path = os.path.join(output_dir, f"{frame_idx:04d}.png")
        cv2.imwrite(out_path, frame)
        frame_paths.append(out_path)
        
        # Print progress to indicate the program is still running for large videos
        if frame_idx % 50 == 0 and frame_idx > 0:
            print(f"  Extracted {frame_idx} frames...")
            
        frame_idx += 1

    cap.release()
    print(f"Extraction complete! Total {frame_idx} frames saved to: {output_dir}")
    return frame_paths

def load_frames(frame_dir):
    """
    Load the sequence of extracted frames.
    """
    # Read in alphabetical order of filenames to ensure correct chronological sequence
    frame_paths = sorted(glob.glob(os.path.join(frame_dir, "*.png")))
    frames = [cv2.imread(p) for p in frame_paths]
    return frames

# ==========================================
# Test block: This will be executed when you run the script directly
# ==========================================
if __name__ == "__main__":
    test_video = "data/wild/test_video.mp4" 
    output_frames_dir = "data/wild/extracted_frames"
    
    # Run extraction logic (you can replace test_video with your actual file path to test)
    # It will run if the file exists
    if os.path.exists(test_video):
        extract_frames(test_video, output_frames_dir)
    else:
        print(f"Test video {test_video} does not exist. Please place a video file first to test.")