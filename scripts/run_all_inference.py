import os
import sys

# Ensure the script can find the models and utils folders in the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from models.srcnn import SRCNN
from utils.interpolation import upscale_bicubic

def process_image(img_path, output_dir, model, device, transform):
    """Process a single image: generate Bicubic and SRCNN images"""
    filename = os.path.basename(img_path)
    img = Image.open(img_path).convert('RGB')
    
    # Simulate low-resolution input (downsampling)
    lr_img = img.resize((img.width//4, img.height//4), Image.BICUBIC)
    
    # --- 1. Generate Bicubic baseline result ---
    bicubic_out = lr_img.resize((img.width, img.height), Image.BICUBIC)
    bicubic_dir = os.path.join(output_dir, "bicubic")
    os.makedirs(bicubic_dir, exist_ok=True)
    bicubic_out.save(os.path.join(bicubic_dir, filename))

    # --- 2. Generate SRCNN result ---
    if model is not None:
        # SRCNN requires the input to be upscaled to the target size using Bicubic first
        srcnn_input = lr_img.resize((img.width, img.height), Image.BICUBIC)
        input_tensor = transform(srcnn_input).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            
        srcnn_out = transforms.ToPILImage()(output.squeeze(0).cpu().clamp(0, 1))
        srcnn_dir = os.path.join(output_dir, "srcnn")
        os.makedirs(srcnn_dir, exist_ok=True)
        srcnn_out.save(os.path.join(srcnn_dir, filename))

def run_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting full dataset inference, device: {device}")

    # Load SRCNN model
    model = SRCNN().to(device)
    weights_path = "srcnn_weights.pth"
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()
    else:
        print("SRCNN weights not found, running Bicubic baseline only.")
        model = None

    transform = transforms.ToTensor()

    # Define dictionary of dataset directories to run {input_path: output_root_path}
    datasets = {
        "data/wild/extracted_frames": "results/part1/wild",
        "data/sample/REDS-sample/REDS-sample/002": "results/part1/sample/REDS/002",
        "data/sample/vimeo-RL/vimeo-RL/00018/0043": "results/part1/sample/Vimeo/00018/0043"
    }

    for in_dir, out_dir in datasets.items():
        if not os.path.exists(in_dir):
            print(f"Skipping {in_dir} (path does not exist)")
            continue
            
        print(f"\nProcessing folder: {in_dir}")
        img_files = [f for f in os.listdir(in_dir) if f.endswith('.png')]
        
        for filename in tqdm(img_files):
            process_image(os.path.join(in_dir, filename), out_dir, model, device, transform)

    print("\nAll datasets processed! You can now run evaluate_metrics.py to calculate scores.")

if __name__ == "__main__":
    run_pipeline()