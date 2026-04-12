import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Assuming your previously written SRCNN is in models/srcnn.py
from models.srcnn import SRCNN 

# 1. Define data loader (feeding your REDS-sample to the model)
class MiniREDSDataset(Dataset):
    def __init__(self, image_dir):
        # Read all PNG images in the directory
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')]
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        hr_img = Image.open(self.image_paths[idx]).convert('RGB')
        # To avoid freezing the local machine, crop a small 256x256 patch for training
        hr_img = transforms.CenterCrop(256)(hr_img)
        
        # SRCNN specific requirement: downscale first (create low-res), then upscale back to original size using Bicubic, allowing the model to learn how to sharpen blurred images
        lr_img = hr_img.resize((64, 64), Image.BICUBIC).resize((256, 256), Image.BICUBIC)
        
        return self.transform(lr_img), self.transform(hr_img)

def train():
    # 2. Basic settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Point to a specific folder in the REDS-sample you extracted earlier (e.g., 002)
    data_dir = "data/sample/REDS-sample/REDS-sample/002" 
    dataset = MiniREDSDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 3. Initialize model, loss function, and optimizer
    model = SRCNN().to(device)
    criterion = nn.MSELoss() # Mean Squared Error loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 4. Start training (run 5 epochs for a quick test)
    epochs = 5
    print("Starting SRCNN training...")
    for epoch in range(epochs):
        epoch_loss = 0
        for lr_imgs, hr_imgs in dataloader:
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
            
            # Forward pass
            outputs = model(lr_imgs)
            loss = criterion(outputs, hr_imgs)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(dataloader):.6f}")

    # 5. Save the trained model weights
    torch.save(model.state_dict(), "srcnn_weights.pth")
    print("Training complete! Model saved as srcnn_weights.pth")

if __name__ == "__main__":
    train()