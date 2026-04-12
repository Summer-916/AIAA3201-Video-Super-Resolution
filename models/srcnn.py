import torch
import torch.nn as nn

class SRCNN(nn.Module):
    def __init__(self, num_channels=3):
        super(SRCNN, self).__init__()
        # Patch extraction and representation
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=4)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Non-linear mapping
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        
        # Reconstruction
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=2)

    def forward(self, x):
        # x should be an LR image tensor upscaled using Bicubic interpolation
        out = self.relu1(self.conv1(x))
        out = self.relu2(self.conv2(out))
        out = self.conv3(out)
        return out