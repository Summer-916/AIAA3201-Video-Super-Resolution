import torch
import torch.nn as nn

class SRCNN(nn.Module):
    def __init__(self, num_channels=3):
        super(SRCNN, self).__init__()
        # 提取图像块并表示 (Patch extraction and representation)
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=4)
        self.relu1 = nn.ReLU(inplace=True)
        
        # 非线性映射 (Non-linear mapping)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        
        # 重建 (Reconstruction)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=2)

    def forward(self, x):
        # x 应该是先经过 Bicubic 放大的 LR 图像张量
        out = self.relu1(self.conv1(x))
        out = self.relu2(self.conv2(out))
        out = self.conv3(out)
        return out