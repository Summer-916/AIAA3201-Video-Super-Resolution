import torch
import torch.nn as nn

class SRCNN(nn.Module):
    """Small SRCNN baseline used in Part 1.

    The original SRCNN formulation first upsamples an LR image with bicubic
    interpolation, then applies a shallow CNN to refine the HR estimate. We keep
    this model intentionally simple because Part 1 is meant to be a baseline
    against stronger video-specific models in Part 2.
    """

    def __init__(self, num_channels=3):
        super(SRCNN, self).__init__()
        # Patch extraction and representation. The 9x9 kernel gives the first
        # layer a wider receptive field, matching the classic SRCNN design.
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=4)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Non-linear mapping from high-dimensional patch features to a more
        # compact representation.
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        
        # Reconstruction back to RGB. No activation is used here; callers clamp
        # the final output into [0, 1] before converting back to an image.
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=2)

    def forward(self, x):
        # x should already be bicubic-upscaled to the desired HR size.
        out = self.relu1(self.conv1(x))
        out = self.relu2(self.conv2(out))
        out = self.conv3(out)
        return out
