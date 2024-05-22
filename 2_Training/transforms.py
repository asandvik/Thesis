import torch
import torch.nn as nn

# https://github.com/pytorch/vision/blob/main/references/video_classification/transforms.py
class ConvertBCHWtoCBHW(nn.Module):
    """Convert tensor from (B, C, H, W) to (C, B, H, W)"""

    def forward(self, vid: torch.Tensor) -> torch.Tensor:
        return vid.permute(1, 0, 2, 3)