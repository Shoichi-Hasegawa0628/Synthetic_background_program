import math
import random

import PIL
import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.transforms import functional as F, InterpolationMode


class MultipleImageRandomReposition(nn.Module):

    def __init__(self, size):
        super(MultipleImageRandomReposition, self).__init__()
        self._size = size

    def forward(self, imgs):
        """
        Args:
            img (PIL Image or Tensor): Image to be scaled.

        Returns:
            PIL Image or Tensor: Rescaled image.
        """
        for img in imgs:
            width, height = F._get_image_size(imgs)
            scale = random.uniform(self._scale[0], self._scale[1])
            return [F.resize(img, [int(height * scale), int(width * scale)], self.interpolation, self.max_size, self.antialias) for img in imgs]
