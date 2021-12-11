import math
import random

import PIL
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F, InterpolationMode


class MultipleImageRandomResized(transforms.Resize):

    def __init__(self, size, interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=None):
        super().__init__(size, interpolation, max_size, antialias)

    def forward(self, imgs):
        """
        Args:
            img (PIL Image or Tensor): Image to be scaled.

        Returns:
            PIL Image or Tensor: Rescaled image.
        """
        width, height = F._get_image_size(imgs[0])
        scale = self.size / max(width, height)
        return [F.resize(img, [int(height * scale), int(width * scale)], self.interpolation, self.max_size, self.antialias) for img in imgs]
