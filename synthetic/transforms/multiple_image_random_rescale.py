import math
import random

import PIL
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F, InterpolationMode


class MultipleImageRandomRescale(transforms.Resize):

    def __init__(self, scale, interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=None):
        super().__init__(0, interpolation, max_size, antialias)
        self._scale = scale

    def forward(self, imgs):
        """
        Args:
            img (PIL Image or Tensor): Image to be scaled.

        Returns:
            PIL Image or Tensor: Rescaled image.
        """
        width, height = F._get_image_size(imgs[0])
        scale = random.uniform(self._scale[0], self._scale[1])
        size_x, size_y = int(width * scale), int(height * scale)

        if size_x == 0 or size_y == 0:
            size_x, size_y = int(width * self._scale[1]), int(height * self._scale[1])

        return [F.resize(img, [size_y, size_x], self.interpolation, self.max_size, self.antialias) for img in imgs]
