import random

from torchvision import transforms as tf
from torchvision.transforms import functional as F


class MultipleImageRandomGaussianBlur(tf.GaussianBlur):

    def __init__(self, p=0.5, *args, **kwargs):
        super(MultipleImageRandomGaussianBlur, self).__init__(*args, **kwargs)
        self._p = p

    def forward(self, imgs):
        """
        Args:
            imgs (PIL Image or Tensor): image to be blurred.

        Returns:
            PIL Image or Tensor: Gaussian blurred image
        """
        if random.random() < self._p:
            sigma = self.get_params(self.sigma[0], self.sigma[1])
            return [F.gaussian_blur(img, self.kernel_size, [sigma, sigma]) for img in imgs]
        else:
            return imgs
