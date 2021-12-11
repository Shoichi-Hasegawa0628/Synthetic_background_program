from torch import Tensor
from torchvision import transforms as tf
from torchvision.transforms import functional as F, InterpolationMode


class MultipleImageRandomRotation(tf.RandomRotation):

    def forward(self, imgs):
        """
        Args:
            imgs (List of PIL Image or Tensor): Input image.

        Returns:
            List of PIL Image or Tensor: Rotated image.

        """
        fill = self.fill
        angle = self.get_params(self.degrees)

        outs = []
        for img in imgs:
            if isinstance(img, Tensor):
                if isinstance(fill, (int, float)):
                    fill = [float(fill)] * F._get_image_num_channels(img)
                else:
                    fill = [float(f) for f in fill]

            out = F.rotate(img, angle, self.resample, self.expand, self.center, fill)
            outs.append(out)

        return outs
