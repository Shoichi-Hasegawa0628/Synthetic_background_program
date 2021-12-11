from torchvision import transforms as tf
from torchvision.transforms import functional as F


class MultipleImageColorJitter(tf.ColorJitter):

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super(MultipleImageColorJitter, self).__init__(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def forward(self, imgs):
        """
        Args:
            imgs (List of PIL Image or Tensor): Input image.

        Returns:
            List of PIL Image or Tensor: Color jittered image.

        """
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = self.get_params(self.brightness, self.contrast, self.saturation, self.hue)

        outs = []
        for img in imgs:
            for fn_id in fn_idx:
                if fn_id == 0 and brightness_factor is not None:
                    img = F.adjust_brightness(img, brightness_factor)
                elif fn_id == 1 and contrast_factor is not None:
                    img = F.adjust_contrast(img, contrast_factor)
                elif fn_id == 2 and saturation_factor is not None:
                    img = F.adjust_saturation(img, saturation_factor)
                elif fn_id == 3 and hue_factor is not None:
                    img = F.adjust_hue(img, hue_factor)
            outs.append(img)

        return outs
