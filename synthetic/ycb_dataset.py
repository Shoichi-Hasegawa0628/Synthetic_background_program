import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as tf
from tqdm import tqdm
from natsort import natsorted

from color import Color
from ycb_dataset_info import YCBDatasetInfo
from transforms import (MultipleImageColorJitter, MultipleImageRandomRotation, MultipleImageRandomResized, MultipleImageRandomGaussianBlur,
                        MultipleImageRandomRescale)


class YCBDataset(Dataset):

    def __init__(self, objects_dir, background_dir, scale1, scale2):
        self._object_image_paths = self._load_object_paths(objects_dir)
        self._n_object_images = len(self._object_image_paths)
        self._background_image_paths = self._load_background_paths(background_dir)
        self._n_background_images = len(self._background_image_paths)

        self._n_additional_background = 100

        self.common_transforms = tf.Compose([
            # MultipleImageColorJitter(brightness=0.05, contrast=0.05, saturation=0.05),
            # MultipleImageRandomGaussianBlur(p=0.1, kernel_size=(5, 9), sigma=(0.1, 5))
        ])
        self.background_transforms = tf.Compose([
            # tf.RandomHorizontalFlip(p=0.5),
            # tf.RandomVerticalFlip(p=0.5),
            # tf.ColorJitter(hue=(-0.2, 0.2)),
        ])
        self.after_transforms = tf.Compose([
            # tf.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=(-0.01, 0.01)),
        ])
        self.object_rgb_transforms = tf.Compose([
            tf.ColorJitter(brightness=(0.8, 1.2)),
        ])
        self.object_transforms1 = tf.Compose([
            # MultipleImageRandomRotation(degrees=(0, 360), expand=True),
            MultipleImageRandomResized(size=200),
        ])
        self.object_transforms2 = tf.Compose([
            MultipleImageRandomResized(size=200),
        ])

        self.object_transforms3 = tf.Compose([
            MultipleImageRandomRescale(scale=scale1)
        ])

        self.object_transforms4 = tf.Compose([
            MultipleImageRandomRescale(scale=scale2)
        ])

        self._to_tensor = tf.PILToTensor()
        self._object_image_paths = random.sample(self._object_image_paths, len(self._object_image_paths))



    # ==================================================================================================
    #
    #   Property
    #
    # ==================================================================================================
    def __len__(self):
        return len(self._background_image_paths) + self._n_additional_background

    def __getitem__(self, index):
        if index < self._n_background_images:
            background_path = self._background_image_paths[index]
            background_image = self._load_background_image(background_path)
            indexes = np.random.randint(0, self._n_object_images, size=random.randint(1, 8))

        else:
            background_image = Image.new(mode="RGB", size=(640, 480), color=tuple(Color.index_to_rgb(random.randint(0, 255))))
            indexes = np.random.randint(0, self._n_object_images, size=50)

        object_paths = [self._object_image_paths[i] for i in indexes]

        object_rgbs, object_masks = self._load_object_images(object_paths)
        object_classes = [path[0] for path in object_paths]

        background_image, object_rgbs, object_masks, object_classes = self._transform(background_image, object_rgbs, object_masks, object_classes)

        object_rectangles = self._generate_object_area(background_image, object_rgbs)
        mask, is_visibles = self._proto_synthetic_object_masks(background_image, object_masks, object_classes, object_rectangles)
        if False in set(is_visibles):
            object_rgbs = [v for v, b in zip(object_rgbs, is_visibles) if b]
            object_masks = [v for v, b in zip(object_masks, is_visibles) if b]
            object_classes = [v for v, b in zip(object_classes, is_visibles) if b]
            object_rectangles = [v for v, b in zip(object_rectangles, is_visibles) if b]

            mask = self._synthetic_object_masks(background_image, object_masks, object_classes, object_rectangles)

        rgb = self._synthetic_object_rgbs(background_image, object_rgbs, object_rectangles)
        rgb = self.after_transforms(rgb)
        bounding_boxes = [self._rectangle_to_bounding_box(*object_rectangle) for object_rectangle in object_rectangles]

        return rgb, mask, bounding_boxes, object_classes

    # ==================================================================================================
    #
    #   Instance Method (Public)
    #
    # ==================================================================================================
    def _transform(self, background, object_rgbs, object_masks, object_classes):

        rgb_outs, mask_outs, class_outs = [], [], []

        common_transformed = self.common_transforms([background] + object_rgbs)

        background = self.background_transforms(common_transformed[0])
        background = self._to_tensor(background)

        object_rgbs = [self.object_rgb_transforms(rgb) for rgb in common_transformed[1:]]

        for object_rgb, object_mask, object_class in zip(object_rgbs, object_masks, object_classes):
            object_rgb, object_mask = self.object_transforms1([object_rgb, object_mask])
            object_rgb, object_mask = self._to_tensor(object_rgb), self._to_tensor(object_mask)

            min_x, min_y, max_x, max_y = self._calc_necessary_area_rectangle(condition=object_mask[0] > 0)
            object_rgb = self._cutout_image(object_rgb, min_x, min_y, max_x, max_y)
            object_mask = self._cutout_image(object_mask, min_x, min_y, max_x, max_y)

            # if min_x == max_x or min_y == max_y:
            #     continue

            object_rgb, object_mask = self.object_transforms2([object_rgb, object_mask])
            if random.random() < 0.95:
                object_rgb, object_mask = self.object_transforms3([object_rgb, object_mask])
            else:
                object_rgb, object_mask = self.object_transforms4([object_rgb, object_mask])

            rgb_outs.append(object_rgb)
            mask_outs.append(object_mask)
            class_outs.append(object_class)

        return background, rgb_outs, mask_outs, class_outs

    # ==================================================================================================
    #
    #   Class Method (Private)
    #
    # ==================================================================================================
    @staticmethod
    def _generate_object_area(background, ycb_objects):
        _, bg_height, bg_width = background.shape

        rectangles = []
        for ycb_object in ycb_objects:
            _, obj_height, obj_width = ycb_object.shape
            x = random.randint(0, bg_width - obj_width)
            y = random.randint(239, bg_height - obj_height)

            rectangle = (x, y, x + obj_width, y + obj_height)
            rectangles.append(rectangle)

        return rectangles

    @classmethod
    def _proto_synthetic_object_masks(cls, background, object_masks, object_classes, object_rectangles):
        mask = torch.zeros((1, *background.shape[1:]), dtype=torch.int32)

        for object_mask, object_rectangle, object_class in zip(object_masks, object_rectangles, object_classes):
            sx, sy, ex, ey = object_rectangle

            # mask[:, sy:ey, sx:ex][object_mask > 0] = 200
            mask[:, sy:ey, sx:ex][object_mask > 128] = object_class + 1

        is_visibles = []
        for i, (object_rectangle, object_class) in enumerate(zip(object_rectangles, object_classes)):
            sx, sy, ex, ey = object_rectangle
            is_visibles.append(cls._is_area_visible(mask[0, sy:ey, sx:ex] == object_class + 1))

        return mask, is_visibles

    @classmethod
    def _synthetic_object_masks(cls, background, object_masks, object_classes, object_rectangles):
        mask = torch.zeros((1, *background.shape[1:]), dtype=torch.int32)

        for object_mask, object_rectangle, object_class in zip(object_masks, object_rectangles, object_classes):
            sx, sy, ex, ey = object_rectangle
            # mask[:, sy:ey, sx:ex][object_mask > 0] = 256
            mask[:, sy:ey, sx:ex][object_mask > 128] = object_class + 1

        return mask

    @classmethod
    def _synthetic_object_rgbs(cls, background, object_rgbs, object_rectangles):
        rgb = background.clone()

        for object_rgb, object_rectangle in zip(object_rgbs, object_rectangles):

            sx, sy, ex, ey = object_rectangle
            alpha = object_rgb[3:] / 255.0
            rgb[:, sy:ey, sx:ex] = cls._alpha_blend(rgb[:, sy:ey, sx:ex], object_rgb[:3], alpha)

        return rgb

    @staticmethod
    def _alpha_blend(bg, fg, fg_alpha):
        return (bg * (1 - fg_alpha)) + (fg * fg_alpha)

    @staticmethod
    def _rectangle_to_bounding_box(sx, sy, ex, ey):
        return sx, sy, ex - sx, ey - sy

    @staticmethod
    def _bounding_box_to_yolo(box_x, box_y, box_width, box_height, img_width, img_height):
        center_x = (box_x + box_width / 2) / img_width
        center_y = (box_y + box_height / 2) / img_height

        width = box_width / img_width
        height = box_height / img_height

        return center_x, center_y, width, height

    @staticmethod
    def _yolo_to__bounding_box(center_x, center_y, box_width, box_height, img_width, img_height):
        box_width = box_width * img_width
        box_height = box_height * img_height

        box_x = (center_x * img_width - box_width / 2)
        box_y = (center_y * img_height - box_height / 2)

        return int(box_x), int(box_y), int(box_width), int(box_height)

    # ==================================================================================================
    #
    #   Instance Method (Private)
    #
    # ==================================================================================================
    @staticmethod
    def _calc_necessary_area_rectangle(condition):
        axis_x = torch.any(condition, dim=0)
        axis_y = torch.any(condition, dim=1)

        axis_x = np.where(axis_x)[0]
        axis_y = np.where(axis_y)[0]

        min_x, max_x = axis_x[0], axis_x[-1]
        min_y, max_y = axis_y[0], axis_y[-1]

        return min_x, min_y, max_x, max_y

    @staticmethod
    def _is_area_visible(condition):
        axis_x = torch.any(condition, dim=0)
        axis_y = torch.any(condition, dim=1)

        axis_x = np.where(axis_x)[0]
        axis_y = np.where(axis_y)[0]
        if len(axis_x) > 20 and len(axis_y) > 20:
            return True
        return False

    @staticmethod
    def _cutout_image(image, min_x, min_y, max_x, max_y):
        return image[:, min_y:max_y, min_x:max_x]

    @staticmethod
    def index_image_to_rgb_image(image):
        """
        Args:
            image (np.ndarray):

        Returns:
            np.ndarray:

        Shapes:
            -> [H, W] or [1, H, W]
            <- [3, H, W]

        """
        H, W = image.shape
        color_image = np.zeros(shape=(H, W, 3), dtype=np.uint8)
        labels = np.unique(image)

        rgbs = Color.indexes_to_rgbs(labels)
        for label, rgb in zip(labels, rgbs):
            color_image[image == label] = rgb

        return color_image

    @staticmethod
    def _load_background_paths(background_dir):
        return [path for path in tqdm(list(natsorted(background_dir.iterdir())))]

    @staticmethod
    def _load_object_paths(objects_dir):
        """

        Args:
            object_dir (Path):

        Returns:

        """
        rgb_dict = {}

        for object_dir in tqdm(list(natsorted(objects_dir.iterdir()))):
            object_no = YCBDatasetInfo.object_id_to_index_dict[int(object_dir.name.split("_")[1])]


            if object_no not in rgb_dict:
                rgb_dict[object_no] = []

            rgb_images = natsorted(object_dir.joinpath("rgb").iterdir())
            mask_images = natsorted(object_dir.joinpath("mask").iterdir())
            #print("RGB:{}, Mask:{}\n".format(rgb_images, mask_images))

            rgb_dict[object_no] += list(zip(map(str, rgb_images), map(str, mask_images)))

        n_max = max([len(value) for value in rgb_dict.values()])
        paths = []
        for key, value in rgb_dict.items():
            # rgb_values, mask_values = rgb_dict[key]

            value = value * (n_max // len(value))

            keys = [key] * len(value)
            paths += list(zip(keys, value))

        return paths

    @staticmethod
    def _load_object_images(paths):
        object_rgbs = [Image.open(path[1][0]).convert("RGBA") for path in paths]
        object_masks = [Image.open(path[1][1]).convert("L") for path in paths]

        return object_rgbs, object_masks

    @staticmethod
    def _load_background_image(path):
        return Image.open(path).convert("RGB")
