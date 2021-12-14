import shutil
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from ycb_dataset import YCBDataset
import numpy as np


def save_data(root_path, dataset, iter, s=0):
    save_rgb_image_dir = root_path.joinpath("images")
    save_rgb_image_dir.mkdir()
    save_mask_image_dir = root_path.joinpath("masks")
    save_mask_image_dir.mkdir()
    save_label_dir = root_path.joinpath("labels")
    save_label_dir.mkdir()

    for i in tqdm(range(iter)):
        for k in tqdm(range(len(dataset))):
            file_name = f"{i + s}_{k}"
            image_file_name = f"{file_name}.png"
            txt_file_name = f"{file_name}.txt"

            rgb, mask, bounding_boxes, classes = dataset[k]
            rgb = rgb.numpy()
            rgb = rgb.transpose((1, 2, 0))
            Image.fromarray(rgb, mode="RGB").save(save_rgb_image_dir.joinpath(image_file_name))

            mask = mask.numpy()[0].astype(np.uint8)
            # mask = YCBDataset.index_image_to_rgb_image(mask)
            Image.fromarray(mask, mode="L").save(save_mask_image_dir.joinpath(image_file_name))

            img_height, img_width = rgb.shape[:2]
            with open(save_label_dir.joinpath(txt_file_name), "w") as f:
                for object_class, bounding_box in zip(classes, bounding_boxes):
                    x, y, w, h = bounding_box
                    center_x, center_y, width, height = dataset._bounding_box_to_yolo(*bounding_box, img_width, img_height)
                    f.write(" ".join(map(str, [object_class, center_x, center_y, width, height])))
                    f.write("\n")


if __name__ == '__main__':
    root = Path.home().joinpath("HSR/datasets/frontiers2021_object")
    objects_dir = root.joinpath("objects")
    background_dir_dir = root.joinpath("backgrounds")

    save_root_dir = Path.home().joinpath("HSR/datasets/frontiers2021_object_for_yolov5")
    if save_root_dir.exists():
        shutil.rmtree(save_root_dir)
    save_root_dir.mkdir(parents=True)

    dataset = YCBDataset(objects_dir, background_dir_dir, scale1=(0.2, 0.5), scale2=(0.2, 0.5))
    root_val_dir = save_root_dir.joinpath("val")
    root_val_dir.mkdir()
    save_data(root_val_dir, dataset, iter=3)

    dataset = YCBDataset(objects_dir, background_dir_dir, scale1=(0.2, 0.5), scale2=(0.2, 0.5))
    root_train_dir = save_root_dir.joinpath("train")
    root_train_dir.mkdir()
    save_data(root_train_dir, dataset, iter=10)
