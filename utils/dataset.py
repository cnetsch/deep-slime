from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
from pathlib import Path
import os
import random
# from kornia.geometry.transform import warp_affine


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir=None, scale=1, mask_suffix="_mask"):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.config = {
            "scale_range": [0.5, 2.0],
            "shear_range": [0.8, 1.2],
            "rot_range": [0, 360],
            "tile_size": 100,
        }
        self.augment = False
        
        self.training = masks_dir is not None

        assert 0 < scale <= 1, "Scale must be between 0 and 1"

        self.ids = [
            splitext(file)[0] for file in listdir(imgs_dir) if not file.startswith(".")
        ]
        logging.info(f"Creating dataset with {len(self.ids)} examples")

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def random_float(range: list) -> float:
        return random.uniform(range[0], range[1])

    # def get_random_tile(self, img, lbl):
    #     scale = self.random_float(self.config["scale_range"])
    #     shear = self.random_float(self.config["shear_range"])
    #     rot = self.random_float(self.config["rot_range"])
    #     sin = torch.sin(torch.deg2rad(rot))
    #     cos = torch.cos(torch.deg2rad(rot))
    #     size = self.config["tile_size"]
    #     center_x = self.random_float((size / 2, img.shape[-2] - (size / 2)))
    #     center_y = self.random_float((size / 2, img.shape[-1] - (size / 2)))

    #     Msc = torch.FloatTensor([[scale, 0, 0], [0, scale, 0], [0, 0, 1]])
    #     Msh = torch.FloatTensor([[1, shear, 0], [shear, 1, 0], [0, 0, 1]])
    #     Mt1 = torch.FloatTensor([[1, 0, -center_x], [0, 1, -center_y], [0, 0, 1]])
    #     Mt2 = torch.FloatTensor([[1, 0, size / 2.0], [0, 1, size / 2.0], [0, 0, 1]])
    #     Mrt = torch.FloatTensor([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])

    #     affine_trans = torch.mm(
    #         torch.mm(torch.mm(torch.mm(Mt2, Msc), Msh), Mrt), Mt1
    #     )  # M = Mt2 @ Ms @ Msh @ Mr @ Mt1

    #     img = warp_affine(img, affine_trans[:2, :].unsqueeze(0), (size, size))
    #     lbl = warp_affine(lbl, affine_trans[:2, :].unsqueeze(0), (size, size))

    #     return img, lbl

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size

        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, "Scale is too small"
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        if self.training:
            mask_file = glob(self.masks_dir + idx + self.mask_suffix + ".*")
            assert (
                len(mask_file) == 1
            ), f"Either no mask or multiple masks found for the ID {idx}: {mask_file}"
            mask = Image.open(mask_file[0])


        img_file = glob(self.imgs_dir + idx + ".*")
        assert (
            len(img_file) == 1
        ), f"Either no image or multiple images found for the ID {idx}: {img_file}"
        img = Image.open(img_file[0])

        if self.training:
            assert (
                img.size == mask.size
            ), f"Image and mask {idx} should be the same size, but are {img.size} and {mask.size}"

            img = self.preprocess(img, self.scale)
            mask = self.preprocess(mask, self.scale)

            if self.augment:
                # img, mask = self.get_random_tile(img, mask)
                logging.info("Currently no augmentations implemented.")

            return {
                "image": torch.from_numpy(img).type(torch.FloatTensor),
                "mask": torch.from_numpy(mask).type(torch.FloatTensor),
            }
            
        else:
            img = self.preprocess(img, self.scale)
            return {
                "image": torch.from_numpy(img).type(torch.FloatTensor),
                "name": Path(img_file[0]).stem
            }

class BiofilmDataset(BasicDataset):
    MASK_IDENTIFIER = "_binary_remove outliers"

    def __init__(self, imgs_dir, masks_dir=None, scale=1):
        super().__init__(imgs_dir, masks_dir, scale, mask_suffix="")

    @staticmethod
    def rename_files(path, *replace):
        for filename in os.listdir(path):
            if ".keep" in filename:
                continue
            for r in replace:
                src = filename
                dst = filename.replace(r[0], r[1])
                os.rename(os.path.join(path, src), os.path.join(path, dst))
