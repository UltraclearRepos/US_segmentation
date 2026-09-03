"""Dataset for grayscale ultrasound segmentation."""

import random
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    image_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

    def __init__(self, pairs, label_mapping, data_config, augment):
        self.pairs = pairs
        self.label_mapping = label_mapping
        self.image_size = data_config["image_size"]
        self.augmentation = data_config["augmentation"]
        self.augment = augment and self.augmentation["enabled"]

    def _load_image(self, path, dtype=np.float32):
        with Image.open(path) as image:
            return np.asarray(image.convert("L"), dtype=dtype)

    def _resize(self, array, is_mask=False):
        height, width = self.image_size
        image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), "L")
        mode = Image.Resampling.NEAREST if is_mask else Image.Resampling.BILINEAR
        return np.asarray(image.resize((width, height), mode), dtype=np.uint8 if is_mask else np.float32)

    def _remap_mask(self, mask):
        mapped = np.zeros_like(mask, dtype=np.int64)
        for raw_value, class_id in self.label_mapping.items():
            mapped[mask == raw_value] = class_id
        return mapped

    def _augment(self, image, mask):
        if random.random() < self.augmentation["flip_probability"]:
            image, mask = np.flip(image, 1).copy(), np.flip(mask, 1).copy()
        if random.random() < self.augmentation["flip_probability"]:
            image, mask = np.flip(image, 0).copy(), np.flip(mask, 0).copy()
        if random.random() < self.augmentation["rot90_probability"]:
            k = random.randint(0, 3)
            image, mask = np.rot90(image, k).copy(), np.rot90(mask, k).copy()
        if random.random() < self.augmentation["intensity_probability"]:
            image = np.clip(
                image * random.uniform(0.85, 1.15) + random.uniform(-0.05, 0.05), 0, 1
            )
        if random.random() < self.augmentation["noise_probability"]:
            image = np.clip(
                image + np.random.normal(0, 0.02, image.shape), 0, 1
            ).astype(np.float32)
        return image, mask

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]
        image = np.clip(
            self._resize(self._load_image(image_path)) / 255, 0, 1
        ).astype(np.float32)
        mask = self._remap_mask(
            self._resize(self._load_image(mask_path, np.uint8), is_mask=True)
        )
        if self.augment:
            image, mask = self._augment(image, mask)
        return torch.from_numpy(image[None]).float(), torch.from_numpy(mask).long()
