"""PyTorch dataset for an already prepared segmentation dataset."""

from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset



class SegmentationDataset(Dataset):
    def __init__(
        self, dataset_dir, sample_ids, image_size, num_classes, augmentation=None
    ):
        self.image_size = image_size
        self.num_classes = num_classes
        self.augmentation = augmentation

        dataset_dir = Path(dataset_dir)
        manifest = pd.read_csv(dataset_dir / "manifest.csv").set_index("sample_id")
        self.samples = manifest.loc[sample_ids].copy()
        for column in ("image_path", "mask_path"):
            self.samples[column] = self.samples[column].map(
                lambda path: dataset_dir / path
            )


    def _resize(self, array, is_mask):
        height, width = self.image_size
        image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), "L")
        mode = Image.Resampling.NEAREST if is_mask else Image.Resampling.BILINEAR
        dtype = np.uint8 if is_mask else np.float32
        return np.asarray(image.resize((width, height), mode), dtype=dtype)

    
    def _load_image(self, path):
        with Image.open(path) as image:
            array = np.asarray(image.convert("L"), dtype=np.float32)
        array = self._resize(array, is_mask=False)
        return np.clip(array / 255.0, 0.0, 1.0).astype(np.float32)


    def _load_mask(self, path):
        with Image.open(path) as image:
            mask = np.asarray(image.convert("L"), dtype=np.int64)

        invalid = np.unique(mask[(mask < 0) | (mask >= self.num_classes)])
        if invalid.size:
            raise ValueError(f"Invalid class IDs {invalid.tolist()} in mask: {path}")
        return mask


    def iter_masks(self):
        for path in self.samples["mask_path"]:
            mask = self._load_mask(path)
            yield self._resize(mask, is_mask=True).astype(np.int64)


    def _augment(self, image, mask):
        if random.random() < self.augmentation["flip_probability"]:
            image = np.flip(image, 1).copy()
            mask = np.flip(mask, 1).copy()
        if random.random() < self.augmentation["flip_probability"]:
            image = np.flip(image, 0).copy()
            mask = np.flip(mask, 0).copy()
        if random.random() < self.augmentation["rot90_probability"]:
            k = random.randint(0, 3)
            image = np.rot90(image, k).copy()
            mask = np.rot90(mask, k).copy()
        if random.random() < self.augmentation["intensity_probability"]:
            scale = random.uniform(0.85, 1.15)
            shift = random.uniform(-0.05, 0.05)
            image = np.clip(image * scale + shift, 0, 1)
        if random.random() < self.augmentation["noise_probability"]:
            noise = np.random.normal(0, 0.02, image.shape)
            image = np.clip(image + noise, 0, 1).astype(np.float32)
        return image, mask


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, index):
        sample = self.samples.iloc[index]
        image = self._load_image(sample["image_path"])
        original_mask = self._load_mask(sample["mask_path"])
        mask = self._resize(original_mask, is_mask=True).astype(np.int64)

        if self.augmentation and self.augmentation["enabled"]:
            image, mask = self._augment(image, mask)

        image_tensor = torch.from_numpy(image).float()
        image_tensor = image_tensor.unsqueeze(0)

        mask_tensor = torch.from_numpy(mask).long()
        original_mask_tensor = torch.from_numpy(original_mask).long()

        return image_tensor, mask_tensor, original_mask_tensor
