"""PyTorch dataset for an already prepared segmentation dataset."""

from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset



class SegmentationDataset(Dataset):
    def __init__(
        self, dataset_dir, sample_ids, image_size, num_classes, task_name, augmentation=None
    ):
        self.image_size = image_size
        self.num_classes = num_classes
        self.task_name = task_name
        self.augmentation = augmentation

        dataset_dir = Path(dataset_dir)
        manifest = pd.read_csv(dataset_dir / "manifest.csv").set_index("sample_id", drop=False)
        self.samples = manifest.loc[sample_ids].copy()
        for column in ("image_path", "mask_path"):
            self.samples[column] = self.samples[column].map(
                lambda path: dataset_dir / path
            )

        self.study_by_sample = {}
        patients_path = dataset_dir / "patients.csv"
        if not patients_path.exists():
            raise FileNotFoundError(f"Missing patients.csv in dataset directory: {dataset_dir}")

        patients = pd.read_csv(patients_path, dtype={"patient": str, "study": str})
        self._validate_patients(patients)
        self.study_by_sample = {
            row.sample_id: (row.patient, row.study)
            for row in patients.itertuples(index=False)
        }


    def _validate_patients(self, patients):
        required_columns = {"patient", "study", "sample_id"}
        if not required_columns.issubset(set(patients.columns)):
            raise ValueError(f"Missing required columns in patients.csv: {required_columns - set(patients.columns)}")

        empty_cells = patients.isna() | patients.astype(str).apply(lambda column: column.str.strip().eq(""))
        if empty_cells.any().any():
            empty_rows = patients[empty_cells.any(axis=1)]
            raise ValueError(f"Empty cells found in patients.csv:\n{empty_rows}")

        if patients["sample_id"].duplicated().any():
            duplicated_rows = patients[patients["sample_id"].duplicated(keep=False)]
            raise ValueError(f"Duplicate sample_id values found in patients.csv:\n{duplicated_rows}")

        missing_samples = set(self.samples.index) - set(patients["sample_id"])
        if missing_samples:
            raise ValueError(f"Missing sample_id values in patients.csv: {missing_samples}")


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
        # Horizontal flip
        if random.random() < self.augmentation["flip"]["probability"]:
            image = np.flip(image, 1).copy()
            mask = np.flip(mask, 1).copy()

        # Rotation
        if random.random() < self.augmentation["rotation"]["probability"]:
            max_angle = self.augmentation["rotation"]["max_angle"]
            angle = random.uniform(-max_angle, max_angle)

            image_pil = Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8))
            mask_pil = Image.fromarray(mask.astype(np.uint8), "L")

            image_pil = image_pil.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=0)
            mask_pil = mask_pil.rotate(angle, resample=Image.Resampling.NEAREST, fillcolor=0)

            image = np.asarray(image_pil, dtype=np.float32) / 255.0
            mask = np.asarray(mask_pil, dtype=np.int64)

        # Intensity scaling and shifting
        if random.random() < self.augmentation["intensity"]["probability"]:
            scale_min = self.augmentation["intensity"]["scale_min"]
            scale_max = self.augmentation["intensity"]["scale_max"]
            shift_range = self.augmentation["intensity"]["shift"]
            scale = random.uniform(scale_min, scale_max)
            shift = random.uniform(-shift_range, shift_range)
            image = np.clip(image * scale + shift, 0, 1)

        # Gamma
        if random.random() < self.augmentation["gamma"]["probability"]:
            gamma = random.uniform(self.augmentation["gamma"]["min_gamma"], self.augmentation["gamma"]["max_gamma"])
            image = np.clip(image ** gamma, 0, 1)

        # Multiplicative noise
        if random.random() < self.augmentation["noise"]["probability"]:
            noise = np.random.normal(0, self.augmentation["noise"]["sigma"], (*image.shape[:2], 1))
            image = np.clip(image * (1.0 + noise), 0, 1).astype(np.float32)

        # Gaussian blur
        if random.random() < self.augmentation["gaussian_blur"]["probability"]:
            radius = random.uniform(self.augmentation["gaussian_blur"]["min_radius"], self.augmentation["gaussian_blur"]["max_radius"])

            image_pil = Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8))
            image_pil = image_pil.filter(ImageFilter.GaussianBlur(radius=radius))

            image = np.asarray(image_pil, dtype=np.float32) / 255.0
        
        return image.astype(np.float32), mask.astype(np.int64)

    
    def _get_context_ids(self, sample_id):
        study = self.study_by_sample[sample_id]
        previous_ids = []

        for offset in (1, 2):
            previous_sample_id = sample_id - offset

            if previous_sample_id in self.samples.index and self.study_by_sample[previous_sample_id] == study:
                previous_ids.append(previous_sample_id)

        return [sample_id] * (3 - len(previous_ids)) + previous_ids


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, index):
        sample = self.samples.iloc[index]
        context_sample_ids = self._get_context_ids(sample.sample_id)

        images = {
            context_sample_id: self._load_image(self.samples.loc[context_sample_id, "image_path"])
            for context_sample_id in set(context_sample_ids)
        }
        image = np.stack(
            [images[context_sample_id] for context_sample_id in context_sample_ids],
            axis=-1
        )

        original_mask = self._load_mask(sample["mask_path"])
        mask = self._resize(original_mask, is_mask=True).astype(np.int64)

        if self.augmentation and self.augmentation["enabled"]:
            image, mask = self._augment(image, mask)

        image_tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).contiguous().float()

        mask_tensor = torch.from_numpy(mask).long()
        original_mask_tensor = torch.from_numpy(original_mask).long()

        return image_tensor, mask_tensor, original_mask_tensor, self.task_name
