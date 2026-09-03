"""LightningDataModule for segmentation."""

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from dataset import SegmentationDataset
from utils import make_generator, worker_init_fn


class SegmentationDataModule(pl.LightningDataModule):
    def __init__(self, config, images_dir, masks_dir):
        super().__init__()
        self.config = config
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.dataset_config = config["data"]
        self.training_config = config["training"]
        self.model_config = config["model"]
        self.generator = make_generator(self.training_config["seed"] + 30)

    def _load_mask(self, path):
        from PIL import Image
        with Image.open(path) as image:
            return np.asarray(image.convert("L"), dtype=np.uint8)

    def _resize_mask(self, mask):
        from PIL import Image
        height, width = self.dataset_config["image_size"]
        image = Image.fromarray(mask, "L")
        return np.asarray(image.resize((width, height), Image.Resampling.NEAREST))

    def _remap_mask(self, mask):
        mapped = np.zeros_like(mask, dtype=np.int64)
        for raw_value, class_id in self.label_mapping.items():
            mapped[mask == raw_value] = class_id
        return mapped

    def _build_pairs(self):
        extensions = SegmentationDataset.image_extensions
        if not self.images_dir.exists() or not self.masks_dir.exists():
            raise FileNotFoundError(f"Images: {self.images_dir}; masks: {self.masks_dir}")
        pairs = []
        for image_path in sorted(path for path in self.images_dir.iterdir() if path.suffix.lower() in extensions):
            masks = sorted(path for path in self.masks_dir.glob(f"{image_path.stem}.*") if path.suffix.lower() in extensions)
            if not masks:
                raise FileNotFoundError(f"No mask for {image_path.name}")
            pairs.append((image_path, masks[0]))
        if not pairs:
            raise RuntimeError(f"No images in {self.images_dir}")
        return pairs

    def _split_pairs(self, pairs):
        import random
        pairs = pairs.copy()
        random.Random(self.training_config["seed"]).shuffle(pairs)
        split = int(len(pairs) * self.dataset_config["train_ratio"])
        if split in (0, len(pairs)):
            raise ValueError("Need non-empty train and validation datasets")
        return pairs[:split], pairs[split:]

    def _build_label_mapping(self, pairs):
        values = sorted({int(value) for _, path in pairs for value in np.unique(self._load_mask(path))})
        if len(values) > self.model_config["num_classes"]:
            raise ValueError(f"Mask values {values}; num_classes={self.model_config['num_classes']}")
        return values, {value: index for index, value in enumerate(values)}

    def _class_counts(self, pairs):
        counts = np.zeros(self.model_config["num_classes"])
        for _, path in pairs:
            mask = self._remap_mask(self._resize_mask(self._load_mask(path)))
            counts += np.bincount(mask.ravel(), minlength=len(counts))[:len(counts)]
        return counts

    def _sample_weights(self, pairs, class_weights, power):
        result = []
        for _, path in pairs:
            present = np.unique(self._remap_mask(self._resize_mask(self._load_mask(path))))
            result.append(float(class_weights[present].mean()) ** power)
        return result

    def setup(self, stage=None):
        if hasattr(self, "train_dataset"):
            return
        pairs = self._build_pairs()
        self.mask_values, self.label_mapping = self._build_label_mapping(pairs)
        train_pairs, val_pairs = self._split_pairs(pairs)
        self.train_dataset = SegmentationDataset(
            train_pairs, self.label_mapping, self.dataset_config, augment=True
        )
        self.val_dataset = SegmentationDataset(
            val_pairs, self.label_mapping, self.dataset_config, augment=False
        )
        counts = self._class_counts(train_pairs)
        frequencies = np.maximum(counts, 1) / max(counts.sum(), 1)
        weights = 1 / np.power(frequencies, self.config["loss"]["class_weight_power"])
        self.class_weights = torch.tensor(weights / weights.mean(), dtype=torch.float32)
        self.dataset_info = {
            "num_pairs": len(pairs),
            "num_train": len(train_pairs),
            "num_val": len(val_pairs),
            "mask_values": self.mask_values,
            "label_mapping": self.label_mapping,
            "class_counts": counts.tolist(),
            "class_weights": self.class_weights.tolist(),
        }
        sampler_config = self.dataset_config["weighted_sampler"]
        self.train_sampler = None
        if sampler_config["enabled"]:
            sample_weights = self._sample_weights(
                train_pairs, self.class_weights, sampler_config["minority_sample_power"]
            )
            self.train_sampler = WeightedRandomSampler(
                sample_weights,
                len(sample_weights),
                replacement=True,
                generator=self.generator,
            )

    def _loader_kwargs(self):
        return {
            "batch_size": self.training_config["batch_size"],
            "num_workers": self.training_config["num_workers"],
            "pin_memory": self.training_config["pin_memory"],
            "persistent_workers": self.training_config["num_workers"] > 0,
            "worker_init_fn": worker_init_fn,
        }

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=self.train_sampler is None,
            sampler=self.train_sampler,
            generator=self.generator,
            **self._loader_kwargs()
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False, **self._loader_kwargs())
