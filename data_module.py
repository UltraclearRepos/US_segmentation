"""LightningDataModule splitting a prepared segmentation manifest."""

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from dataset import SegmentationDataset
from utils import make_generator, worker_init_fn


def segmentation_collate(batch):
    """Stack model-sized tensors and keep native-sized masks as a list."""
    images, resized_masks, original_masks = zip(*batch)
    return (
        torch.stack(images),
        torch.stack(resized_masks),
        list(original_masks),
    )


class SegmentationDataModule(pl.LightningDataModule):
    def __init__(self, config, dataset_dir):
        super().__init__()
        self.config = config
        self.dataset_dir = Path(dataset_dir)
        self.dataset_config = config["data"]
        self.training_config = config["training"]
        self.generator = make_generator(self.training_config["seed"] + 30)

        self.manifest_path = self.dataset_dir / "manifest.csv"
        self.classes_path = self.dataset_dir / "classes.csv"
        self.class_name_mapping = self._load_classes()
        self.num_classes = len(self.class_name_mapping)

    def _load_classes(self):
        with self.classes_path.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))

        required = {"class_id", "class_name"}
        if not rows or not required.issubset(rows[0]):
            raise ValueError(f"{self.classes_path} must contain: {sorted(required)}")

        classes = sorted(rows, key=lambda row: int(row["class_id"]))
        class_ids = [int(row["class_id"]) for row in classes]
        if class_ids != list(range(len(classes))):
            raise ValueError("class_id values must be consecutive and start at 0")
        if len(set(row["class_name"] for row in classes)) != len(classes):
            raise ValueError("class_name values must be unique")

        return {
            int(row["class_id"]): row["class_name"]
            for row in classes
        }

    def _find_split_file(self):
        json_files = list(self.dataset_dir.glob("*.json"))

        if not json_files:
            raise FileNotFoundError(f"No JSON split file found in {self.dataset_dir}")

        if len(json_files) > 1:
            raise ValueError(f"Multiple JSON split files found in {self.dataset_dir}: {json_files}")

        return json_files[0]

    def _split_manifest(self, manifest):
        if len(manifest) < 2:
            raise ValueError("Need at least two samples for non-empty train and validation sets")

        self.split_path = self._find_split_file()
        with self.split_path.open("r", encoding="utf-8") as file:
            split_data = json.load(file)

        train_ids = split_data["train"]
        val_ids = split_data["val"]

        manifest_ids = set(manifest["sample_id"].tolist())

        unknown_train_ids = set(train_ids) - manifest_ids
        unknown_val_ids = set(val_ids) - manifest_ids

        if unknown_train_ids or unknown_val_ids:
            raise ValueError(
                f"Split file {self.split_path} contains unknown sample IDs:\n"
                f"  Train: {sorted(unknown_train_ids)}\n"
                f"  Val: {sorted(unknown_val_ids)}"
            )

        if set(train_ids) & set(val_ids):
            raise ValueError(
                f"Split file {self.split_path} contains overlapping sample IDs:\n"
                f"  Overlap: {sorted(set(train_ids) & set(val_ids))}"
            )

        return train_ids, val_ids

    def _class_counts(self, dataset):
        counts = np.zeros(self.num_classes, dtype=np.int64)
        for mask in dataset.iter_masks():
            counts += np.bincount(mask.ravel(), minlength=self.num_classes)
        return counts

    def setup(self, stage=None):
        if hasattr(self, "train_dataset"):
            return

        manifest = pd.read_csv(self.manifest_path)
        train_ids, val_ids = self._split_manifest(manifest)

        self.train_dataset = SegmentationDataset(
            self.dataset_dir,
            train_ids,
            self.dataset_config["image_size"],
            self.num_classes,
            augmentation=self.dataset_config["augmentation"],
        )
        self.val_dataset = SegmentationDataset(
            self.dataset_dir,
            val_ids,
            self.dataset_config["image_size"],
            self.num_classes,
        )

        counts = self._class_counts(self.train_dataset)
        frequencies = np.maximum(counts, 1) / max(counts.sum(), 1)
        weights = 1 / np.power(frequencies, self.config["loss"]["class_weight_power"])
        self.class_weights = torch.tensor(weights / weights.mean(), dtype=torch.float32)

        self.dataset_info = {
            "dataset_dir": str(self.dataset_dir),
            "num_samples": len(manifest),
            "num_train": len(train_ids),
            "num_val": len(val_ids),
            "classes": self.class_name_mapping,
            "class_counts": counts.tolist(),
            "class_weights": self.class_weights.tolist(),
        }

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
            shuffle=True,
            generator=self.generator,
            collate_fn=segmentation_collate,
            **self._loader_kwargs(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            collate_fn=segmentation_collate,
            **self._loader_kwargs(),
        )
