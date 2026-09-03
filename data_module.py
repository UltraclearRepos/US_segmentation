"""LightningDataModule splitting a prepared segmentation manifest."""

import csv
from pathlib import Path
import random

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset import SegmentationDataset
from utils import make_generator, worker_init_fn


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
        self.classes = self._load_classes()
        self.num_classes = len(self.classes)

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

        return [
            {"class_id": int(row["class_id"]), "class_name": row["class_name"]}
            for row in classes
        ]

    def _split_manifest(self, manifest):
        group_ids = sorted(manifest["group_id"].unique())
        random.Random(self.training_config["seed"]).shuffle(group_ids)
        split_index = int(len(group_ids) * self.dataset_config["train_ratio"])
        if split_index in (0, len(group_ids)):
            raise ValueError("Need at least two groups for non-empty train and validation sets")

        train_groups = set(group_ids[:split_index])
        is_train = manifest["group_id"].isin(train_groups)
        train_ids = manifest.loc[is_train, "sample_id"].tolist()
        val_ids = manifest.loc[~is_train, "sample_id"].tolist()
        val_groups = sorted(set(group_ids) - train_groups)
        return train_ids, val_ids, sorted(train_groups), val_groups

    def _class_counts(self, dataset):
        counts = np.zeros(self.num_classes, dtype=np.int64)
        for mask in dataset.iter_masks():
            counts += np.bincount(mask.ravel(), minlength=self.num_classes)
        return counts

    def _sample_weights(self, dataset, class_weights, power):
        return [
            float(class_weights[np.unique(mask).tolist()].mean()) ** power
            for mask in dataset.iter_masks()
        ]

    def setup(self, stage=None):
        if hasattr(self, "train_dataset"):
            return

        manifest = pd.read_csv(self.manifest_path)
        train_ids, val_ids, train_groups, val_groups = self._split_manifest(manifest)

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
            "train_groups": train_groups,
            "val_groups": val_groups,
            "classes": self.classes,
            "class_counts": counts.tolist(),
            "class_weights": self.class_weights.tolist(),
        }

        sampler_config = self.dataset_config["weighted_sampler"]
        self.train_sampler = None
        if sampler_config["enabled"]:
            sample_weights = self._sample_weights(
                self.train_dataset,
                self.class_weights,
                sampler_config["minority_sample_power"],
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
            **self._loader_kwargs(),
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False, **self._loader_kwargs())
