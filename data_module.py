"""LightningDataModule splitting a prepared segmentation manifest."""

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import ConcatDataset, DataLoader

from dataset import SegmentationDataset
from utils import make_generator, worker_init_fn


def segmentation_collate(batch):
    """Stack model-sized tensors and keep native-sized masks as a list."""
    images, resized_masks, original_masks, task_names = zip(*batch)
    return (
        torch.stack(images),
        torch.stack(resized_masks),
        list(original_masks),
        task_names,
    )


class SegmentationDataModule(pl.LightningDataModule):
    def __init__(self, config, dataset_dirs):
        super().__init__()
        self.config = config
        self.dataset_dirs = {
            task_name: Path(dataset_dir)
            for task_name, dataset_dir in dataset_dirs.items()
        }
        self.dataset_config = config["data"]
        self.training_config = config["training"]
        self.generator = make_generator(self.training_config["seed"] + 30)

        self.class_name_mappings = {
            task_name: self._load_classes(dataset_dir / "classes.csv")
            for task_name, dataset_dir in self.dataset_dirs.items()
        }
        class_counts = {len(mapping) for mapping in self.class_name_mappings.values()}
        if class_counts != {2}:
            raise ValueError("Each task must have exactly two classes")
        self.num_classes = 2

    def _load_classes(self, classes_path):
        with classes_path.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))

        required = {"class_id", "class_name"}
        if not rows or not required.issubset(rows[0]):
            raise ValueError(f"{classes_path} must contain: {sorted(required)}")

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

    def _find_split_file(self, dataset_dir):
        json_files = sorted(dataset_dir.glob("*.json"))

        if not json_files:
            raise FileNotFoundError(f"No JSON split file found in {dataset_dir}")

        return json_files[0]

    def _split_manifest(self, manifest, dataset_dir):
        if len(manifest) < 2:
            raise ValueError("Need at least two samples for non-empty train and validation sets")

        split_path = self._find_split_file(dataset_dir)
        with split_path.open("r", encoding="utf-8") as file:
            split_data = json.load(file)

        train_ids = split_data["train"]
        val_ids = split_data["val"]

        manifest_ids = set(manifest["sample_id"].tolist())

        unknown_train_ids = set(train_ids) - manifest_ids
        unknown_val_ids = set(val_ids) - manifest_ids

        if unknown_train_ids or unknown_val_ids:
            raise ValueError(
                f"Split file {split_path} contains unknown sample IDs:\n"
                f"  Train: {sorted(unknown_train_ids)}\n"
                f"  Val: {sorted(unknown_val_ids)}"
            )

        if set(train_ids) & set(val_ids):
            raise ValueError(
                f"Split file {split_path} contains overlapping sample IDs:\n"
                f"  Overlap: {sorted(set(train_ids) & set(val_ids))}"
            )

        return train_ids, val_ids, split_path

    def _class_counts(self, dataset):
        counts = np.zeros(self.num_classes, dtype=np.int64)
        for mask in dataset.iter_masks():
            counts += np.bincount(mask.ravel(), minlength=self.num_classes)
        return counts

    def setup(self, stage=None):
        if hasattr(self, "train_dataset"):
            return

        train_datasets = []
        val_datasets = []
        self.class_weights = {}
        self.split_paths = {}
        self.dataset_info = {}

        for task_name, dataset_dir in self.dataset_dirs.items():
            manifest = pd.read_csv(dataset_dir / "manifest.csv")
            train_ids, val_ids, split_path = self._split_manifest(manifest, dataset_dir)

            train_dataset = SegmentationDataset(
                dataset_dir,
                train_ids,
                self.dataset_config["image_size"],
                self.num_classes,
                task_name,
                augmentation=self.dataset_config["augmentation"],
            )
            val_dataset = SegmentationDataset(
                dataset_dir,
                val_ids,
                self.dataset_config["image_size"],
                self.num_classes,
                task_name,
            )
            train_datasets.append(train_dataset)
            val_datasets.append(val_dataset)
            self.split_paths[task_name] = split_path

            counts = self._class_counts(train_dataset)
            frequencies = np.maximum(counts, 1) / max(counts.sum(), 1)
            weights = 1 / np.power(
                frequencies,
                self.config["loss"]["class_weight_power"],
            )
            weights = torch.tensor(weights / weights.mean(), dtype=torch.float32)
            self.class_weights[task_name] = weights

            self.dataset_info[task_name] = {
                "dataset_dir": str(dataset_dir),
                "split": str(split_path),
                "num_samples": len(manifest),
                "num_train": len(train_ids),
                "num_val": len(val_ids),
                "classes": self.class_name_mappings[task_name],
                "class_counts": counts.tolist(),
                "class_weights": weights.tolist(),
            }

        self.train_dataset = ConcatDataset(train_datasets)
        self.val_dataset = ConcatDataset(val_datasets)

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
