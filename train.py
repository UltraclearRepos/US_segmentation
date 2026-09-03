"""Train segmentation from a prepared dataset: python train.py --config config.json."""

import argparse
import json
import shutil
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger

from callbacks import build_callbacks
from config import load_config, resolve_path
from data_module import SegmentationDataModule
from segmentation_module import SegmentationModule
from utils import ensure_dir, seed_everything


def save_json(path, data):
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset-dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(config["training"]["seed"])

    dataset_dir = resolve_path(
        config, args.dataset_dir or config["paths"]["dataset_dir"]
    )
    output_dir = resolve_path(config, config["paths"]["models_root"]) / dataset_dir.name
    ensure_dir(output_dir)

    data_module = SegmentationDataModule(config, dataset_dir)
    data_module.setup()
    save_json(output_dir / "dataset_info.json", data_module.dataset_info)
    shutil.copy2(config["_path"], output_dir / "config.json")

    logger = TensorBoardLogger(
        save_dir=resolve_path(config, config["tensorboard"]["root_dir"]),
        name="segmentation",
        version=dataset_dir.name,
    )
    module = SegmentationModule(
        config,
        data_module.class_weights,
        num_classes=data_module.num_classes,
    )
    trainer = pl.Trainer(
        max_epochs=config["training"]["epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=logger,
        callbacks=build_callbacks(config, output_dir / "checkpoints"),
        deterministic=True,
        log_every_n_steps=config["tensorboard"]["log_every_n_steps"],
    )
    trainer.fit(module, datamodule=data_module)


if __name__ == "__main__":
    main()
