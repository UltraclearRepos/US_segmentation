"""Train one segmentation variant: python train.py --config config.json."""
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


def save_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--variant", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(config["training"]["seed"])

    variant = args.variant or config["data"]["variant"]
    training_root = resolve_path(config, config["paths"]["training_root"])
    masks_dir = training_root / config["data"]["masks_dir"]
    original_images_dir = training_root / config["data"]["images_dir"]
    denoised_root = training_root / "denoised"
    images_dir = original_images_dir if variant == "original" else denoised_root / variant

    if not images_dir.exists():
        raise FileNotFoundError(f"Missing images for variant '{variant}': {images_dir}")

    output_dir = resolve_path(config, config["paths"]["models_root"]) / variant
    ensure_dir(output_dir)

    data_module = SegmentationDataModule(config, images_dir, masks_dir)
    data_module.setup()
    save_json(output_dir / "dataset_info.json", data_module.dataset_info)
    shutil.copy2(config["_path"], output_dir / "config.json")

    logger = TensorBoardLogger(
        save_dir=resolve_path(config, config["tensorboard"]["root_dir"]),
        name="segmentation",
        version=variant,
    )
    module = SegmentationModule(config, data_module.class_weights)
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
