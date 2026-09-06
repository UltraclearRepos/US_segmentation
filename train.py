"""Train segmentation from a prepared dataset: python train.py --config config.json."""

import argparse
from datetime import datetime
import json
import shutil

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger

from callbacks import build_callbacks
from config import load_config
from data_module import SegmentationDataModule
from model_wrapper import SegmentationModelWrapper
from utils import seed_everything


def save_json(path, data):
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to the JSON configuration file.")
    return parser.parse_args()

def main():
    args = parse_args()

    config = load_config(args.config)
    seed_everything(config["training"]["seed"])

    dataset_dir = config["paths"]["dataset_dir"]
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = config["paths"]["output_root"] / dataset_dir.name / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    data_module = SegmentationDataModule(config, dataset_dir)
    data_module.setup()
    save_json(run_dir / "dataset_info.json", data_module.dataset_info)
    shutil.copy2(config["_path"], run_dir / "config.json")
    shutil.copy2(data_module.split_path, run_dir / "split.json")

    logger = TensorBoardLogger(
        save_dir=run_dir,
        name="tensorboard",
        version="",
    )
    model_wrapper = SegmentationModelWrapper(
        config=config,
        class_weights=data_module.class_weights,
        num_classes=data_module.num_classes,
        class_name_mapping=data_module.class_name_mapping,
    )
    trainer = pl.Trainer(
        max_epochs=config["training"]["epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=logger,
        callbacks=build_callbacks(config, run_dir / "checkpoints"),
        deterministic="warn",
        log_every_n_steps=config["tensorboard"]["log_every_n_steps"],
        gradient_clip_val=1.0
    )
    trainer.fit(model_wrapper, datamodule=data_module)


if __name__ == "__main__":
    main()
