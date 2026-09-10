"""Train segmentation from a prepared dataset: python train.py --config config.json."""

import argparse
from datetime import datetime
import json
import shutil

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger

from src.utils.callbacks import build_callbacks
from src.utils.config import load_config
from src.data_pipeline.data_module import SegmentationDataModule
from src.evaluation.diagnostics import generate_validation_diagnostics
from src.model.model_wrapper import SegmentationModelWrapper
from src.utils.utils import seed_everything


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

    dataset_dirs = config["datasets"]
    if not dataset_dirs:
        raise ValueError("Configure at least one dataset")

    run_name = str(config["run_name"])
    dataset_group = "_".join(sorted(dataset_dirs))
    run_dir = config["paths"]["output_root"] / dataset_group / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    data_module = SegmentationDataModule(config, dataset_dirs)
    data_module.setup()
    save_json(run_dir / "dataset_info.json", data_module.dataset_info)
    shutil.copy2(config["_path"], run_dir / "config.json")
    for task_name, split_path in data_module.split_paths.items():
        shutil.copy2(split_path, run_dir / f"{task_name}_split.json")

    logger = TensorBoardLogger(
        save_dir=run_dir,
        name="tensorboard",
        version="",
    )
    model_wrapper = SegmentationModelWrapper(
        config=config,
        class_weights=data_module.class_weights,
        num_classes=data_module.num_classes,
        class_name_mappings=data_module.class_name_mappings,
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

    best_checkpoint = torch.load(
        trainer.checkpoint_callback.best_model_path,
        map_location="cpu",
        weights_only=False,
    )
    model_wrapper.load_state_dict(best_checkpoint["state_dict"])
    del best_checkpoint
    generate_validation_diagnostics(model_wrapper, data_module.val_dataset, run_dir)


if __name__ == "__main__":
    main()
