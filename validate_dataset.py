"""Generate diagnostics for a finished run on an arbitrary prepared dataset."""

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import ConcatDataset

from src.utils.config import load_config
from src.data_pipeline.dataset import SegmentationDataset
from src.evaluation.diagnostics import generate_validation_diagnostics
from src.model.model_wrapper import SegmentationModelWrapper
from validate import find_checkpoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="Path to a finished run.")
    parser.add_argument("--dataset-dir", required=True, help="Path to a prepared dataset.")
    parser.add_argument("--checkpoint-name", help="Optional checkpoint file name.")
    parser.add_argument("--task", required=True, help="Task head to use (e.g. thyroid, nodule).")
    return parser.parse_args()


def load_patients(dataset_dir):
    """Map str(sample_id) -> patient from patients.csv, or None if the file is missing."""
    patients_path = dataset_dir / "patients.csv"
    if not patients_path.exists():
        return None
    patients = pd.read_csv(patients_path, dtype=str)
    return dict(zip(patients["sample_id"], patients["patient"]))


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    config = load_config(run_dir / "config.json")

    checkpoint_path = find_checkpoint(run_dir, args.checkpoint_name)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hparams = checkpoint["hyper_parameters"]

    model = SegmentationModelWrapper(
        config=config,
        class_weights=hparams["class_weights"],
        num_classes=hparams["num_classes"],
        class_name_mappings=hparams["class_name_mappings"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    del checkpoint
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    task_name = args.task
    if task_name not in model.class_name_mappings:
        raise ValueError(f"Task '{task_name}' not in model tasks: {sorted(model.class_name_mappings)}")

    sample_ids = pd.read_csv(dataset_dir / "manifest.csv")["sample_id"].tolist()
    dataset = SegmentationDataset(
        dataset_dir,
        sample_ids,
        config["data"]["image_size"],
        model.hparams["num_classes"],
        task_name,
    )
    print(f"Running task '{task_name}' on {len(dataset)} samples from {dataset_dir}")

    sample_subdirs = load_patients(dataset_dir)
    generate_validation_diagnostics(
        model,
        ConcatDataset([dataset]),
        run_dir,
        Path(dataset_dir.name) / "val",
        sample_subdirs=sample_subdirs,
    )


if __name__ == "__main__":
    main()
