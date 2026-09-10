"""Generate validation diagnostics for a finished run."""

import argparse
from pathlib import Path

import torch

from src.utils.config import load_config
from src.data_pipeline.data_module import SegmentationDataModule
from src.evaluation.diagnostics import generate_validation_diagnostics
from src.model.model_wrapper import SegmentationModelWrapper


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="Path to a finished run.")
    parser.add_argument("--checkpoint-name", help="Optional checkpoint path.")
    return parser.parse_args()



def find_checkpoint(run_dir, checkpoint_name):
    if checkpoint_name:
        checkpoint_path = run_dir / "checkpoints" / checkpoint_name
        return Path(checkpoint_path).expanduser().resolve()

    checkpoints = list((run_dir / "checkpoints").glob("best-*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(f"No best checkpoint found in {run_dir / 'checkpoints'}")
    return max(checkpoints, key=lambda path: path.stat().st_mtime)


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    config = load_config(run_dir / "config.json")

    data_module = SegmentationDataModule(config, config["datasets"])
    data_module.setup()
    model = SegmentationModelWrapper(
        config=config,
        class_weights=data_module.class_weights,
        num_classes=data_module.num_classes,
        class_name_mappings=data_module.class_name_mappings,
    )

    checkpoint_path = find_checkpoint(run_dir, args.checkpoint_name)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    del checkpoint
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    generate_validation_diagnostics(model, data_module.val_dataset, run_dir)


if __name__ == "__main__":
    main()
