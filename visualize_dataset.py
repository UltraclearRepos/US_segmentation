"""Save a random dataset sample after __getitem__, using config.json."""

import argparse
from pathlib import Path
import random

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from src.utils.config import load_config


def save_preview(images, mask, context_ids, output_path, num_classes):
    _, height, width = images.shape
    preview = Image.new("RGB", (4 * width, height + 28), "black")
    draw = ImageDraw.Draw(preview)

    panels = list(np.clip(images * 255, 0, 255).astype(np.uint8))
    panels.append((mask * 255 / max(num_classes - 1, 1)).astype(np.uint8))
    labels = [
        f"Current: {context_ids[0]}",
        f"Context 1: {context_ids[1]}",
        f"Context 2: {context_ids[2]}",
        f"Mask: {context_ids[0]}",
    ]

    for column, (panel, label) in enumerate(zip(panels, labels)):
        preview.paste(Image.fromarray(panel).convert("RGB"), (column * width, 28))
        draw.text((column * width + 8, 8), label, fill="white")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview.save(output_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()

    from src.data_pipeline.dataset import SegmentationDataset

    config = load_config(args.config)
    dataset_dir = args.dataset.resolve()
    manifest = pd.read_csv(dataset_dir / "manifest.csv")
    num_classes = len(pd.read_csv(dataset_dir / "classes.csv"))
    dataset = SegmentationDataset(
        dataset_dir=dataset_dir,
        sample_ids=manifest["sample_id"].tolist(),
        image_size=config["data"]["image_size"],
        num_classes=num_classes,
        task_name=dataset_dir.name,
        augmentation=config["data"]["augmentation"],
    )

    index = random.randrange(len(dataset))
    sample_id = dataset.samples.iloc[index].sample_id
    context_ids = dataset._get_context_ids(sample_id)
    images, mask, _, _ = dataset[index]

    output_path = Path("visualize") / dataset_dir.name / f"sample_{sample_id}.png"
    save_preview(images.numpy(), mask.numpy(), context_ids, output_path, num_classes)
    print(f"Context IDs: {context_ids}; image tensor: {tuple(images.shape)}")
    print(f"Saved: {output_path.resolve()}")


if __name__ == "__main__":
    main()
