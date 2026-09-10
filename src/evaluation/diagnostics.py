"""Save validation predictions next to the original ultrasound images."""

from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


COLORS = {
    "tp": np.array([0, 220, 0], dtype=np.float32),
    "fn": np.array([255, 0, 0], dtype=np.float32),
    "fp": np.array([0, 80, 255], dtype=np.float32),
}


def _add_overlay(image, prediction, target, alpha=0.5):
    result = np.repeat(image[..., None], 3, axis=2).astype(np.float32)
    regions = {
        "tp": (prediction == 1) & (target == 1),
        "fn": (prediction == 0) & (target == 1),
        "fp": (prediction == 1) & (target == 0),
    }
    for name, region in regions.items():
        result[region] = (1 - alpha) * result[region] + alpha * COLORS[name]
    return result.astype(np.uint8)


def _save_comparison(image_path, prediction, target, output_path):
    with Image.open(image_path) as source:
        image = np.asarray(source.convert("L"), dtype=np.uint8)

    overlay = _add_overlay(image, prediction, target)
    height, width = image.shape
    header_height = 28
    comparison = Image.new("RGB", (2 * width, height + header_height), "black")
    comparison.paste(Image.fromarray(image, mode="L").convert("RGB"), (0, header_height))
    comparison.paste(Image.fromarray(overlay, mode="RGB"), (width, header_height))

    draw = ImageDraw.Draw(comparison)
    draw.text((8, 8), "USG", fill="white")
    draw.text(
        (width + 8, 8),
        "Green: correct   Red: missed   Blue: extra",
        fill="white",
    )
    comparison.save(output_path)


@torch.inference_mode()
def generate_validation_diagnostics(model, val_dataset, run_dir):
    """Run inference and save one side-by-side PNG for every validation sample."""
    output_dir = Path(run_dir) / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    model.eval()
    inference_times = []

    for dataset in val_dataset.datasets:
        for position, (sample_id, sample) in enumerate(dataset.samples.iterrows()):
            image, _, original_mask, task_name = dataset[position]
            image = image.unsqueeze(0).to(device)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = perf_counter()
            outputs = model(image)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_times.append(perf_counter() - start)

            logits = outputs[task_name]
            prediction = logits.argmax(dim=1, keepdim=True).float()
            prediction = F.interpolate(
                prediction,
                size=original_mask.shape,
                mode="nearest",
            )[0, 0].cpu().numpy().astype(np.uint8)

            output_path = output_dir / task_name / f"{str(sample_id)}.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            _save_comparison(
                sample["image_path"],
                prediction,
                original_mask.numpy(),
                output_path,
            )

    print(f"Saved validation diagnostics to {output_dir}")
    if inference_times:
        times_ms = np.asarray(inference_times) * 1000
        print(
            f"Inference time: {times_ms.mean():.2f} +/- {times_ms.std():.2f} ms "
            f"(mean +/- std, n={len(times_ms)})"
        )
