"""Prepare images, remap masks and create a manifest for segmentation."""

import argparse
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from PIL import Image


EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
CLASS_COLUMNS = ["class_id", "class_name", "source_value"]


def load_classes(path):
    classes = pd.read_csv(path)
    if classes.empty or not set(CLASS_COLUMNS).issubset(classes.columns):
        raise ValueError(f"{path} must contain: {CLASS_COLUMNS}")

    classes = classes[CLASS_COLUMNS].copy()
    classes[["class_id", "source_value"]] = classes[
        ["class_id", "source_value"]
    ].astype(int)
    classes["class_name"] = classes["class_name"].astype(str).str.strip()
    classes = classes.sort_values("class_id").reset_index(drop=True)

    if classes["class_id"].tolist() != list(range(len(classes))):
        raise ValueError("class_id values must be consecutive and start at 0")
    if classes["class_name"].eq("").any() or classes.duplicated().any():
        raise ValueError("Class rows must be non-empty and unique")
    if classes["class_name"].duplicated().any() or classes["source_value"].duplicated().any():
        raise ValueError("class_name and source_value values must be unique")
    if not classes["source_value"].between(0, 255).all() or len(classes) > 256:
        raise ValueError("Grayscale masks support values from 0 to 255")
    return classes


def supported_files(directory):
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in EXTENSIONS
    )


def build_pairs(source_dir):
    images_dir, masks_dir = source_dir / "images", source_dir / "masks"
    images, all_masks = supported_files(images_dir), set(supported_files(masks_dir))
    if not images:
        raise ValueError(f"No supported files in {images_dir}")

    pairs, paired_masks, sample_ids = [], set(), set()
    for image_path in images:
        if image_path.stem in sample_ids:
            raise ValueError(f"Duplicate image name without extension: {image_path.stem}")
        sample_ids.add(image_path.stem)
        matching_masks = {
            path
            for path in masks_dir.glob(f"{image_path.stem}.*")
            if path.is_file() and path.suffix.lower() in EXTENSIONS
        }
        if len(matching_masks) != 1:
            raise ValueError(
                f"Expected one mask for {image_path.name}, found {len(matching_masks)}"
            )
        mask_path = matching_masks.pop()
        pairs.append((image_path.stem, image_path, mask_path))
        paired_masks.add(mask_path)

    orphan_masks = sorted(path.name for path in all_masks - paired_masks)
    if orphan_masks:
        raise ValueError(f"Masks without images: {orphan_masks}")
    return pd.DataFrame(pairs, columns=["sample_id", "image_source", "mask_source"])


def add_groups(pairs, path):
    if not path.exists():
        return pairs.assign(group_id=pairs["sample_id"])

    groups = pd.read_csv(path, dtype=str)
    if not {"sample_id", "group_id"}.issubset(groups.columns):
        raise ValueError("groups.csv must contain sample_id and group_id")
    groups = groups[["sample_id", "group_id"]]
    if groups.isna().any().any() or groups["sample_id"].duplicated().any():
        raise ValueError("groups.csv contains empty or duplicate entries")
    groups["group_id"] = groups["group_id"].str.strip()
    if groups["group_id"].eq("").any() or set(groups["sample_id"]) != set(pairs["sample_id"]):
        raise ValueError("groups.csv must contain exactly one group for every sample")
    
    return pairs.merge(groups, on="sample_id", validate="one_to_one")

def remap_mask(path, mapping):
    with Image.open(path) as image:
        mask = np.asarray(image.convert("L"), dtype=np.uint8)

    unknown = sorted(int(value) for value in set(np.unique(mask)) - set(mapping))
    if unknown:
        raise ValueError(f"Unknown source values {unknown} in mask: {path}")

    result = np.empty_like(mask)
    for source_value, class_id in mapping.items():
        result[mask == source_value] = class_id
    return result


def prepare_dataset(source_dir, output_dir):
    source_dir, output_dir = source_dir.resolve(), output_dir.resolve()
    required = [source_dir / "images", source_dir / "masks", source_dir / "classes.csv"]
    missing = [path for path in required if not path.exists()]
    if source_dir == output_dir:
        raise ValueError("Source and output directories must be different")
    if missing:
        raise FileNotFoundError(f"Missing source dataset entries: {missing}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")

    classes = load_classes(source_dir / "classes.csv")
    pairs = add_groups(build_pairs(source_dir), source_dir / "groups.csv")

    mapping = dict(zip(classes["source_value"], classes["class_id"]))
    masks = {
        row.sample_id: remap_mask(row.mask_source, mapping)
        for row in pairs.itertuples()
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir()
    (output_dir / "masks").mkdir()

    image_paths, mask_paths = [], []
    for row in pairs.itertuples():
        image_target = output_dir / "images" / row.image_source.name
        mask_target = output_dir / "masks" / f"{row.sample_id}.png"
        shutil.copy2(row.image_source, image_target)
        Image.fromarray(masks[row.sample_id], mode="L").save(mask_target)
        image_paths.append(image_target.relative_to(output_dir).as_posix())
        mask_paths.append(mask_target.relative_to(output_dir).as_posix())

    manifest = pairs[["sample_id", "group_id"]].copy()
    manifest["image_path"], manifest["mask_path"] = image_paths, mask_paths
    manifest.to_csv(output_dir / "manifest.csv", index=False)
    classes.to_csv(output_dir / "classes.csv", index=False)
    print(f"Prepared {len(manifest)} samples in: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    prepare_dataset(args.source_dir, args.output_dir)


if __name__ == "__main__":
    main()
