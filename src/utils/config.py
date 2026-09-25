"""Configuration loading."""

import json
from pathlib import Path

PROJECT_ROOT = Path.cwd()


def load_config(path):
    p = Path(path).expanduser().resolve()
    with p.open(encoding="utf-8") as f:
        config = json.load(f)

    missing = {"datasets", "paths", "data", "model", "training", "loss"} - set(config)
    if missing:
        raise ValueError(f"Missing config sections: {sorted(missing)}")

    config["_path"] = p
    config["paths"] = {
        name: (PROJECT_ROOT / value).resolve()
        for name, value in config["paths"].items()
    }

    dataset_sections = ("split", "train", "val")
    datasets = config["datasets"]

    if set(datasets) != set(dataset_sections):
        raise ValueError(
            f"Config 'datasets' section must contain exactly these keys: {dataset_sections}"
        )

    resolved_datasets = {}
    seen_paths = {}

    for section in dataset_sections:
        resolved_datasets[section] = {}

        for task_name, dataset_paths in datasets[section].items():
            if not isinstance(dataset_paths, list):
                raise ValueError(f"Config 'datasets' section '{section}' must contain a list of paths for each task")

            resolved_paths = []

            for dataset_path in dataset_paths:
                resolved_dataset_path = (PROJECT_ROOT / dataset_path).resolve()

                if resolved_dataset_path in seen_paths:
                    previous_section, previous_task = seen_paths[resolved_dataset_path]
                    raise ValueError(
                        f"Dataset path '{resolved_dataset_path}' is used in both '{previous_section}/{previous_task}' and '{section}/{task_name}'"
                    )

                seen_paths[resolved_dataset_path] = (section, task_name)
                resolved_paths.append(resolved_dataset_path)

            resolved_datasets[section][task_name] = resolved_paths

    config["datasets"] = resolved_datasets
    
    return config
