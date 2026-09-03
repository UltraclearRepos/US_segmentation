"""Configuration loading."""

import json
from pathlib import Path

PROJECT_ROOT = Path.cwd()


def load_config(path):
    p = Path(path).expanduser().resolve()
    with p.open(encoding="utf-8") as f:
        config = json.load(f)

    missing = {"paths", "data", "model", "training", "loss", "export"} - set(config)
    if missing:
        raise ValueError(f"Missing config sections: {sorted(missing)}")

    config["_path"] = p
    config["paths"] = {
        name: (PROJECT_ROOT / value).resolve()
        for name, value in config["paths"].items()
    }
    return config
