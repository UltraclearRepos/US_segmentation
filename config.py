"""Configuration loading."""

import json
from pathlib import Path

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.json")


def load_config(path=None):
    p = Path(path or DEFAULT_CONFIG_PATH).expanduser().resolve()
    with p.open(encoding="utf-8") as f:
        c = json.load(f)
    missing = {"paths", "data", "model", "training", "loss", "export"} - set(c)
    if missing:
        raise ValueError(f"Missing config sections: {sorted(missing)}")
    root = Path(c["paths"]["project_root"])
    c["_path"] = p
    c["_project_root"] = (p.parent / root).resolve() if not root.is_absolute() else root
    return c


def resolve_path(c, value):
    p = Path(value)
    return p if p.is_absolute() else c["_project_root"] / p
