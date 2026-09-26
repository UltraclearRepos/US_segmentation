"""Fetch datasets and a pretrained checkpoint referenced by a training config.

The Prepare_dataset Freestyle job is expected to:

* expose a string parameter named ``BUILD_NAME`` (for example ``tg3k``), and
* archive ``${BUILD_NAME}.zip``.

Each archive must contain one top-level directory whose name matches the final
component of the configured dataset path.  For example, ``tg3k.zip`` contains
``tg3k/...`` and is extracted into ``data/prepared`` to produce
``data/prepared/tg3k/...``.

The same Jenkins job also stores model artifacts.  For an ``input_checkpoint`` equal
to ``input_checkpoints/best_model.pth``, its build parameter must be
``BUILD_NAME=best_model`` and the archived artifact must be ``best_model.pth``.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
import zipfile


DATASET_SECTIONS = ("split", "train", "val")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--prepare-job-url",
        required=True,
        help=(
            "URL of the Prepare_dataset job, for example "
            "https://jenkins.example/job/UCLR-165/job/Prepare_dataset/"
        ),
    )
    parser.add_argument(
        "--build-name-parameter",
        default="BUILD_NAME",
        help="Prepare_dataset parameter used to identify a dataset (default: BUILD_NAME)",
    )
    parser.add_argument(
        "--max-builds",
        type=int,
        default=100,
        help="Maximum number of recent Prepare_dataset builds to inspect (default: 100)",
    )
    return parser.parse_args()


def load_dataset_paths(config_path: Path) -> list[Path]:
    with config_path.resolve().open(encoding="utf-8") as file:
        config = json.load(file)

    datasets = config.get("datasets")
    if not isinstance(datasets, dict):
        raise ValueError("Config must contain an object named 'datasets'")

    paths: list[Path] = []
    seen: set[Path] = set()
    for section in DATASET_SECTIONS:
        tasks = datasets.get(section)
        if not isinstance(tasks, dict):
            raise ValueError(f"Config datasets.{section} must be an object")

        for task_name, task_paths in tasks.items():
            if not isinstance(task_paths, list):
                raise ValueError(
                    f"Config datasets.{section}.{task_name} must be a list of paths"
                )
            for value in task_paths:
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(
                        f"Config datasets.{section}.{task_name} contains an invalid path"
                    )
                path = Path(value)
                if path not in seen:
                    seen.add(path)
                    paths.append(path)

    if not paths:
        raise ValueError("Config does not reference any datasets")
    return paths


def load_input_checkpoint_path(config_path: Path) -> Path:
    with config_path.resolve().open(encoding="utf-8") as file:
        config = json.load(file)

    try:
        value = config["paths"]["input_checkpoint"]
    except (KeyError, TypeError) as error:
        raise ValueError("Config must contain paths.input_checkpoint") from error
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Config paths.input_checkpoint must be a non-empty path")
    return Path(value)


def _authorization_header() -> str | None:
    username = os.environ.get("JENKINS_API_USER")
    token = os.environ.get("JENKINS_API_TOKEN")
    if bool(username) != bool(token):
        raise ValueError(
            "Set both JENKINS_API_USER and JENKINS_API_TOKEN, or neither for anonymous access"
        )
    if not username:
        return None
    encoded = base64.b64encode(f"{username}:{token}".encode()).decode("ascii")
    return f"Basic {encoded}"


def _open_url(url: str, authorization: str | None):
    headers = {"Accept": "application/json", "User-Agent": "US-segmentation-Jenkins"}
    if authorization:
        headers["Authorization"] = authorization
    try:
        return urlopen(Request(url, headers=headers), timeout=60)
    except HTTPError as error:
        raise RuntimeError(f"Jenkins returned HTTP {error.code} for {url}") from error
    except URLError as error:
        raise RuntimeError(f"Could not connect to Jenkins at {url}: {error.reason}") from error


def load_builds(job_url: str, max_builds: int, authorization: str | None) -> list[dict]:
    if max_builds < 1:
        raise ValueError("--max-builds must be greater than zero")

    tree = (
        "builds[number,result,url,actions[parameters[name,value]],"
        f"artifacts[fileName,relativePath]]{{0,{max_builds}}}"
    )
    api_url = f"{job_url.rstrip('/')}/api/json?{urlencode({'tree': tree})}"
    with _open_url(api_url, authorization) as response:
        payload = json.load(response)
    return payload.get("builds", [])


def _build_parameter(build: dict, parameter_name: str):
    for action in build.get("actions", []):
        for parameter in action.get("parameters", []):
            if parameter.get("name") == parameter_name:
                return parameter.get("value")
    return None


def find_artifact(
    builds: list[dict],
    build_name: str,
    artifact_name: str,
    parameter_name: str,
    job_label: str,
) -> tuple[int, str]:
    for build in sorted(builds, key=lambda value: value.get("number", -1), reverse=True):
        if build.get("result") != "SUCCESS":
            continue
        if str(_build_parameter(build, parameter_name)) != build_name:
            continue

        build_number = int(build["number"])
        artifact = next(
            (
                value
                for value in build.get("artifacts", [])
                if value.get("fileName") == artifact_name
            ),
            None,
        )
        if artifact is None:
            raise FileNotFoundError(
                f"{job_label} build #{build_number} matches '{build_name}', "
                f"but does not contain {artifact_name}"
            )

        relative_path = quote(artifact["relativePath"], safe="/")
        return build_number, f"{build['url'].rstrip('/')}/artifact/{relative_path}"

    raise FileNotFoundError(
        f"No successful {job_label} build with {parameter_name}={build_name!r} "
        f"was found among the inspected builds"
    )


def resolve_workspace_path(path: Path, project_root: Path, label: str) -> Path:
    target = (project_root / path).resolve()
    try:
        target.relative_to(project_root)
    except ValueError as error:
        raise ValueError(f"{label} path escapes the workspace: {path}") from error
    return target


def download_file(url: str, target: Path, authorization: str | None):
    with _open_url(url, authorization) as response, target.open("wb") as output:
        shutil.copyfileobj(response, output)


def extract_dataset(zip_path: Path, dataset_path: Path, project_root: Path):
    target = resolve_workspace_path(dataset_path, project_root, "Dataset")

    if target.exists():
        raise FileExistsError(
            f"Dataset target already exists: {target}. Start from a clean workspace."
        )

    dataset_name = target.name
    extraction_root = target.parent
    extraction_root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as archive:
        entries: list[tuple[zipfile.ZipInfo, PurePosixPath]] = []
        top_level_names: set[str] = set()

        for info in archive.infolist():
            normalized_name = info.filename.replace("\\", "/")
            path = PurePosixPath(normalized_name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Unsafe path in dataset archive: {info.filename}")
            if not path.parts:
                continue

            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError(f"Symbolic links are not allowed in dataset archive: {info.filename}")

            top_level_names.add(path.parts[0])
            entries.append((info, path))

        if top_level_names != {dataset_name}:
            raise ValueError(
                f"Archive for '{dataset_name}' must contain exactly one top-level "
                f"directory named '{dataset_name}', found: {sorted(top_level_names)}"
            )

        for info, path in entries:
            destination = extraction_root.joinpath(*path.parts)
            if info.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, destination.open("wb") as output:
                shutil.copyfileobj(source, output)


def main():
    args = parse_args()
    project_root = Path.cwd().resolve()
    dataset_paths = load_dataset_paths(args.config)
    checkpoint_path = load_input_checkpoint_path(args.config)
    authorization = _authorization_header()
    dataset_builds = load_builds(args.prepare_job_url, args.max_builds, authorization)

    with tempfile.TemporaryDirectory(prefix="jenkins-datasets-") as temp_dir:
        temp_root = Path(temp_dir)
        for dataset_path in dataset_paths:
            dataset_name = dataset_path.name
            build_number, artifact_url = find_artifact(
                dataset_builds,
                dataset_name,
                f"{dataset_name}.zip",
                args.build_name_parameter,
                "Prepare_dataset",
            )
            archive_path = temp_root / f"{dataset_name}.zip"
            print(
                f"Fetching dataset '{dataset_name}' from Prepare_dataset "
                f"build #{build_number}"
            )
            download_file(artifact_url, archive_path, authorization)
            extract_dataset(archive_path, dataset_path, project_root)
            print(f"Extracted '{dataset_name}' to {(project_root / dataset_path).resolve()}")

    checkpoint_name = checkpoint_path.stem
    checkpoint_target = resolve_workspace_path(
        checkpoint_path, project_root, "Input checkpoint"
    )
    if checkpoint_target.exists():
        raise FileExistsError(
            f"Input checkpoint already exists: {checkpoint_target}. "
            "Start from a clean workspace."
        )

    model_build_number, model_artifact_url = find_artifact(
        dataset_builds,
        checkpoint_name,
        checkpoint_path.name,
        args.build_name_parameter,
        "Prepare_dataset",
    )
    checkpoint_target.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"Fetching checkpoint '{checkpoint_path.name}' from Prepare_dataset "
        f"build #{model_build_number}"
    )
    download_file(model_artifact_url, checkpoint_target, authorization)
    print(f"Saved checkpoint '{checkpoint_name}' to {checkpoint_target}")


if __name__ == "__main__":
    main()
