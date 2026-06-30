#!/usr/bin/env python3
"""
Download W&B run summaries into local folders whose names match run.name.

Example:
    python download_wandb_summaries.py \
        --entity my-team \
        --project my-project \
        --root ./outputs

Assumption:
    ./outputs/
        run-name-1/
        run-name-2/
        ...

Each matching folder gets:
    ./outputs/<run.name>/wandb-summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import wandb


def json_safe(obj: Any) -> Any:
    """
    Convert W&B summary values into JSON-serializable values.
    Handles numpy scalars/arrays and other objects conservatively.
    """
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]

    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass

    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            pass

    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)


def download_summaries(
    entity: str,
    project: str,
    root: Path,
    output_filename: str = "wandb-summary.json",
    create_missing_folders: bool = False,
    overwrite: bool = True,
) -> None:
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")

    root = root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root folder does not exist: {root}")

    downloaded = 0
    skipped = 0

    for run in runs:
        run_name = run.name
        target_dir = root / run_name

        if not target_dir.exists():
            if create_missing_folders:
                target_dir.mkdir(parents=True, exist_ok=True)
            else:
                print(f"[skip] No matching folder for run '{run_name}': {target_dir}")
                skipped += 1
                continue

        output_path = target_dir / output_filename

        if output_path.exists() and not overwrite:
            print(f"[skip] Summary already exists: {output_path}")
            skipped += 1
            continue

        summary = dict(run.summary)
        summary["_wandb"] = {
            "run_id": run.id,
            "run_name": run.name,
            "run_path": run.path,
            "state": run.state,
            "url": run.url,
            "created_at": run.created_at,
        }

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(json_safe(summary), f, indent=2, sort_keys=True)

        print(f"[ok] Saved summary for '{run_name}' -> {output_path}")
        downloaded += 1

    print()
    print(f"Done. Downloaded: {downloaded}, skipped: {skipped}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", required=True, help="W&B entity/team/user name")
    parser.add_argument("--project", required=True, help="W&B project name")
    parser.add_argument(
        "--root",
        required=True,
        type=Path,
        help="Local root folder containing folders named after W&B run names",
    )
    parser.add_argument(
        "--filename",
        default="wandb-summary.json",
        help="Output filename to write inside each run folder",
    )
    parser.add_argument(
        "--create-missing-folders",
        action="store_true",
        help="Create folders for runs when no matching local folder exists",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Do not overwrite existing summary files",
    )

    args = parser.parse_args()

    download_summaries(
        entity=args.entity,
        project=args.project,
        root=args.root,
        output_filename=args.filename,
        create_missing_folders=args.create_missing_folders,
        overwrite=not args.no_overwrite,
    )


if __name__ == "__main__":
    main()
