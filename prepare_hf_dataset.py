#!/usr/bin/env python3
"""Prepare the private OperAI-EYE test dataset from the Hugging Face cache."""

from __future__ import annotations

import argparse
import json
import sys

from hf_dataset import (
    HF_DATASET_REPO_ID,
    HF_DATASET_REVISION,
    DatasetPreparationError,
    prepare_hf_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=HF_DATASET_REPO_ID)
    parser.add_argument("--revision", default=HF_DATASET_REVISION)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print machine-readable status JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        status = prepare_hf_dataset(
            repo_id=args.repo_id,
            revision=args.revision,
            max_workers=args.max_workers,
            local_files_only=args.local_files_only,
        )
    except DatasetPreparationError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(status, indent=2))
    else:
        print(f"Dataset ready: {status['image_count']} images")
        print(f"Snapshot: {status['snapshot_path']}")
        print(f"Images: {status['split_path']}")
        print(f"Labels: {status['labels_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
