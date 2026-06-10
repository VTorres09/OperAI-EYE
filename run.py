#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path

from hf_dataset import DatasetPreparationError, prepare_hf_dataset

ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT / "frontend"
STATIC_DIR = ROOT / "static"


def build_frontend():
    if not (FRONTEND_DIR / "node_modules").is_dir():
        print("Installing frontend dependencies...")
        subprocess.run(["npm", "install"], cwd=FRONTEND_DIR, check=True)

    print("Building frontend...")
    subprocess.run(["npm", "run", "build"], cwd=FRONTEND_DIR, check=True)
    print(f"Frontend built to {STATIC_DIR}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument(
        "--prepare-dataset",
        action="store_true",
        help="Download the private test images and precomputed labels before starting the UI.",
    )
    args = parser.parse_args()

    if args.prepare_dataset:
        print("Preparing private test dataset and labels...")
        try:
            status = prepare_hf_dataset()
        except DatasetPreparationError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)
        print(f"Dataset ready: {status['image_count']} images")

    if not args.skip_build:
        build_frontend()

    if not STATIC_DIR.is_dir():
        print("ERROR: static/ not found. Run without --skip-build first.", file=sys.stderr)
        sys.exit(1)

    cmd = [
        sys.executable, "-m", "uvicorn",
        "app.main:app",
        "--host", args.host,
        "--port", str(args.port),
    ]
    if args.reload:
        cmd.append("--reload")

    print(f"Starting server on http://{args.host}:{args.port}")
    subprocess.run(cmd, cwd=ROOT)


if __name__ == "__main__":
    main()
