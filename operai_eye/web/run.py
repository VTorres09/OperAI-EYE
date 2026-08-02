import argparse
import subprocess
import sys

from operai_eye.paths import FRONTEND_DIR, PROJECT_ROOT, STATIC_DIR
from operai_eye.pipeline.hf_dataset import (
    DatasetPreparationError,
    dataset_status,
    prepare_hf_dataset,
)


def build_frontend():
    if not (FRONTEND_DIR / "node_modules").is_dir():
        print("Installing frontend dependencies...")
        subprocess.run(["npm", "install"], cwd=FRONTEND_DIR, check=True)

    print("Building frontend...")
    subprocess.run(["npm", "run", "build"], cwd=FRONTEND_DIR, check=True)
    print(f"Frontend built to {STATIC_DIR}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument(
        "--prepare-dataset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Prepare the private test images and precomputed labels when missing "
            "(default: enabled)."
        ),
    )
    args = parser.parse_args()

    status = dataset_status()
    if args.prepare_dataset and not status["ready"]:
        print("Preparing private test dataset and labels...")
        try:
            status = prepare_hf_dataset()
        except DatasetPreparationError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        print(f"Dataset ready: {status['image_count']} images")
    elif status["ready"]:
        print(f"Dataset already ready: {status['image_count']} images")

    if not args.skip_build:
        build_frontend()

    if not STATIC_DIR.is_dir():
        print(
            "ERROR: static/ not found. Run without --skip-build first.", file=sys.stderr
        )
        return 1

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "operai_eye.web.main:app",
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    if args.reload:
        cmd.append("--reload")

    print(f"Starting server on http://{args.host}:{args.port}")
    try:
        return subprocess.run(cmd, cwd=PROJECT_ROOT, check=False).returncode
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
