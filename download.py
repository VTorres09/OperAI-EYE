#!/usr/bin/env python3
"""Download EgoExOR model weights, test JSON, and MISS HDF5 data."""
import argparse
import zipfile
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download


def download_model(output_dir: Path):
    print("Downloading EgoExOR model weights...")
    model_dir = output_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    zip_path = hf_hub_download(
        repo_id="ardamamur/EgoExOR",
        filename="llava-v1.5-7b-task-lora_hybridor_qlora_4perm_EgoExOR.zip",
        repo_type="model",
        cache_dir=str(output_dir / "cache"),
    )

    print(f"Extracting model to {model_dir}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(model_dir)

    print(f"Model extracted to {model_dir}")
    return model_dir


def download_test_json(output_dir: Path):
    print("Downloading test JSON...")
    path = hf_hub_download(
        repo_id="ardamamur/EgoExOR",
        filename="test_1perm_Falsetemp_Falsetempaug_EgoExOR_5k_samples_drophistory0.5.json",
        repo_type="model",
        cache_dir=str(output_dir / "cache"),
        local_dir=str(output_dir),
    )
    print(f"Test JSON saved to {path}")
    return path


def download_miss_data(output_dir: Path):
    print("Downloading MISS HDF5 files + splits...")
    data_dir = output_dir / "hdf5"
    data_dir.mkdir(parents=True, exist_ok=True)

    miss_files = ["miss_1.h5", "miss_2.h5", "miss_3.h5", "miss_4.h5", "splits.h5"]
    downloaded = []
    for fname in miss_files:
        print(f"  Downloading {fname}...")
        path = hf_hub_download(
            repo_id="ardamamur/EgoExOR",
            filename=fname,
            repo_type="dataset",
            cache_dir=str(output_dir / "cache"),
            local_dir=str(data_dir),
        )
        downloaded.append(path)
        print(f"  -> {path}")

    return downloaded, data_dir


def main():
    parser = argparse.ArgumentParser(description="Download EgoExOR evaluation artifacts")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory")
    parser.add_argument("--skip-model", action="store_true", help="Skip model download")
    parser.add_argument("--skip-data", action="store_true", help="Skip HDF5 data download")
    parser.add_argument("--miss-only", action="store_true", help="Download only MISS data (no ultrasound)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_model:
        download_model(output_dir)
        download_test_json(output_dir)

    if not args.skip_data:
        _, data_dir = download_miss_data(output_dir)

        miss_files = sorted(data_dir.glob("miss_*.h5"))
        if miss_files:
            print(f"\nTo merge MISS HDF5 files, run:")
            print(f"  python -m EgoExOR.data.utils.merge_h5 \\")
            print(f"    --data_dir {data_dir} \\")
            fargs = " ".join(f.name for f in miss_files)
            print(f"    --input_files {fargs} \\")
            splits = data_dir / "splits.h5"
            print(f"    --splits_file splits.h5 \\")
            print(f"    --output_file {output_dir / 'egoexor_miss.h5'}")

    print("\nDone!")


if __name__ == "__main__":
    main()
