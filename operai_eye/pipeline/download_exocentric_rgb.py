"""Download EgoExOR (legacy) from HuggingFace and extract exocentric RGB images
organized by train/validation/test splits.

Usage:
    python download_exocentric_rgb.py [--data-dir ./data] [--keep-hdf5] [--only FILES...]

Options:
    --data-dir    Root directory for raw downloads and output images (default: ./data)
    --keep-hdf5   Keep downloaded HDF5 files after extraction (default: delete to save space)
    --only        Process only the specified HDF5 files (e.g. --only miss_4.h5)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import h5py
from huggingface_hub import hf_hub_download
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ID = "ardamamur/EgoExOR"
HDF5_FILES = [
    "miss_1.h5",
    "miss_2.h5",
    "miss_3.h5",
    "miss_4.h5",
    "ultrasound_1.h5",
    "ultrasound_2.h5",
    "ultrasound_3.h5",
    "ultrasound_4.h5",
    "ultrasound_5_14.h5",
    "ultrasound_5_58.h5",
]
SPLITS_FILE = "splits.h5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download EgoExOR and extract exocentric RGB images by split."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./data"),
        help="Root directory for downloads and output (default: ./data)",
    )
    parser.add_argument(
        "--keep-hdf5",
        action="store_true",
        help="Keep HDF5 files after extraction (default: delete to save space)",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Process only these HDF5 files (e.g. --only miss_4.h5)",
    )
    return parser.parse_args()


def download_file(filename: str, data_dir: Path) -> Path:
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    local_path = raw_dir / filename
    if local_path.exists():
        logger.info("%s already exists, skipping download", filename)
        return local_path
    logger.info("Downloading %s from %s ...", filename, REPO_ID)
    downloaded = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type="dataset",
        local_dir=str(raw_dir),
        local_dir_use_symlinks=False,
    )
    return Path(downloaded)


def load_splits(splits_path: Path) -> dict[str, set[tuple[str, int, int, int]]]:
    splits: dict[str, set[tuple[str, int, int, int]]] = {}
    with h5py.File(splits_path, "r") as f:
        for split_name in ("train", "validation", "test"):
            ds = f[f"splits/{split_name}"]
            surgery_types = [
                s.decode("utf-8") if isinstance(s, bytes) else s
                for s in ds["surgery_type"]
            ]
            proc_ids = ds["procedure_id"].astype(int).tolist()
            take_ids = ds["take_id"].astype(int).tolist()
            frame_ids = ds["frame_id"].astype(int).tolist()
            splits[split_name] = set(zip(surgery_types, proc_ids, take_ids, frame_ids))
            logger.info(
                "Split '%s': %d frame entries", split_name, len(splits[split_name])
            )
    return splits


def get_take_exo_cameras(h5_file: h5py.File, take_path: str) -> dict[int, str]:
    sources_grp = h5_file[f"{take_path}/sources"]
    source_count = int(sources_grp.attrs["source_count"])
    exo: dict[int, str] = {}
    for i in range(source_count):
        attr_key = f"source_{i}"
        if attr_key in sources_grp.attrs:
            name = sources_grp.attrs[attr_key]
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            if str(name).startswith("external"):
                exo[i] = str(name)
    return exo


def get_take_frame_to_split(
    splits: dict[str, set[tuple[str, int, int, int]]],
) -> dict[tuple[str, int, int, int], str]:
    mapping: dict[tuple[str, int, int, int], str] = {}
    for split_name, entries in splits.items():
        for entry in entries:
            mapping[entry] = split_name
    return mapping


def _enumerate_takes(
    h5_file: h5py.File,
) -> list[tuple[str, str, str, int, int]]:
    takes: list[tuple[str, str, str, int, int]] = []
    if "data" not in h5_file:
        return takes
    for surgery_type in h5_file["data"]:
        for proc_id_str in h5_file[f"data/{surgery_type}"]:
            take_base = f"data/{surgery_type}/{proc_id_str}/take"
            if take_base not in h5_file:
                continue
            for take_id_str in h5_file[take_base]:
                rgb_path = f"{take_base}/{take_id_str}/frames/rgb"
                if rgb_path in h5_file:
                    takes.append(
                        (
                            surgery_type,
                            proc_id_str,
                            take_id_str,
                            int(proc_id_str),
                            int(take_id_str),
                        )
                    )
    return takes


def process_hdf5(
    h5_path: Path,
    output_dir: Path,
    frame_to_split: dict[tuple[str, int, int, int], str],
) -> int:
    logger.info("Processing %s ...", h5_path.name)
    total_saved = 0

    with h5py.File(h5_path, "r") as f:
        if "data" not in f:
            logger.warning("No 'data' group in %s", h5_path.name)
            return 0

        takes = _enumerate_takes(f)
        if not takes:
            logger.warning("No takes found in %s", h5_path.name)
            return 0

        for surgery_type, proc_id_str, take_id_str, proc_id, take_id in tqdm(
            takes, desc=f"{h5_path.name} takes", unit="take"
        ):
            take_path = f"data/{surgery_type}/{proc_id_str}/take/{take_id_str}"
            rgb_path = f"{take_path}/frames/rgb"
            rgb_ds = f[rgb_path]
            num_frames = rgb_ds.shape[0]

            exo_cam_indices = get_take_exo_cameras(f, take_path)
            if not exo_cam_indices:
                continue
            sorted_cams = sorted(exo_cam_indices.items())

            split_dirs: dict[tuple[str, str], Path] = {}
            for cam_idx, cam_name in sorted_cams:
                for split_name in ("train", "validation", "test"):
                    d = (
                        output_dir
                        / split_name
                        / surgery_type
                        / str(proc_id)
                        / f"take_{take_id}"
                        / cam_name
                    )
                    split_dirs[(split_name, cam_name)] = d

            needed = False
            for cam_idx, cam_name in sorted_cams:
                for split_name in ("train", "validation", "test"):
                    sample = split_dirs[(split_name, cam_name)] / "frame_000000.png"
                    if not sample.exists():
                        needed = True
                        break
                if needed:
                    break

            if not needed:
                all_exist = True
                for frame_idx in range(num_frames):
                    key = (surgery_type, proc_id, take_id, frame_idx)
                    split = frame_to_split.get(key)
                    if split is None:
                        continue
                    for cam_idx, cam_name in sorted_cams:
                        out_file = (
                            split_dirs[(split, cam_name)] / f"frame_{frame_idx:06d}.png"
                        )
                        if not out_file.exists():
                            all_exist = False
                            break
                    if not all_exist:
                        needed = True
                        break
                if not needed:
                    total_saved += sum(
                        1
                        for fi in range(num_frames)
                        if frame_to_split.get((surgery_type, proc_id, take_id, fi))
                    ) * len(sorted_cams)
                    continue

            logger.info(
                "Loading RGB for %s/%s/take_%d (%d frames, %d exo cams) ...",
                surgery_type,
                proc_id,
                take_id,
                num_frames,
                len(sorted_cams),
            )
            all_rgb = rgb_ds[:]
            frame_split_map: dict[int, str] = {}
            for frame_idx in range(num_frames):
                split = frame_to_split.get((surgery_type, proc_id, take_id, frame_idx))
                if split is not None:
                    frame_split_map[frame_idx] = split

            for cam_idx, cam_name in sorted_cams:
                for frame_idx, split in tqdm(
                    frame_split_map.items(),
                    desc=f"  {cam_name}",
                    unit="frame",
                    leave=False,
                ):
                    out_dir = split_dirs[(split, cam_name)]
                    out_file = out_dir / f"frame_{frame_idx:06d}.png"
                    if out_file.exists():
                        total_saved += 1
                        continue
                    out_dir.mkdir(parents=True, exist_ok=True)
                    frame = all_rgb[frame_idx, cam_idx]
                    img = Image.fromarray(frame)
                    img.save(out_file)
                    total_saved += 1

    logger.info("Saved %d images from %s", total_saved, h5_path.name)
    return total_saved


def main() -> int:
    args = parse_args()
    data_dir: Path = args.data_dir
    output_dir = data_dir / "exocentric_rgb"
    output_dir.mkdir(parents=True, exist_ok=True)

    files_to_process = args.only if args.only else HDF5_FILES
    for fname in files_to_process:
        if fname not in HDF5_FILES:
            logger.error("Unknown file: %s. Must be one of: %s", fname, HDF5_FILES)
            return 1

    splits_path = download_file(SPLITS_FILE, data_dir)
    splits = load_splits(splits_path)
    frame_to_split = get_take_frame_to_split(splits)

    total = 0
    for fname in files_to_process:
        h5_path = download_file(fname, data_dir)
        saved = process_hdf5(h5_path, output_dir, frame_to_split)
        total += saved
        if not args.keep_hdf5:
            logger.info("Deleting %s to free disk space ...", h5_path)
            h5_path.unlink()

    logger.info("Done! Total exocentric RGB images saved: %d", total)
    logger.info("Output directory: %s", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
