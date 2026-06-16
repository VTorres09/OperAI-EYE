#!/usr/bin/env python3
"""Label surgical OR images using OpenAI-compatible API with async concurrency.

Usage:
    python label_data.py --split validation
    python label_data.py --split test --prompt prompts/custom.txt
    python label_data.py --split test --concurrency 20 --limit 100
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import csv
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_PROMPT_PATH = Path("prompts/or_phase.txt")
DATA_DIR = Path("data/exocentric_rgb")
OUTPUT_DIR = Path("output")
VALID_PHASES = {"IDLE", "PATIENT_IN_ROOM", "SURGERY_ACTIVE", "UNKNOWN"}
FIELDNAMES = [
    "path",
    "split",
    "surgery_type",
    "procedure_id",
    "take_id",
    "camera",
    "frame_id",
    "phase",
    "confidence",
    "key_visual_cues",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Label OR images using async API calls.")
    parser.add_argument("--split", choices=["train", "validation", "test"], required=True, help="Split to label")
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT_PATH, help="Prompt file path")
    parser.add_argument("--model", default=os.getenv("MODEL_NAME", "gemini-2.0-flash"), help="Model name")
    parser.add_argument("--concurrency", type=int, default=10, help="Number of concurrent requests")
    parser.add_argument("--limit", type=int, help="Limit number of images to process")
    parser.add_argument("--seed", type=int, default=42, help="Random sampling seed")
    parser.add_argument("--dry-run", action="store_true", help="List images without submitting")
    parser.add_argument("--rephase", help="Force re-labeling of images currently labeled with this phase (e.g. TURNOVER)")
    return parser.parse_args()


def get_client() -> AsyncOpenAI:
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url and "googleapis" in base_url:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error("GEMINI_API_KEY required when using Google base URL")
            sys.exit(1)
    elif base_url and "moonshot" in base_url:
        api_key = os.getenv("MOONSHOT_API_KEY")
        if not api_key:
            logger.error("MOONSHOT_API_KEY required when using Moonshot base URL")
            sys.exit(1)
    else:
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("MOONSHOT_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error("Set OPENAI_API_KEY, MOONSHOT_API_KEY, or GEMINI_API_KEY environment variable")
            sys.exit(1)
    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return AsyncOpenAI(**kwargs)


def load_prompt(prompt_path: Path) -> str:
    if not prompt_path.exists():
        logger.error("Prompt file not found: %s", prompt_path)
        sys.exit(1)
    return prompt_path.read_text().strip()


def deterministic_sample(paths: list[Path], limit: int | None, seed: int) -> list[Path]:
    """Return a stable shuffled prefix so larger samples contain smaller ones."""

    sampled = sorted(paths)
    random.Random(seed).shuffle(sampled)
    if limit is not None:
        sampled = sampled[:limit]
    return sampled


def discover_images(
    split: str,
    limit: int | None = None,
    seed: int = 42,
) -> list[Path]:
    split_dir = DATA_DIR / split
    if not split_dir.exists():
        logger.error("Split directory not found: %s", split_dir)
        sys.exit(1)
    all_images = list(split_dir.rglob("*.png"))
    images = deterministic_sample(all_images, limit, seed)
    logger.info(
        "Selected %d of %d images from %s split with seed %d",
        len(images),
        len(all_images),
        split,
        seed,
    )
    return images


def parse_image_path(path: Path) -> dict[str, str]:
    parts = path.relative_to(DATA_DIR).parts
    return {
        "split": parts[0],
        "surgery_type": parts[1],
        "procedure_id": parts[2],
        "take_id": parts[3].replace("take_", ""),
        "camera": parts[4],
        "frame_id": path.stem.replace("frame_", ""),
    }


def load_existing_labels(output_csv: Path) -> set[str]:
    if not output_csv.exists():
        return set()
    labeled = set()
    with output_csv.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            path = (row.get("path") or "").strip()
            phase = (row.get("phase") or "").strip()
            error = (row.get("error") or "").strip()
            if path and not error and phase in VALID_PHASES:
                labeled.add(path)
    logger.info("Found %d already labeled images in %s", len(labeled), output_csv)
    return labeled


def load_phase_map(output_csv: Path) -> dict[str, str]:
    if not output_csv.exists():
        return {}
    phase_map = {}
    with output_csv.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            path = (row.get("path") or "").strip()
            phase = (row.get("phase") or "").strip()
            if path:
                phase_map[path] = phase
    return phase_map


def image_to_base64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def parse_response(content: str) -> dict[str, Any]:
    content = content.strip()
    if content.startswith("```"):
        lines = content.split("\n")
        content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    import re
    phase_match = re.search(r'"phase"\s*:\s*"([^"]+)"', content)
    conf_match = re.search(r'"confidence"\s*:\s*([\d.]+)', content)
    cues_match = re.search(r'"key_visual_cues"\s*:\s*\[([^\]]*)\]', content)
    if phase_match:
        phase = phase_match.group(1)
        confidence = float(conf_match.group(1)) if conf_match else 0.5
        cues = []
        if cues_match:
            cues = re.findall(r'"([^"]+)"', cues_match.group(1))
        return {"phase": phase, "confidence": confidence, "key_visual_cues": cues}
    return {"phase": "UNKNOWN", "confidence": 0.0, "key_visual_cues": ["parse_error"]}


async def label_image(
    client: AsyncOpenAI,
    image_path: Path,
    prompt: str,
    model: str,
    semaphore: asyncio.Semaphore,
    max_retries: int = 5,
) -> dict[str, Any]:
    meta = parse_image_path(image_path)
    rel_path = str(image_path.relative_to(DATA_DIR))
    async with semaphore:
        b64 = image_to_base64(image_path)
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                                },
                            ],
                        }
                    ],
                    max_tokens=1024,
                    extra_body={"thinking": {"type": "disabled"}},
                )
                content = response.choices[0].message.content or ""
                parsed = parse_response(content)
                phase = parsed.get("phase", "UNKNOWN")
                if phase not in VALID_PHASES:
                    phase = "UNKNOWN"
                return {
                    "path": rel_path,
                    **meta,
                    "phase": phase,
                    "confidence": parsed.get("confidence", 0.0),
                    "key_visual_cues": "|".join(parsed.get("key_visual_cues", [])),
                    "error": "",
                }
            except Exception as e:
                error_str = str(e).lower()
                is_rate_limit = "429" in str(e) or "rate" in error_str or "quota" in error_str
                is_server_error = "500" in str(e) or "502" in str(e) or "503" in str(e) or "504" in str(e)
                if (is_rate_limit or is_server_error) and attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 5
                    logger.warning("Rate limit/server error for %s, retry %d/%d in %ds: %s", rel_path, attempt + 1, max_retries, wait_time, e)
                    await asyncio.sleep(wait_time)
                else:
                    logger.warning("Error processing %s after %d attempts: %s", rel_path, attempt + 1, e)
                    return {
                        "path": rel_path,
                        **meta,
                        "phase": "ERROR",
                        "confidence": 0.0,
                        "key_visual_cues": "",
                        "error": str(e),
                    }
        return {
            "path": rel_path,
            **meta,
            "phase": "ERROR",
            "confidence": 0.0,
            "key_visual_cues": "",
            "error": "Max retries exceeded",
        }


def append_to_csv(results: list[dict[str, Any]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_csv.exists()
    with output_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerows(results)


def update_csv_rows(results: list[dict[str, Any]], output_csv: Path) -> None:
    if not output_csv.exists():
        append_to_csv(results, output_csv)
        return
    updated_paths = {r["path"] for r in results}
    rows = []
    with output_csv.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("path") in updated_paths:
                continue
            rows.append(row)
    rows.extend(results)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


async def process_batch(
    client: AsyncOpenAI,
    images: list[Path],
    prompt: str,
    model: str,
    concurrency: int,
    output_csv: Path,
) -> tuple[int, int]:
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [label_image(client, img, prompt, model, semaphore) for img in images]
    success = 0
    errors = 0
    batch_size = 100
    for i in tqdm(range(0, len(tasks), batch_size), desc="Processing batches"):
        batch_tasks = tasks[i : i + batch_size]
        batch_results = await asyncio.gather(*batch_tasks)
        # Always upsert so retried failures replace their previous row.
        update_csv_rows(batch_results, output_csv)
        for r in batch_results:
            if r["error"]:
                errors += 1
            else:
                success += 1
    return success, errors


async def main_async() -> int:
    args = parse_args()
    client = get_client()
    prompt = load_prompt(args.prompt)
    if args.rephase:
        split_dir = DATA_DIR / args.split
        all_images = deterministic_sample(
            list(split_dir.rglob("*.png")),
            None,
            args.seed,
        )
        output_csv = OUTPUT_DIR / f"{args.split}_labels.csv"
        phase_map = load_phase_map(output_csv)
        to_process = [
            img for img in all_images
            if str(img.relative_to(DATA_DIR)) in phase_map
            and phase_map.get(str(img.relative_to(DATA_DIR))) == args.rephase
        ]
        if args.limit:
            to_process = to_process[:args.limit]
        existing = load_existing_labels(output_csv)
        logger.info("Re-labeling %d images with phase %s", len(to_process), args.rephase)
    else:
        images = discover_images(args.split, args.limit, args.seed)
        output_csv = OUTPUT_DIR / f"{args.split}_labels.csv"
        existing = load_existing_labels(output_csv)
        to_process = [img for img in images if str(img.relative_to(DATA_DIR)) not in existing]
    if not to_process:
        logger.info("No images to process")
        return 0
    logger.info("Processing %d images (%d already labeled)", len(to_process), len(existing))
    if args.dry_run:
        logger.info("Dry run: would process %d images", len(to_process))
        return 0
    start_time = time.time()
    success, errors = await process_batch(
        client,
        to_process,
        prompt,
        args.model,
        args.concurrency,
        output_csv,
    )
    elapsed = time.time() - start_time
    logger.info("=" * 50)
    logger.info("LABELING COMPLETE")
    logger.info("=" * 50)
    logger.info("Success: %d | Errors: %d", success, errors)
    logger.info("Time: %.1fs (%.2f images/sec)", elapsed, success / elapsed if elapsed > 0 else 0)
    logger.info("Output: %s", output_csv)
    return 0 if errors == 0 else 1


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    sys.exit(main())
