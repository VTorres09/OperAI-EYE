# Data and labeling pipeline

Pipeline commands live in `operai_eye/pipeline/`. Run them from the repository
root after `uv sync`.

## Download EgoExOR

```bash
uv run operai-download-data --data-dir ./data
```

This downloads `ardamamur/EgoExOR`, extracts exocentric RGB images, and deletes
the source HDF5 files by default. Add `--keep-hdf5` when the original arrays are
needed for replay.

Extracted images use this layout:

```text
data/exocentric_rgb/{split}/{surgery_type}/{procedure_id}/take_{take_id}/{camera}/frame_{frame_id}.png
```

## Prepare the private test dataset

Authenticate with Hugging Face, then resolve the pinned dataset snapshot:

```bash
uv run hf auth login
uv run operai-prepare-hf
```

The explorer reads images and `labels/test_labels.csv` directly from the Hugging
Face cache; it does not copy them into the repository.

## Label images

Configure one compatible API key in `.env` or the environment:

- `OPENAI_API_KEY`
- `MOONSHOT_API_KEY`
- `GEMINI_API_KEY`

`OPENAI_BASE_URL` and `MODEL_NAME` are optional overrides.

```bash
uv run operai-label-data --split train --limit 10000 --seed 42
uv run operai-label-data --split validation --limit 2000 --seed 42
```

Labels are written to `output/{split}_labels.csv`. Successful rows resume;
failed rows are retried. Sampling remains deterministic when a limit increases
with the same seed.

## Audit and publish SFT data

```bash
uv run operai-audit-labels --json
uv run operai-prepare-sft analyze
uv run operai-prepare-sft stage
uv run operai-prepare-sft publish
```

The audit command writes candidates and summaries under `output/`. The SFT
commands analyze class/source coverage, build verified archives, and publish the
private `OperAI-Research/operai-eye-exocentric-rgb-sft` dataset. Rows labeled
`UNKNOWN` remain auditable but are excluded from model training archives.
