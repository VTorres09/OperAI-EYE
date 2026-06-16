# OperAI-EYE

Surgical operating room phase classification pipeline. Downloads exocentric RGB images from the [EgoExOR](https://huggingface.co/datasets/ardamamur/EgoExOR) dataset, labels them using a vision LLM API, evaluates local models against those labels, and visualizes results in a web UI.

## Phases

| Phase | Description |
|---|---|
| `IDLE` | Empty OR, no patient, no activity |
| `PATIENT_IN_ROOM` | Patient present, no active surgery |
| `SURGERY_ACTIVE` | Active surgical procedure underway |
| `UNKNOWN` | View obstructed or unclassifiable |

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Run the Python tests with:

```bash
uv run --with pytest pytest
```

### View the existing labels and evaluation

You do not need to rerun labeling or Moondream evaluation just to use the UI. The test labels and images are stored in the private Hugging Face dataset, while the completed evaluation results are tracked in this repository.

Make sure your Hugging Face account has access to `OperAI-Research/operai-eye-exocentric-rgb-test`, then run:

```bash
uv run hf auth login  # skip if already authenticated
uv run python run.py
```

Alternatively, put `HF_TOKEN=hf_...` in `.env`. A GitHub token is only needed to clone the repository; it is not used by the data or UI runtime.

On the first run, `run.py` downloads the private test dataset into the Hugging Face cache. The UI reads the images and `labels/test_labels.csv` directly from the pinned snapshot path. It does not copy or symlink dataset files into the repository. Only evaluation artifacts are versioned in Git. Use `--no-prepare-dataset` only when you want to require an already-cached snapshot.

## Pipeline

### Prepare the private test dataset

```bash
HF_TOKEN=hf_... uv run python prepare_hf_dataset.py
```

This downloads and validates `OperAI-Research/operai-eye-exocentric-rgb-test` in the Hugging Face cache, then prints the exact snapshot, image, and labels paths used by the application.

### 1. Download data

```bash
python download_exocentric_rgb.py [--data-dir ./data] [--keep-hdf5]
```

Downloads HDF5 files from HuggingFace and extracts exocentric RGB images into `data/exocentric_rgb/{split}/...`. HDF5 files are deleted after extraction by default.

### 2. Label data

```bash
export OPENAI_BASE_URL=https://api.moonshot.ai/v1
export MOONSHOT_API_KEY=...
export MODEL_NAME=kimi-k2.6
uv run python label_data.py --split train --limit 10000 --seed 42
uv run python label_data.py --split validation --limit 2000 --seed 42
```

Sends images to an OpenAI-compatible vision API and writes labels to `output/{split}_labels.csv`. Resumes from existing output (skips already-labeled images).

Environment variables:
- `OPENAI_API_KEY`, `MOONSHOT_API_KEY`, or `GEMINI_API_KEY` — required
- `OPENAI_BASE_URL` — custom API endpoint
- `MODEL_NAME` — override model (default: `gemini-2.0-flash`)

Sampling is deterministic. Increasing `--limit` with the same `--seed` keeps
the existing sample and appends new images. Successful rows are resumed while
failed rows are retried and replaced.

### Prepare and publish the SFT dataset

Run the analysis after each labeling pass:

```bash
uv run python prepare_sft_dataset.py analyze
```

The current target is 10,000 deterministic training images and 2,000 validation
images. The analysis may recommend validation increments of 500 up to 3,000
when minority-class or source metadata coverage is still insufficient. Label
the recommended limits, rerun analysis, then:

```bash
uv run python prepare_sft_dataset.py stage
uv run python prepare_sft_dataset.py publish
```

This creates and verifies the private dataset
`OperAI-Research/operai-eye-exocentric-rgb-sft`. Set `HF_TOKEN` in `.env` to a
token with write access to the organization.

### Fine-tune Moondream

Set `MOONDREAM_API_KEY` in `.env`, then launch the resumable cloud SFT run:

```bash
uv run python -m finetuning.moondream.finetune_moondream
```

The run uses rank 8, batch size 8, learning rate `2e-4`, up to three epochs,
and deterministic validation. Progress is saved after every training batch in
`output/moondream_sft_state.json`; rerunning the command resumes the same
fine-tune. Use `--dry-run` to validate the pinned Hub dataset without creating
a cloud job. The current Moondream run report is in
`finetuning/moondream/REPORT.md`.

### 3. Evaluate model

```bash
uv run python evaluate_moondream.py [--limit N]
```

Runs [Moondream2](https://huggingface.co/vikhyatk/moondream2) locally and compares predictions against API labels. Outputs `output/eval_results.csv` and `output/eval_metrics.json`.

By default, evaluation prepares the private HF test dataset first. Evaluation CSV/JSON files under `output/eval_*` are intentionally committable; the dataset and labels remain ignored.

## Visualization

```bash
uv run python run.py [--no-prepare-dataset] [--port 8000] [--reload]
```

Builds the React frontend and starts a FastAPI server serving both the API (`/api/*`) and the SPA on a single port.

If the private test dataset is not cached, the Explorer view shows a download button. The download uses `HF_TOKEN` from `.env` or the environment, and the backend reads the resulting Hugging Face snapshot directly.

For frontend-only development:

```bash
cd frontend && npm install && npm run dev
```

The Vite dev server proxies `/api` requests to `localhost:8000`.

## Project structure

```
├── download_exocentric_rgb.py   # Step 1: download & extract dataset
├── label_data.py                # Step 2: LLM-based image labeling
├── prepare_sft_dataset.py       # Analyze, stage, publish, and verify SFT data
├── finetuning/                  # Model-specific fine-tuning entry points
│   └── moondream/               # Resumable Moondream Cloud SFT and report
├── evaluate_moondream.py        # Step 3: local model evaluation
├── run.py                       # Visualization server entry point
├── app/                         # FastAPI backend
├── docs/                        # Historical implementation notes
├── frontend/                    # React + Vite frontend
├── prompts/                     # LLM prompt templates
├── tests/                       # Python unit tests
├── data/                        # Downloaded images (gitignored)
├── output/                      # Labels & eval results (gitignored)
└── static/                      # Frontend build output (gitignored)
```

## Data layout

```
data/exocentric_rgb/{split}/{surgery_type}/{procedure_id}/take_{take_id}/{camera}/frame_{frame_id}.png
```

Splits: `train`, `validation`, `test`
