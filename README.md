# OperAI-EYE

Surgical operating room phase classification pipeline. Downloads exocentric RGB images from the [EgoExOR](https://huggingface.co/datasets/ardamamur/EgoExOR) dataset, labels them using a vision LLM API, evaluates local models against those labels, and visualizes results in a web UI.

## Phases

| Phase | Description |
|---|---|
| `IDLE` | Empty OR, no patient, no activity |
| `TURNOVER` | Staff cleaning/restocking, no patient |
| `PATIENT_IN_ROOM` | Patient present, no active surgery |
| `SURGERY_ACTIVE` | Active surgical procedure underway |
| `UNKNOWN` | View obstructed or unclassifiable |

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

### View the existing labels and evaluation

You do not need to rerun labeling or Moondream evaluation just to use the UI. The test labels and images are stored in the private Hugging Face dataset, while the completed evaluation results are tracked in this repository.

Make sure your Hugging Face account has access to `OperAI-Research/operai-eye-exocentric-rgb-test`, then run:

```bash
uv run hf auth login  # skip if already authenticated
uv run python run.py
```

Alternatively, put `HF_TOKEN=hf_...` in `.env`. A GitHub token is only needed to clone the repository; it is not used by the data or UI runtime.

On the first run, `run.py` downloads the private test dataset into the Hugging Face cache. It creates local symlinks for the images and for `output/test_labels.csv`, so Hugging Face remains the source of truth. The labels are intentionally not committed to Git; only evaluation artifacts are versioned in this repository. Use `--no-prepare-dataset` only when you want to start the server without these artifacts.

## Pipeline

### Prepare the private test dataset

```bash
HF_TOKEN=hf_... uv run python prepare_hf_dataset.py
```

This downloads `OperAI-Research/operai-eye-exocentric-rgb-test` into the Hugging Face cache and symlinks both `output/test_labels.csv` and `data/exocentric_rgb/test` to the cached snapshot.

### 1. Download data

```bash
python download_exocentric_rgb.py [--data-dir ./data] [--keep-hdf5]
```

Downloads HDF5 files from HuggingFace and extracts exocentric RGB images into `data/exocentric_rgb/{split}/...`. HDF5 files are deleted after extraction by default.

### 2. Label data

```bash
export OPENAI_API_KEY=sk-...   # or GEMINI_API_KEY
python label_data.py --split validation [--concurrency 10] [--limit N]
```

Sends images to an OpenAI-compatible vision API and writes labels to `output/{split}_labels.csv`. Resumes from existing output (skips already-labeled images).

Environment variables:
- `OPENAI_API_KEY` or `GEMINI_API_KEY` — required
- `OPENAI_BASE_URL` — custom API endpoint
- `MODEL_NAME` — override model (default: `gemini-2.0-flash`)

### 3. Evaluate model

```bash
python evaluate_moondream.py [--labels output/test_labels.csv] [--limit N]
```

Runs [Moondream2](https://huggingface.co/vikhyatk/moondream2) locally and compares predictions against API labels. Outputs `output/eval_results.csv` and `output/eval_metrics.json`.

By default, evaluation prepares the private HF test dataset first. Evaluation CSV/JSON files under `output/eval_*` are intentionally committable; the dataset and labels remain ignored.

## Visualization

```bash
uv run python run.py [--no-prepare-dataset] [--port 8000] [--reload]
```

Builds the React frontend and starts a FastAPI server serving both the API (`/api/*`) and the SPA on a single port.

If the private test dataset is not local, the Explorer view shows a download button. The download uses `HF_TOKEN` from `.env` or the environment and stores images in the Hugging Face cache rather than duplicating them under `data/`.

For frontend-only development:

```bash
cd frontend && npm install && npm run dev
```

The Vite dev server proxies `/api` requests to `localhost:8000`.

## Project structure

```
├── download_exocentric_rgb.py   # Step 1: download & extract dataset
├── label_data.py                # Step 2: LLM-based image labeling
├── evaluate_moondream.py        # Step 3: local model evaluation
├── run.py                       # Visualization server entry point
├── app/                         # FastAPI backend
├── frontend/                    # React + Vite frontend
├── prompts/                     # LLM prompt templates
├── data/                        # Downloaded images (gitignored)
├── output/                      # Labels & eval results (gitignored)
└── static/                      # Frontend build output (gitignored)
```

## Data layout

```
data/exocentric_rgb/{split}/{surgery_type}/{procedure_id}/take_{take_id}/{camera}/frame_{frame_id}.png
```

Splits: `train`, `validation`, `test`
