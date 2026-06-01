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

## Pipeline

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
python evaluate_moondream.py --labels output/validation_labels.csv [--limit N]
```

Runs [Moondream2](https://huggingface.co/vikhyatk/moondream2) locally and compares predictions against API labels. Outputs `output/eval_results.csv` and `output/eval_metrics.json`.

## Visualization

```bash
uv run python run.py [--port 8000] [--reload]
```

Builds the React frontend and starts a FastAPI server serving both the API (`/api/*`) and the SPA on a single port.

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
