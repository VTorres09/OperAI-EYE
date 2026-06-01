# AGENTS.md

## Setup

```bash
uv sync  # Install dependencies (Python 3.11)
```

## Workflow

1. **Download data**: `python download_exocentric_rgb.py [--data-dir ./data] [--keep-hdf5]`
   - Downloads EgoExOR dataset from HuggingFace (`ardamamur/EgoExOR`)
   - Extracts exocentric RGB images to `data/exocentric_rgb/{split}/...`
   - Deletes HDF5 files by default to save space

2. **Label data**: `python label_data.py --split validation [--concurrency 10] [--limit N]`
   - Requires `OPENAI_API_KEY` or `GEMINI_API_KEY` (and optionally `OPENAI_BASE_URL`)
   - Outputs to `output/{split}_labels.csv`
   - Resumes from existing labels (skips already-labeled images)

3. **Evaluate model**: `python evaluate_moondream.py --labels output/validation_labels.csv [--limit N]`
   - Loads Moondream2 model (`vikhyatk/moondream2`)
   - Outputs results to `output/eval_results.csv` and metrics to `output/eval_metrics.json`

## Data Structure

```
data/exocentric_rgb/{split}/{surgery_type}/{procedure_id}/take_{take_id}/{camera}/frame_{frame_id}.png
```

Splits: `train`, `validation`, `test`

## Valid Phases

`IDLE`, `TURNOVER`, `PATIENT_IN_ROOM`, `SURGERY_ACTIVE`, `UNKNOWN`

## Environment Variables

- `OPENAI_API_KEY` or `GEMINI_API_KEY` - Required for labeling
- `OPENAI_BASE_URL` - Optional custom API endpoint
- `MODEL_NAME` - Override default model (default: `gemini-2.0-flash`)

## Visualization App

```bash
uv run python run.py [--port 8000] [--reload] [--skip-build]
```

- Builds the React frontend (`frontend/`) into `static/`, then starts FastAPI on a single port
- Serves both the API (`/api/*`) and the SPA on the same port
- `--skip-build` skips the frontend build if `static/` already exists
- `--reload` enables uvicorn hot-reload for backend development
- For frontend-only dev: `cd frontend && npm run dev` (proxies `/api` to `:8000`)

## Notes

- Prompts are in `prompts/or_phase.txt`
- All scripts use argparse; run with `--help` for options
- Labeling uses async concurrency (default 10 concurrent requests)
- Evaluation uses PyTorch with automatic device detection (MPS/CUDA/CPU)
