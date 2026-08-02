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

Rows labeled `UNKNOWN` remain in the raw label CSVs for auditability, but are
excluded from staged train/validation archives and model evaluation.

### Audit label consistency

Use the Audit tab in the visualization app to review temporal label candidates,
save correction decisions, rebuild corrected train/validation archives, and
publish the corrected SFT dataset back to Hugging Face.

For a non-UI report of the same heuristics:

```bash
uv run python audit_labels.py --json
```

The script writes review candidates to `output/label_audit_suspects.csv` and a
summary to `output/label_audit_summary.json`. The UI reads the pinned Hugging
Face SFT dataset, stores reviewer decisions in `output/sft_label_corrections.csv`,
and publishes corrected archives only when you press Publish.

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

### Fine-tune DINOv3 with LightlyTrain

Prepare multilabel CSVs from the pinned Hugging Face SFT dataset:

```bash
uv run python -m finetuning.dinov3.train_lightly prepare
```

The mapping is hierarchical:

| Phase | Multilabel targets |
|---|---|
| `IDLE` | `idle` |
| `PATIENT_IN_ROOM` | `people_in_room`, `surgery_inactive` |
| `SURGERY_ACTIVE` | `people_in_room`, `surgery_active` |

When you are ready to run the fine-tune, install LightlyTrain for that command
and launch the DINOv3 ViT-B/16 multilabel classifier:

```bash
uv run --with lightly-train python -m finetuning.dinov3.train_lightly train \
  --model dinov3/vitb16 \
  --steps auto \
  --batch-size auto
```

Outputs are prepared under `output/lightly_dinov3/` and Lightly checkpoints/logs
go under `output/lightly_dinov3/runs/dinov3_vitb16_multilabel/`.
The completed DINOv3 run report is in `finetuning/dinov3/REPORT.md`.

### Run DINOv3 continuously on Raspberry Pi

The `edge_app/` package is a 24/7 Raspberry Pi runtime for the fine-tuned
DINOv3 model. Every minute it captures a five-frame burst at one frame per
second, performs one batched ONNX inference call, and stores the majority-vote
result locally. It includes Picamera2/USB camera support, systemd deployment,
SQLite history, bounded optional image retention, and replay from extracted
frames, EgoExOR HDF5 files, or conventional video.

See [`docs/raspberry_pi_edge.md`](docs/raspberry_pi_edge.md) for model export,
installation, replay, and operations.

To open the polished live camera view locally, use:

```bash
uv run --extra edge operai-edge --config edge.local.toml ui
```

Grant camera access in the browser. It displays the live feed, captures a
five-frame burst, runs one local batch, and visualizes the per-frame predictions
and majority vote every minute.

For a compact container deployment with ONNX Runtime and the camera UI in one
service:

```bash
cp .env.compose.example .env
docker compose up -d --build
```

Then open `http://localhost:8080`. See
[`docs/docker_compose.md`](docs/docker_compose.md) for Raspberry Pi guidance,
INT8/FP32 selection, persistent data, health checks, and GPU notes.

### 3. Evaluate model

```bash
uv run python evaluate_moondream.py [--limit N]
```

Runs [Moondream2](https://huggingface.co/vikhyatk/moondream2) locally and compares predictions against API labels. Outputs `output/eval_results.csv` and `output/eval_metrics.json`.

By default, evaluation prepares the private HF test dataset first. Evaluation CSV/JSON files under `output/eval_*` are intentionally committable; the dataset and labels remain ignored. Rows with `UNKNOWN` ground truth are ignored for metric computation.

## Visualization

```bash
uv run python run.py [--no-prepare-dataset] [--port 8000] [--reload]
```

Builds the React frontend and starts a FastAPI server serving both the API (`/api/*`) and the SPA on a single port.

If the private test dataset is not cached, the Explorer view shows a download button. The download uses `HF_TOKEN` from `.env` or the environment, and the backend reads the resulting Hugging Face snapshot directly.

The Audit view uses the private SFT dataset
`OperAI-Research/operai-eye-exocentric-rgb-sft`. Its prepare button downloads
and extracts `train.zip` and `validation.zip` from Hugging Face, then displays
review candidates from the train/validation labels. Rebuild Archives creates a
corrected local stage under `output/sft_dataset_corrected/`; Publish to HF
uploads that stage and updates `output/sft_dataset_revision.json`.

For frontend-only development:

```bash
cd frontend && npm install && npm run dev
```

The Vite dev server proxies `/api` requests to `localhost:8000`.

## Project structure

```
├── download_exocentric_rgb.py   # Step 1: download & extract dataset
├── label_data.py                # Step 2: LLM-based image labeling
├── audit_labels.py              # Temporal/confidence label review candidates
├── prepare_sft_dataset.py       # Analyze, stage, publish, and verify SFT data
├── finetuning/                  # Model-specific fine-tuning entry points
│   ├── dinov3/                  # LightlyTrain DINOv3 multilabel classifier
│   └── moondream/               # Resumable Moondream Cloud SFT and report
├── edge_app/                    # Raspberry Pi capture, DINO inference, voting
├── deploy/raspberry-pi/         # systemd unit and production config example
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
