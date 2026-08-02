# AGENTS.md

## Setup and tests

```bash
uv sync
uv run --with pytest pytest
```

## Code layout

- `operai_eye/edge/`: Raspberry Pi capture, ONNX inference, and live camera UI
- `operai_eye/web/`: dataset explorer and evaluation API
- `operai_eye/pipeline/`: data, labeling, audit, and evaluation commands
- `operai_eye/training/`: DINOv3 and Moondream workflows
- `frontend/`: React explorer UI
- `deploy/`: Docker and systemd deployment
- `docs/`: operational guides and reports

## Primary commands

```bash
uv run operai-download-data --help
uv run operai-label-data --help
uv run operai-prepare-sft --help
uv run operai-evaluate-moondream --help
uv run operai-web --help
uv run --extra edge operai-edge --help
```

All commands are declared in `pyproject.toml`. Data is stored under `data/`,
artifacts under `output/`, models under `models/`, and runtime state under
`edge_state/`; these directories are not source code.

Valid phases are `IDLE`, `PATIENT_IN_ROOM`, `SURGERY_ACTIVE`, and `UNKNOWN`.

Labeling requires `OPENAI_API_KEY`, `MOONSHOT_API_KEY`, or `GEMINI_API_KEY`.
`OPENAI_BASE_URL` and `MODEL_NAME` are optional.
