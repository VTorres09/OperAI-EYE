# OperAI-EYE

Local operating-room scene classification with DINOv3. The project includes a
Raspberry Pi camera service, a browser dashboard, dataset preparation, labeling,
training, and evaluation tools.

## Quick start

Install Python 3.11 dependencies:

```bash
uv sync
```

Open the dataset explorer and evaluation dashboard:

```bash
uv run operai-web
```

Open the live camera dashboard with the local ONNX model:

```bash
uv run --extra edge operai-edge --config edge.local.toml ui
```

Run the compact Docker deployment:

```bash
cp .env.compose.example .env
docker compose up -d --build
```

Then open <http://localhost:8080>.

## Repository map

```text
operai_eye/
├── edge/       Raspberry Pi capture, ONNX inference, voting, and camera UI
├── web/        Dataset explorer and evaluation FastAPI backend
├── pipeline/   Data download, labeling, auditing, and evaluation commands
└── training/   DINOv3 and Moondream training/evaluation code
frontend/       React source for the dataset explorer
deploy/         Docker and Raspberry Pi deployment files
docs/           Focused guides and experiment reports
prompts/        Vision-labeling prompts
tests/          Python tests
```

Generated or local-only directories are ignored: `data/`, `models/`, `output/`,
`static/`, and `edge_state/`.

## Common commands

| Task | Command |
|---|---|
| Run tests | `uv run --with pytest pytest` |
| Download EgoExOR | `uv run operai-download-data` |
| Prepare private HF test data | `uv run operai-prepare-hf` |
| Label images | `uv run operai-label-data --split validation` |
| Audit labels | `uv run operai-audit-labels --json` |
| Prepare SFT data | `uv run operai-prepare-sft analyze` |
| Train DINOv3 | `uv run --with lightly-train operai-train-dinov3 train` |
| Evaluate DINOv3 | `uv run operai-evaluate-dinov3` |
| Export edge ONNX | `uv run operai-export-model --quantize` |
| Evaluate Moondream | `uv run operai-evaluate-moondream` |

Every command supports `--help`.

## Scene phases

| Phase | Meaning |
|---|---|
| `IDLE` | Empty room with no patient activity |
| `PATIENT_IN_ROOM` | Patient present without active surgery |
| `SURGERY_ACTIVE` | Active surgical procedure |
| `UNKNOWN` | Obstructed, uncertain, or no majority vote |

## Guides

- [Data and labeling pipeline](docs/data_pipeline.md)
- [Training and evaluation](docs/training.md)
- [Raspberry Pi edge service](docs/raspberry_pi_edge.md)
- [Docker deployment](docs/docker_compose.md)
- [Evaluation dashboard](docs/evaluation_visualization.md)
- [DINOv3 report](docs/reports/dinov3.md)
- [Moondream report](docs/reports/moondream.md)
