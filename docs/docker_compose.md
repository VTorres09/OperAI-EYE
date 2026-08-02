# Docker Compose deployment

The Compose stack runs the live OperAI-EYE application as three isolated
services:

| Service | Responsibility |
|---|---|
| `frontend` | Nginx serves the live camera UI and proxies `/api/*` |
| `backend` | FastAPI validates bursts, preprocesses images, applies voting, and stores SQLite history |
| `triton` | NVIDIA Triton serves the exported DINOv3 ONNX graph |

The browser owns the camera and sends one five-image burst to the backend. The
backend preprocesses those images into one `[5, 3, 224, 224]` FP32 tensor and
uses Triton's binary HTTP protocol for a single inference request. No cloud API
or external image transfer is involved.

## Prerequisites

- Docker Engine with Docker Compose
- A 64-bit Linux host (`amd64` or `arm64`)
- The exported ONNX model at `models/operai_eye_dinov3.int8.onnx`
- Enough memory and disk space for Triton and the model

The Triton 26.06 ARM64 image is approximately 8.5 GiB compressed because it
contains several NVIDIA and inference backends. Plan for a substantially larger
unpacked image on disk; the first `docker compose up` can take a while.

Export the model first if it is not already present:

```bash
uv run \
  --with "lightly-train==0.15.1" \
  --with "torchvision>=0.22,<0.23" \
  --with onnx \
  --with onnxruntime \
  python -m edge_app.export_model --quantize
```

The model remains outside the Docker image and is mounted read-only into the
Triton model repository.

## Start the application

```bash
cp .env.compose.example .env
docker compose config --quiet
docker compose up -d --build
docker compose ps
```

Open <http://localhost:8080>, press **Enable camera**, and grant camera access
to the browser. The first observation starts immediately and subsequent bursts
start every minute.

Follow startup or inference logs with:

```bash
docker compose logs -f triton backend frontend
```

Stop the services without deleting prediction history:

```bash
docker compose down
```

SQLite history is kept in the `operai-state` named volume. To intentionally
delete that volume as well, use `docker compose down --volumes`.

## INT8 versus FP32

The Compose default is the dynamically quantized INT8 model because the
existing artifact is roughly 82 MB instead of 327 MB for FP32. DINOv3 ViT-B is
dominated by linear transformer operations, so reduced weight precision can
help memory-constrained CPU deployments even though this is not a generative
LLM.

Quantization is still a deployment tradeoff. Benchmark latency and classification
quality on the actual camera view. To use the float model instead, set:

```dotenv
OPERAI_MODEL_PATH=./models/operai_eye_dinov3.onnx
```

Then recreate Triton:

```bash
docker compose up -d --force-recreate triton
```

Both artifacts expose the same `images` and `logits` tensor contract, so the
backend and Triton model configuration do not otherwise change.

## NVIDIA GPU host

The default Triton configuration deliberately creates one CPU ONNX Runtime
instance, which is appropriate for Raspberry Pi and ordinary Docker hosts. The
included GPU overlay grants the container access to one NVIDIA GPU and selects
the included `configs/gpu.pbtxt` CUDA execution configuration.

```bash
docker compose -f compose.yaml -f compose.gpu.yaml up -d --build
```

Install NVIDIA Container Toolkit on the host before using the overlay.

## Raspberry Pi notes

Triton 26.06 publishes native `linux/arm64` and `linux/amd64` container images,
and NVIDIA lists the ONNX Runtime CPU backend as supported on ARM-SBSA. NVIDIA
does not explicitly certify every Raspberry Pi and Raspberry Pi OS combination,
so treat the exact Pi deployment as a benchmark target rather than guaranteed
hardware acceleration.

Recommended starting point:

- Raspberry Pi 5 with 8 GB RAM
- 64-bit Ubuntu or Raspberry Pi OS
- the INT8 model
- fast SSD storage rather than a small SD card
- active cooling

Triton adds dynamic batching, scheduling, health endpoints, metrics, and model
management. A single camera making one request per minute will not gain much
from cross-request dynamic batching, and Triton has a much larger memory/disk
footprint than direct ONNX Runtime. The existing `operai-edge run` command is
therefore still the leanest Raspberry Pi deployment. Compose is most valuable
when several camera browsers share one inference host or operational separation
is more important than minimum footprint.

Camera APIs work on `localhost` without TLS. If the UI is opened through a Pi's
LAN IP, browsers generally require HTTPS before allowing camera access. Put an
HTTPS reverse proxy in front of port 8080 for remote access.

## Configuration

Common settings live in `.env`:

```dotenv
OPERAI_PORT=8080
OPERAI_MODEL_PATH=./models/operai_eye_dinov3.int8.onnx
TRITON_IMAGE_TAG=26.06-py3
TRITON_TIMEOUT_SECONDS=120
```

The schedule, vote threshold, and SQLite paths are defined in
`deploy/docker/config.toml`. Triton's model shape, batching, and ONNX Runtime
thread settings are defined in
`deploy/triton/model_repository/operai_eye_dinov3/config.pbtxt`.
