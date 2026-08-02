# Docker Compose deployment

The compact deployment runs OperAI-EYE as one self-contained service:

- FastAPI serves the live camera UI and API on one port.
- ONNX Runtime loads the DINOv3 model once and classifies each five-image burst
  in one local batch.
- SQLite prediction history is stored in a persistent Docker volume.

There is no Triton, CUDA, Nginx, cloud inference, or external image transfer.
The browser owns the camera and sends the five captured frames to the local app.

## Prerequisites

- Docker Engine with Docker Compose
- A 64-bit `amd64` or `arm64` host
- The exported ONNX model at `models/operai_eye_dinov3.onnx`

Export the model if it is not already present:

```bash
uv run \
  --with "lightly-train==0.15.1" \
  --with "torchvision>=0.22,<0.23" \
  --with onnx \
  --with onnxruntime \
  operai-export-model
```

The model is not copied into the image. Compose mounts it read-only when the
container starts, so changing models does not rebuild the application image.

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

Follow logs with:

```bash
docker compose logs -f app
```

Stop the service without deleting prediction history:

```bash
docker compose down
```

SQLite history is kept in the `operai-state` named volume. To intentionally
delete that volume too, use `docker compose down --volumes`.

## CPU and Apple GPU behavior

The container installs the CPU-only `onnxruntime` package. That package supports
both ARM CPUs and macOS, while avoiding CUDA libraries and NVIDIA images.

Docker Desktop runs Linux containers inside a VM and does not expose the Apple
GPU through ONNX Runtime. The container therefore uses CPU inference on macOS,
just like it does on Raspberry Pi.

For Apple GPU acceleration, run the app natively on macOS. The default
`model.execution_provider = "auto"` selects CoreML when the installed ONNX
Runtime exposes `CoreMLExecutionProvider`, with CPU fallback. The container
configuration explicitly uses `execution_provider = "cpu"`. CoreML may execute
only supported graph partitions, so this is acceleration with fallback rather
than a guarantee that every model operation runs on the Apple GPU.

## FP32 versus INT8

FP32 is the accuracy-first Compose default. Since the model is mounted rather
than copied, its 327 MB size does not increase the application image size.

After validating classification quality on the target camera view, the smaller
INT8 artifact can be selected in `.env`:

```dotenv
OPERAI_MODEL_PATH=./models/operai_eye_dinov3.int8.onnx
```

Then recreate the service:

```bash
docker compose up -d --force-recreate app
```

INT8 can reduce model memory and CPU latency, but the benefit and accuracy
impact should be measured on the actual Raspberry Pi.

## Raspberry Pi notes

Recommended starting point:

- Raspberry Pi 5 with 8 GB RAM
- 64-bit Raspberry Pi OS or Ubuntu
- fast SSD storage rather than a small SD card
- active cooling

Tune `intra_op_threads` in `deploy/docker/config.toml` for the target Pi. Four
threads is a sensible Pi 5 starting point. A single five-image request per minute
does not need a separate inference scheduler or dynamic batching server.

Camera APIs work on `localhost` without TLS. If the UI is opened through a Pi's
LAN IP, browsers generally require HTTPS before allowing camera access. Put an
HTTPS reverse proxy in front of port 8080 for remote access.

## Configuration

Common host settings live in `.env`:

```dotenv
OPERAI_PORT=8080
OPERAI_MODEL_PATH=./models/operai_eye_dinov3.onnx
```

The schedule, voting threshold, CPU thread count, and SQLite paths are defined
in `deploy/docker/config.toml`.
