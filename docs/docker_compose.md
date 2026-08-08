# Docker Compose deployment

The compact deployment runs OperAI-EYE as one self-contained service:

- FastAPI serves the live camera UI and API on one port.
- ONNX Runtime loads the DINOv3 model once and classifies each five-image burst
  in one local batch.
- SQLite prediction history is stored in a persistent Docker volume.

There is no Triton, CUDA, Nginx, cloud inference, or external image transfer.
On desktops, the browser owns the camera and sends the five captured frames to
the local app. The Raspberry Pi override uses Picamera2 inside the container so
the CSI camera does not depend on Chromium webcam support.

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

The dashboard includes a SQLite-backed capture history below the live view. It
shows observation counts, verdict distribution, inference latency, and recent
bursts. Frame predictions and verdicts are retained; captured images are not.

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

## Raspberry Pi deployment

Recommended starting point:

- Raspberry Pi 5 with 8 GB RAM
- 64-bit Raspberry Pi OS or Ubuntu
- fast SSD storage rather than a small SD card
- active cooling

Install Docker Engine and the Compose plugin using Docker's Debian instructions,
then create `.env` with the Pi's paths and numeric IDs:

```bash
sudo deploy/raspberry-pi/install-docker-debian
```

The repository installer follows Docker's official Debian apt-repository
method, enables the daemon at boot, and adds the invoking user to the `docker`
group. Reconnect after it completes so the new group membership is active.

```dotenv
OPERAI_PORT=8765
OPERAI_MODEL_PATH=./models/operai_eye_dinov3.int8.onnx
OPERAI_STATE_PATH=/home/vinicius/.local/state/operai-eye
OPERAI_UID=1000
OPERAI_GID=1000
OPERAI_VIDEO_GID=44
OPERAI_RENDER_GID=992
```

Build and start the Pi-specific deployment:

```bash
docker compose \
  -f compose.yaml \
  -f deploy/docker/compose.raspberry-pi.yaml \
  up -d --build
```

The override exposes the Raspberry Pi camera stack to the app, selects
Picamera2, and bind-mounts the SQLite state directory. The base Compose service
uses `restart: unless-stopped`, so Docker restarts the app after a process crash
or host reboot. The Docker daemon itself must also be enabled at boot.

Tune `intra_op_threads` in `deploy/docker/config.toml` for the target Pi. Four
threads is a sensible Pi 5 starting point. A single five-image request per minute
does not need a separate inference scheduler or dynamic batching server.

The Pi-native camera API is served by the local container and does not require a
browser camera permission prompt.

## Configuration

Common host settings live in `.env`:

```dotenv
OPERAI_PORT=8080
OPERAI_MODEL_PATH=./models/operai_eye_dinov3.onnx
```

The schedule, voting threshold, CPU thread count, and SQLite paths are defined
in `deploy/docker/config.toml`.
