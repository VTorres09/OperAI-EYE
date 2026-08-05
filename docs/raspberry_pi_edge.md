# Raspberry Pi DINOv3 edge service

The edge app captures five frames one second apart at the start of every
60-second interval. All five frames are preprocessed together and sent through
the DINOv3 ONNX graph in a single batch. Each frame is mapped to an OR phase,
then a majority vote produces the interval result.

The service uses the repository's fine-tuned DINOv3 ViT-B/16 model and the same
phase scoring as `operai_eye/training/dinov3/evaluate_lightly.py`:

- `IDLE = p(idle)`
- `PATIENT_IN_ROOM = p(people_in_room) * p(surgery_inactive)`
- `SURGERY_ACTIVE = p(people_in_room) * p(surgery_active)`

A 2-2-1 split does not have a majority and is stored as `UNKNOWN`. Results go
to a local SQLite database in WAL mode. Raw images are not retained by default.

## 1. Export the trained model

Do this once on a development machine, not on the Raspberry Pi. The private
model repository requires an authenticated Hugging Face account.

```bash
uv run hf auth login
uv run \
  --with "lightly-train==0.15.1" \
  --with "torchvision>=0.22,<0.23" \
  --with onnx \
  --with onnxruntime \
  operai-export-model --quantize
```

This downloads `exported_best.pt` from
`OperAI-Research/operai-eye-dinov3-vitb16-exocentric-rgb`, writes the float
ONNX model, and also writes a dynamic INT8 model. Each model gets a JSON
sidecar containing preprocessing and class-order metadata. Start with the INT8
artifact on the Pi, but benchmark both on the target hardware; operator
availability can make float inference faster on some builds.

If the checkpoint is already local:

```bash
uv run \
  --with "lightly-train==0.15.1" \
  --with "torchvision>=0.22,<0.23" \
  --with onnx \
  --with onnxruntime \
  operai-export-model \
  --checkpoint /path/to/exported_best.pt \
  --output models/operai_eye_dinov3.onnx \
  --quantize
```

Copy the selected `.onnx` file and its matching `.json` file to
`/opt/operai-eye/models/` on the Pi.

## 2. Raspberry Pi installation

Use a 64-bit Raspberry Pi OS installation. Picamera2 comes from Raspberry Pi
OS so it can access the matching libcamera stack.

```bash
sudo apt update
sudo apt install -y python3-picamera2
sudo useradd --system --create-home --groups video operai
sudo mkdir -p /opt/operai-eye /etc/operai-eye
sudo chown -R operai:operai /opt/operai-eye
```

Clone or copy this repository into `/opt/operai-eye`, then install the edge
runtime. The `--system-site-packages` flag exposes the OS Picamera2 package.

```bash
cd /opt/operai-eye
sudo -u operai uv venv --system-site-packages
sudo -u operai uv pip install --python .venv \
  -r deploy/raspberry-pi/requirements.txt
sudo -u operai uv pip install --python .venv --no-deps -e .
sudo cp deploy/raspberry-pi/config.example.toml /etc/operai-eye/config.toml
sudo cp deploy/raspberry-pi/operai-eye.service /etc/systemd/system/
```

The minimal requirements intentionally omit the training stack and PyTorch;
the Pi runs the converted model with ONNX Runtime.

Edit `/etc/operai-eye/config.toml` so the model filename, rotation, and camera
backend match the installation. Test one burst before enabling the daemon:

```bash
sudo -u operai /opt/operai-eye/.venv/bin/operai-edge \
  --config /etc/operai-eye/config.toml once
```

Then enable the 24/7 service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now operai-eye
sudo systemctl status operai-eye
journalctl -u operai-eye -f
```

Systemd restarts the process after crashes. The app also exits after five
consecutive failed bursts so that a stuck camera or inference process is
reinitialized.

## 3. Local status and results

```bash
sudo -u operai /opt/operai-eye/.venv/bin/operai-edge \
  --config /etc/operai-eye/config.toml status
```

The status command reports the latest burst and phase counts for the last 24
hours. Detailed interval and per-frame predictions are in:

```text
/var/lib/operai-eye/predictions.sqlite3
```

SQLite stores probabilities, timing, capture counts, and errors. With
`retain_images = "none"` it stores no image pixels. `uncertain` retains only
`UNKNOWN` bursts and `all` retains every burst. Image cleanup uses
`retention_days`.

## 4. Live camera dashboard

For setup, demonstrations, or a permanently attached display, start the local
camera UI:

```bash
uv run --extra edge operai-edge --config ./edge.local.toml ui
```

The command opens `http://127.0.0.1:8765`. Press **Enable camera** and grant
camera access to the browser. The page shows a continuous preview, samples five
frames one second apart, sends all five to the local ONNX model in one request,
and presents the individual predictions and majority vote. It repeats on the
configured 60-second schedule and writes results to the same SQLite database.

This mode deliberately lets the browser own the camera. On macOS that means
camera permission belongs to Safari, Chrome, or the browser you use—not to the
terminal application. No image or prediction leaves the device.

To view the dashboard from another computer on the same trusted network:

```bash
operai-edge --config /etc/operai-eye/config.toml ui \
  --host 0.0.0.0 --port 8765 --no-open
```

Browser camera APIs require a secure context. `localhost` works without HTTPS;
opening the Pi by a raw LAN IP may require an HTTPS reverse proxy. The ordinary
`run` command remains the recommended unattended, display-free production mode.

On a Raspberry Pi desktop that is already running the unattended user service,
install the demo helper once:

```bash
install -Dm755 deploy/raspberry-pi/operai-eye-demo \
  ~/.local/bin/operai-eye-demo
```

Then open a terminal through Raspberry Pi Connect screen sharing and run:

```bash
~/.local/bin/operai-eye-demo
```

The helper pauses unattended capture, opens the dashboard on the Pi itself, and
restores the production service when the demo exits. Click **Enable camera** in
the dashboard to start the first five-frame observation.

## 5. Offline dataset replay

The EgoExOR download is stored as HDF5 frame arrays rather than ordinary video
files. The extracted directories are the quickest debug source:

```bash
uv run --extra edge operai-edge --config ./edge.local.toml replay-directory \
  data/exocentric_rgb/validation/MISS/2/take_1/external_1 \
  --max-bursts 20
```

Use `--frame-step N` to sample every Nth extracted frame. Offline replay skips
the real-time waits unless `--realtime` is supplied.

If an original HDF5 file was retained:

```bash
uv run --extra edge operai-edge --config ./edge.local.toml replay-hdf5 \
  data/raw/miss_1.h5 \
  --take-path data/MISS/1/take/2 \
  --camera external_1 \
  --frame-step 30 \
  --max-bursts 20
```

Conventional video files are also supported:

```bash
uv run --extra edge operai-edge --config ./edge.local.toml replay-video \
  sample.mp4 --frame-step 30 --max-bursts 20
```

`frame-step` depends on source FPS. For a 30 FPS video, `30` approximates one
sample per second.

## Operational notes

- Keep the camera fixed. The current test results vary materially by camera
  angle, so validate the deployment view before relying on phase trends.
- Use wired networking and a UPS if the device must run continuously, even
  though inference itself is fully offline.
- The classifier is an observational aid, not a clinical safety system.
- Monitor free disk space. The default stores only small SQLite records, while
  retaining all 7,200 daily captures can consume substantial storage.
