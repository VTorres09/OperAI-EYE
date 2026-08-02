# Training and evaluation

Training code lives in `operai_eye/training/`; general evaluation and metadata
tools live in `operai_eye/pipeline/`.

## DINOv3

Prepare multilabel CSVs from the private SFT dataset:

```bash
uv run operai-train-dinov3 prepare
```

Train the ViT-B/16 classifier:

```bash
uv run --with lightly-train operai-train-dinov3 train \
  --model dinov3/vitb16 \
  --steps auto \
  --batch-size auto
```

Evaluate and export the trained model:

```bash
uv run operai-evaluate-dinov3
uv run \
  --with "lightly-train==0.15.1" \
  --with "torchvision>=0.22,<0.23" \
  --with onnx \
  --with onnxruntime \
  operai-export-model --quantize
```

Prepared files and runs are written below `output/lightly_dinov3/`; exported
edge models go in `models/`.

## Moondream

Start or resume cloud SFT after setting `MOONDREAM_API_KEY`:

```bash
uv run operai-train-moondream
```

Validate configuration without creating a job with `--dry-run`. Progress is
saved after each batch under `output/`.

Evaluate the base or fine-tuned model:

```bash
uv run operai-evaluate-moondream
uv run operai-evaluate-moondream-sft
```

Evaluation results and metrics use versioned `output/eval_results_*.csv` and
`output/eval_metrics_*.json` files and appear in the web dashboard.

See [the DINOv3 report](reports/dinov3.md) and
[the Moondream report](reports/moondream.md) for experiment details.
