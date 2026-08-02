# DINOv3 LightlyTrain Fine-Tuning Report

Generated: 2026-06-17

## Summary

The completed DINOv3 ViT-B/16 LightlyTrain run is the new best local vision
classifier for the corrected exocentric RGB SFT dataset.

- Model family: `dinov3/vitb16`
- Task: multilabel image classification
- Best checkpoint step: `8299`
- Best validation micro-accuracy: `0.9246`
- Best validation macro-F1: `0.9090`
- Best validation AUROC macro: `0.9548`
- Best validation loss: `0.2257`
- Private Hub model repo:
  [`OperAI-Research/operai-eye-dinov3-vitb16-exocentric-rgb`](https://huggingface.co/OperAI-Research/operai-eye-dinov3-vitb16-exocentric-rgb)

The last checkpoint completed at step `15999`, but validation had drifted below
the best checkpoint. Use `exported_best.pt` for inference.

## Data

Dataset repository:
[`OperAI-Research/operai-eye-exocentric-rgb-sft`](https://huggingface.co/datasets/OperAI-Research/operai-eye-exocentric-rgb-sft)

Corrected dataset revision: `3dc6de64b66ffb7586ca98cb6ce4379fee84c605`

Rows labeled `UNKNOWN` are ignored for DINOv3 training and validation. The
LightlyTrain CSVs were generated from the corrected SFT train/validation
archives, after the label audit and rebuild step.

| Split | Images | `idle` | `people_in_room` | `surgery_active` | `surgery_inactive` |
|---|---:|---:|---:|---:|---:|
| Train | 9,788 | 546 | 9,242 | 2,958 | 6,284 |
| Validation | 1,874 | 43 | 1,831 | 609 | 1,222 |

The phase-to-multilabel mapping is:

| Phase | Multilabel targets |
|---|---|
| `IDLE` | `idle` |
| `PATIENT_IN_ROOM` | `people_in_room`, `surgery_inactive` |
| `SURGERY_ACTIVE` | `people_in_room`, `surgery_active` |

## Training Setup

Training ran on Lightning AI using a single NVIDIA L4 GPU.

| Setting | Value |
|---|---|
| LightlyTrain | `0.15.1` |
| Model | `dinov3/vitb16` |
| Precision | `bf16-mixed` |
| Batch size | `128` |
| Gradient accumulation | `1` |
| Effective global batch size | `128` |
| Steps | `16,000` |
| Validation interval | `100` steps |
| Checkpoint interval | `500` steps |
| Seed | `42` |

Batch-size probing on the L4 showed that `512`, `384`, `256`, and `192` ran
out of memory. `128` was stable and used about `19.6 GB` of GPU memory in
LightlyTrain, with `nvidia-smi` reporting about `20.9 GB / 23 GB`.

The first launch used LightlyTrain's automatic batch size and reached step
`4000` with an effective batch size of `16`. The run was then resumed from
`last.ckpt` with batch size `128`. LightlyTrain saved `model_init_args["model"]`
but its resume path expected `model_init_args["model_name"]`; that metadata key
was added to `best.ckpt` and `last.ckpt` before resuming. The model weights were
not changed by that metadata compatibility patch.

## Results

Best checkpoint:

| Metric | Value |
|---|---:|
| Step | 8,299 |
| Validation loss | 0.2257 |
| Micro accuracy | 0.9246 |
| Macro F1 | 0.9090 |
| AUROC macro | 0.9548 |
| Average precision macro | 0.9349 |
| Hamming distance | 0.0754 |

Final checkpoint:

| Metric | Value |
|---|---:|
| Step | 15,999 |
| Validation loss | 0.2757 |
| Micro accuracy | 0.9117 |
| Macro F1 | 0.8898 |
| AUROC macro | 0.9435 |
| Average precision macro | 0.9275 |
| Hamming distance | 0.0883 |

Best per-class validation metrics:

| Class | Accuracy | F1 | AUROC | Avg precision |
|---|---:|---:|---:|---:|
| `idle` | 0.9989 | 0.9773 | 0.9997 | 0.9860 |
| `people_in_room` | 0.9989 | 0.9995 | 0.9997 | 1.0000 |
| `surgery_active` | 0.8506 | 0.7760 | 0.9097 | 0.8018 |
| `surgery_inactive` | 0.8501 | 0.8834 | 0.9101 | 0.9519 |

Final per-class validation metrics:

| Class | Accuracy | F1 | AUROC | Avg precision |
|---|---:|---:|---:|---:|
| `idle` | 0.9984 | 0.9655 | 1.0000 | 0.9984 |
| `people_in_room` | 0.9984 | 0.9992 | 1.0000 | 1.0000 |
| `surgery_active` | 0.8255 | 0.7282 | 0.8876 | 0.7744 |
| `surgery_inactive` | 0.8244 | 0.8662 | 0.8866 | 0.9372 |

## Test Split Metadata Evaluation

The exported best checkpoint was evaluated on the pinned private test dataset
revision `b363f3b89449b9d3367e19719178e1199d0b324d`. Ground-truth `UNKNOWN`
rows are ignored, leaving 22,892 labeled test images. The test split contains
one surgery type (`MISS`), five cameras, and three procedure/take groups.

Overall test result:

| Images | Correct | Accuracy | Macro-F1 |
|---:|---:|---:|---:|
| 22,892 | 17,115 | 0.7476 | 0.7802 |

Per-class test metrics:

| Class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| `IDLE` | 1,506 | 0.8627 | 0.9927 | 0.9231 |
| `PATIENT_IN_ROOM` | 15,558 | 0.9000 | 0.7073 | 0.7921 |
| `SURGERY_ACTIVE` | 5,828 | 0.5168 | 0.7920 | 0.6255 |

### Test Metrics by Camera

| Camera | Images | Correct | Accuracy | Macro-F1 | IDLE F1 | PIR F1 | SA F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| external_1 | 4,534 | 3,550 | 0.7830 | 0.8261 | 0.9970 | 0.8214 | 0.6600 |
| external_2 | 4,599 | 3,860 | 0.8393 | 0.7414 | 0.9711 | 0.9005 | 0.3525 |
| external_3 | 4,600 | 3,133 | 0.6811 | 0.7627 | 0.9878 | 0.6114 | 0.6890 |
| external_4 | 4,561 | 3,777 | 0.8281 | 0.7487 | 0.9226 | 0.8910 | 0.4324 |
| external_5 | 4,598 | 2,795 | 0.6079 | 0.6262 | 0.6852 | 0.5352 | 0.6582 |

Camera-level pattern: `external_2` has the best overall accuracy, while
`external_5` is the weakest overall camera. DINOv3 has high `SURGERY_ACTIVE`
recall on `external_3` and `external_5`, but lower precision on those same
cameras because it over-predicts `SURGERY_ACTIVE`.

### Test Camera Metrics by Class

| Camera | Class | Support | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| external_1 | `IDLE` | 335 | 0.9941 | 1.0000 | 0.9970 |
| external_1 | `PATIENT_IN_ROOM` | 2,986 | 0.8969 | 0.7575 | 0.8214 |
| external_1 | `SURGERY_ACTIVE` | 1,213 | 0.5690 | 0.7857 | 0.6600 |
| external_2 | `IDLE` | 319 | 0.9438 | 1.0000 | 0.9711 |
| external_2 | `PATIENT_IN_ROOM` | 3,730 | 0.9043 | 0.8968 | 0.9005 |
| external_2 | `SURGERY_ACTIVE` | 550 | 0.3488 | 0.3564 | 0.3525 |
| external_3 | `IDLE` | 366 | 0.9811 | 0.9945 | 0.9878 |
| external_3 | `PATIENT_IN_ROOM` | 2,562 | 0.9514 | 0.4504 | 0.6114 |
| external_3 | `SURGERY_ACTIVE` | 1,672 | 0.5355 | 0.9659 | 0.6890 |
| external_4 | `IDLE` | 301 | 0.8795 | 0.9701 | 0.9226 |
| external_4 | `PATIENT_IN_ROOM` | 3,493 | 0.8660 | 0.9175 | 0.8910 |
| external_4 | `SURGERY_ACTIVE` | 767 | 0.5303 | 0.3651 | 0.4324 |
| external_5 | `IDLE` | 185 | 0.5211 | 1.0000 | 0.6852 |
| external_5 | `PATIENT_IN_ROOM` | 2,787 | 0.9505 | 0.3724 | 0.5352 |
| external_5 | `SURGERY_ACTIVE` | 1,626 | 0.4989 | 0.9668 | 0.6582 |

### Test Metrics by Procedure and Take

| Procedure | Take | Images | Correct | Accuracy | Macro-F1 | IDLE F1 | PIR F1 | SA F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 12,802 | 9,610 | 0.7507 | 0.6827 | 0.9231 | 0.8242 | 0.3008 |
| 2 | 2 | 4,050 | 3,133 | 0.7736 | 0.5147 | 0.0000 | 0.7532 | 0.7909 |
| 3 | 6 | 6,040 | 4,372 | 0.7238 | 0.4825 | 0.0000 | 0.7180 | 0.7295 |

Procedure/take groups 2 and 3 have no ground-truth `IDLE` support after
excluding `UNKNOWN`, so their `IDLE` F1 is `0.0000` by definition.

## Artifacts

Remote Lightning run directory:

```text
/home/zeus/OperAI-EYE/output/lightly_dinov3/runs/dinov3_vitb16_multilabel/
```

Key files:

```text
checkpoints/best.ckpt
checkpoints/last.ckpt
exported_models/exported_best.pt
exported_models/exported_last.pt
train.log
```

Private Hugging Face model repo:

```text
OperAI-Research/operai-eye-dinov3-vitb16-exocentric-rgb
```

Uploaded files:

```text
exported_best.pt
checkpoints/best.ckpt
metrics.json
classes.json
training_args.json
README.md
```

## Notes

- Prefer the best checkpoint/export for inference and downstream evaluation.
- The final checkpoint is useful for traceability, but it underperforms the
  best checkpoint on macro-F1.
- The strongest remaining class-level weakness is `surgery_active`, which is
  much improved over the Moondream baseline but still lags the hierarchy labels
  for `idle` and `people_in_room`.
