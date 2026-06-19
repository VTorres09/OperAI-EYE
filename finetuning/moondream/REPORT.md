# Moondream Fine-Tuning Report

Generated: 2026-06-16

## Summary

The current best Moondream Cloud SFT run is the 10k-image run:

- Fine-tune ID: `01KV38VJ6DYECNHFCFSK2B96SN`
- Best model: `moondream3-preview/01KV38VJ6DYECNHFCFSK2B96SN@3751`
- Validation: 2,000 Kimi-labeled validation images
- Accuracy: 77.5%
- Core macro-F1: 0.7751 across `IDLE`, `PATIENT_IN_ROOM`, and `SURGERY_ACTIVE`

The earlier 1k-image run remains useful as a baseline, but it was evaluated on
the earlier 500-image validation sample. Treat the numbers as directional rather
than a strict apples-to-apples comparison unless the old checkpoint is re-run on
the 2,000-image validation split.

## Data

Dataset repository:
[`OperAI-Research/operai-eye-exocentric-rgb-sft`](https://huggingface.co/datasets/OperAI-Research/operai-eye-exocentric-rgb-sft)

Current pinned revision: `e63a91530e73cfe2e955c03d01fae58e07aa273f`

Source images come from local EgoExOR exocentric RGB splits and were sampled
with deterministic image-level shuffling using seed `42`. Labels were produced
with Kimi through the existing labeling script and preserved without confidence
filtering. The taxonomy used for SFT is:

- `IDLE`
- `PATIENT_IN_ROOM`
- `SURGERY_ACTIVE`
- `UNKNOWN`

### Current 10k/2k SFT Dataset

| Split | Local pool | Selected | IDLE | PATIENT_IN_ROOM | SURGERY_ACTIVE | UNKNOWN | Mean confidence | Median confidence |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 58,590 | 10,000 | 519 | 6,742 | 2,527 | 212 | 0.7610 | 0.7500 |
| Validation | 28,440 | 2,000 | 42 | 1,304 | 528 | 126 | 0.7345 | 0.7500 |

Representativeness checks passed for both splits.

| Split | Minimum core class rule | Metadata tolerance | Max procedure gap | Max take gap | Max camera gap | Stability gap |
|---|---:|---:|---:|---:|---:|---:|
| Train | 200 | 0.03 | 0.006300 | 0.006303 | 0.009700 | 0.032210 |
| Validation | 30 | 0.05 | 0.006363 | 0.006363 | 0.007500 | N/A |

Artifacts:

- Local analysis: `output/sft_analysis.json`
- Staged ImageFolder dataset: `output/sft_dataset/`
- Train archive: `output/sft_dataset/train.zip`
- Validation archive: `output/sft_dataset/validation.zip`
- Pinned Hub revision: `output/sft_dataset_revision.json`

### Earlier 1k/500 Sample

The first Moondream run used the deterministic prefix of the same sampling order.

| Split | Selected | IDLE | PATIENT_IN_ROOM | SURGERY_ACTIVE | UNKNOWN |
|---|---:|---:|---:|---:|---:|
| Train | 1,000 | 61 | 644 | 272 | 23 |
| Validation | 500 | 9 | 326 | 140 | 25 |

Dataset revision: `5348f727d5b49093d833470bcc2a6435614ed1f9`

## Fine-Tune Runs

All Moondream SFT runs used:

- Rank: `8`
- Batch size: `8`
- Learning rate: `2e-4`
- Seed: `42`
- Skill group: `query`
- Question: `Classify the operating room phase. Respond with exactly one of: IDLE, PATIENT_IN_ROOM, SURGERY_ACTIVE, UNKNOWN.`
- Target shape: `{"answer": "<PHASE>"}`

| Run | Train / Val | Dataset revision | Fine-tune ID | Best model | Best accuracy | Best core macro-F1 | Notes |
|---|---:|---|---|---|---:|---:|---|
| `operai-eye-or-phase-kimi-sft` | 1,000 / 500 | `5348f727d5b49093d833470bcc2a6435614ed1f9` | `01KV1043CNDNMDBNEEQX4H7BPR` | `moondream3-preview/01KV1043CNDNMDBNEEQX4H7BPR@250` | 0.7520 | 0.6306 | Best row is the corrected checkpoint evaluation. |
| `operai-eye-or-phase-kimi-sft-10k` | 10,000 / 2,000 | `e63a91530e73cfe2e955c03d01fae58e07aa273f` | `01KV38VJ6DYECNHFCFSK2B96SN` | `moondream3-preview/01KV38VJ6DYECNHFCFSK2B96SN@3751` | 0.7750 | 0.7751 | Resumed after an internet drop from batch 1134 and completed cleanly. |

### 10k Run Progression

| Stage | Step | Accuracy | Core macro-F1 | Mean SFT loss |
|---|---:|---:|---:|---:|
| Baseline | 0 | 0.0455 | 0.0366 | N/A |
| Epoch 1 | 1,251 | 0.7080 | 0.5466 | 0.0968 |
| Epoch 2 | 2,501 | 0.7470 | 0.6327 | 0.0886 |
| Epoch 3 | 3,751 | 0.7750 | 0.7751 | 0.1118 |

Final 10k per-class metrics:

| Class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| `IDLE` | 42 | 0.8571 | 0.8571 | 0.8571 |
| `PATIENT_IN_ROOM` | 1,304 | 0.8592 | 0.7952 | 0.8260 |
| `SURGERY_ACTIVE` | 528 | 0.5871 | 0.7083 | 0.6421 |
| `UNKNOWN` | 126 | 0.9035 | 0.8175 | 0.8583 |

## Test Split Metadata Evaluation

The fine-tuned Moondream3 checkpoint was re-evaluated on the pinned private test
dataset revision `b363f3b89449b9d3367e19719178e1199d0b324d`. Ground-truth
`UNKNOWN` rows are ignored, leaving 22,892 labeled test images. The test split
contains one surgery type (`MISS`), five cameras, and three procedure/take
groups.

Overall test result:

| Images | Correct | Accuracy | Macro-F1 |
|---:|---:|---:|---:|
| 22,892 | 18,206 | 0.7953 | 0.8117 |

Per-class test metrics:

| Class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| `IDLE` | 1,506 | 0.8752 | 0.9920 | 0.9300 |
| `PATIENT_IN_ROOM` | 15,558 | 0.8824 | 0.8096 | 0.8444 |
| `SURGERY_ACTIVE` | 5,828 | 0.6204 | 0.7062 | 0.6606 |

### Test Metrics by Camera

| Camera | Images | Correct | Accuracy | Macro-F1 | IDLE F1 | PIR F1 | SA F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| external_1 | 4,534 | 3,559 | 0.7850 | 0.8421 | 0.9985 | 0.8373 | 0.6905 |
| external_2 | 4,599 | 3,837 | 0.8343 | 0.7428 | 0.9800 | 0.8969 | 0.3515 |
| external_3 | 4,600 | 3,308 | 0.7191 | 0.7943 | 0.9945 | 0.6749 | 0.7136 |
| external_4 | 4,561 | 3,764 | 0.8253 | 0.7346 | 0.9270 | 0.8904 | 0.3866 |
| external_5 | 4,598 | 3,738 | 0.8130 | 0.7734 | 0.6968 | 0.8431 | 0.7804 |

Camera-level pattern: `external_2` has the best overall accuracy, while
`external_3` has the weakest overall accuracy but high `SURGERY_ACTIVE` recall.
`external_2` and `external_4` are the weakest cameras for `SURGERY_ACTIVE` F1.

### Test Camera Metrics by Class

| Camera | Class | Support | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| external_1 | `IDLE` | 335 | 1.0000 | 0.9970 | 0.9985 |
| external_1 | `PATIENT_IN_ROOM` | 2,986 | 0.8766 | 0.8014 | 0.8373 |
| external_1 | `SURGERY_ACTIVE` | 1,213 | 0.6951 | 0.6859 | 0.6905 |
| external_2 | `IDLE` | 319 | 0.9608 | 1.0000 | 0.9800 |
| external_2 | `PATIENT_IN_ROOM` | 3,730 | 0.9052 | 0.8887 | 0.8969 |
| external_2 | `SURGERY_ACTIVE` | 550 | 0.3355 | 0.3691 | 0.3515 |
| external_3 | `IDLE` | 366 | 0.9945 | 0.9945 | 0.9945 |
| external_3 | `PATIENT_IN_ROOM` | 2,562 | 0.9497 | 0.5234 | 0.6749 |
| external_3 | `SURGERY_ACTIVE` | 1,672 | 0.5682 | 0.9587 | 0.7136 |
| external_4 | `IDLE` | 301 | 0.8875 | 0.9701 | 0.9270 |
| external_4 | `PATIENT_IN_ROOM` | 3,493 | 0.8570 | 0.9264 | 0.8904 |
| external_4 | `SURGERY_ACTIVE` | 767 | 0.5198 | 0.3077 | 0.3866 |
| external_5 | `IDLE` | 185 | 0.5347 | 1.0000 | 0.6968 |
| external_5 | `PATIENT_IN_ROOM` | 2,787 | 0.8575 | 0.8292 | 0.8431 |
| external_5 | `SURGERY_ACTIVE` | 1,626 | 0.7977 | 0.7638 | 0.7804 |

### Test Metrics by Procedure and Take

| Procedure | Take | Images | Correct | Accuracy | Macro-F1 | IDLE F1 | PIR F1 | SA F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 12,802 | 10,816 | 0.8449 | 0.7402 | 0.9300 | 0.9010 | 0.3895 |
| 2 | 2 | 4,050 | 3,084 | 0.7615 | 0.5071 | 0.0000 | 0.7459 | 0.7752 |
| 3 | 6 | 6,040 | 4,306 | 0.7129 | 0.4745 | 0.0000 | 0.7300 | 0.6935 |

Procedure/take groups 2 and 3 have no ground-truth `IDLE` support after
excluding `UNKNOWN`, so their `IDLE` F1 is `0.0000` by definition.

## Local Organization

Moondream-specific code and reporting now live under:

```text
finetuning/moondream/
├── __init__.py
├── finetune_moondream.py
└── REPORT.md
```

Run the Moondream SFT entry point from the repository root:

```bash
uv run python -m finetuning.moondream.finetune_moondream
```

For the completed 10k experiment, the persisted state and final results are:

```bash
output/moondream_sft_10k_state.json
output/moondream_sft_10k_results.json
```

The default command still writes to `output/moondream_sft_state.json` and
`output/moondream_sft_results.json`. Use `--state` and `--results` when starting
a new experiment so completed runs remain immutable:

```bash
uv run python -m finetuning.moondream.finetune_moondream \
  --name operai-eye-or-phase-kimi-sft-next \
  --state output/moondream_sft_next_state.json \
  --results output/moondream_sft_next_results.json
```

## Notes for Future Model Experiments

- Reuse the pinned Hugging Face ImageFolder dataset for comparable experiments.
- Keep model-specific runners under `finetuning/<model_family>/`.
- Save each run's state and metrics with a model-specific filename under
  `output/`.
- If comparing directly against the 1k Moondream run, re-evaluate the old
  checkpoint on the 2,000-image validation set first.
