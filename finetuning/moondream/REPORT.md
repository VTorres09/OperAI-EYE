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
