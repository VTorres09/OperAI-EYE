# OperAI-EYE

Evaluation of the [EgoExOR](https://github.com/ardamamur/EgoExOR) scene graph generation model on the MISS test split using exocentric RGB frames only.

## Setup

Requires Python 3.11, CUDA GPU, and [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install dependencies
uv sync

# 2. Clone EgoExOR repo and install LLaVA
bash setup.sh

# 3. Download model weights + test JSON + MISS HDF5 data (~62 GB)
python download.py --miss-only
```

After download, merge MISS HDF5 files:

```bash
python -m EgoExOR.data.utils.merge_h5 \
  --data_dir data/hdf5 \
  --input_files miss_1.h5 miss_2.h5 miss_3.h5 miss_4.h5 \
  --splits_file splits.h5 \
  --output_file data/egoexor_miss.h5
```

## Evaluate

```bash
python evaluate.py \
  --model_path data/model/llava-v1.5-7b-task-lora_hybridor_qlora_4perm_EgoExOR \
  --test_json data/test_1perm_Falsetemp_Falsetempaug_EgoExOR_5k_samples_drophistory0.5.json \
  --hdf5_path data/egoexor_miss.h5 \
  --output_csv eval_miss_exo_results.csv
```

## Output

- `eval_miss_exo_results.csv` — per-frame predictions with ground truth, triplet counts, and frame-level F1
- `eval_miss_exo_results.summary.txt` — overall precision, recall, F1
