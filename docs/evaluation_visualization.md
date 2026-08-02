# Dataset explorer and evaluation dashboard

The web application combines three workflows:

- **Explorer** browses labeled test images and metadata.
- **Evaluation** compares registered model runs and confusion matrices.
- **Audit** reviews suspicious SFT labels and records corrections.

The FastAPI backend lives in `operai_eye/web/`; the React frontend lives in
`frontend/`.

## Start the application

```bash
uv run operai-web
```

The command prepares the private test dataset when needed, builds the frontend,
and opens the app at <http://localhost:8000>. Use `--no-prepare-dataset` to
require an existing Hugging Face snapshot or `--skip-build` to reuse `static/`.

For frontend-only development:

```bash
cd frontend
npm install
npm run dev
```

Vite proxies API and image requests to the FastAPI server on port 8000.

## Register evaluation results

Evaluators can register their result files automatically. To add metadata
manually:

```bash
uv run operai-register-eval \
  --model-id moondream2_v1 \
  --model-name "Moondream2 2B" \
  --prompt prompts/or_phase_simple.txt \
  --description "Baseline evaluation"
```

Versioned artifacts live in `output/`:

```text
eval_metadata.json
eval_results_{model_id}.csv
eval_metrics_{model_id}.json
```

## Main API routes

| Route | Purpose |
|---|---|
| `GET /api/dataset/status` | Dataset availability and download state |
| `GET /api/images` | Filtered, paginated explorer data |
| `GET /api/eval/models` | Registered evaluation runs |
| `GET /api/eval/results/{model_id}` | Metrics and results for one model |
| `GET /api/eval/compare` | Side-by-side model comparison |
| `GET /api/audit/candidates` | Label-review candidates |

The backend implementation is split into `data.py`, `eval_data.py`, and
`audit_data.py` so each dashboard area has one obvious data layer.
