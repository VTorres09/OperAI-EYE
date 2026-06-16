# Evaluation Visualization with Model Versioning

Historical implementation notes for the evaluation dashboard.

## Overview

Implemented a complete evaluation visualization system in the web app with support for comparing multiple model versions side-by-side.

## Features

### Backend (`app/eval_data.py`)
- **Model Registration**: Track evaluation runs with metadata (model name, prompt, description)
- **Results Storage**: Versioned CSV files (`eval_results_{model_id}.csv`)
- **Metrics Calculation**: Automatic computation of accuracy, precision, recall, F1 per class
- **Confusion Matrix**: Full confusion matrix for each model
- **Model Comparison**: Side-by-side comparison of multiple models

### API Endpoints
- `GET /api/eval/models` - List all registered model evaluations
- `GET /api/eval/results/{model_id}` - Get detailed results for a specific model
- `GET /api/eval/compare?model_ids=id1,id2` - Compare multiple models
- `POST /api/eval/register` - Register a new model evaluation

### Frontend Components
- **EvalDashboard**: Main evaluation view with model selection and results display
- **ModelSelector**: Checkbox-based multi-model selection
- **MetricsDisplay**: Accuracy, precision, recall, F1 metrics with per-class breakdown
- **ConfusionMatrix**: Visual heatmap of prediction vs ground truth

### Navigation
Added tab-based navigation in the app header:
- **Explorer**: Original image browsing and filtering
- **Evaluation**: New model evaluation dashboard

## Usage

### 1. Register an Evaluation

```bash
# Register with explicit model ID
python register_eval.py \
  --model-id moondream2_v1 \
  --model-name "Moondream2 2B (2025-01-09)" \
  --prompt prompts/or_phase_simple.txt \
  --description "Initial evaluation with simple prompt"

# Or use --model-id flag in evaluate_moondream.py (auto-registers on completion)
python evaluate_moondream.py \
  --model-id moondream2_v2 \
  --model-name "Moondream2 2B (fine-tuned)" \
  --prompt prompts/or_phase_v2.txt
```

### 2. View Results

```bash
# Start the app
python run.py

# Open browser to http://localhost:8000
# Click "Evaluation" tab
```

### 3. Compare Models

In the Evaluation dashboard:
1. Select multiple models using checkboxes
2. View side-by-side comparison table
3. See per-class metrics for each model
4. Identify which model performs best for each class

## Current Evaluation Status

**Model**: moondream2_v1 (Moondream2 2B, revision 2025-01-09)
- **Progress**: 3,054 / 23,001 images (13.3%)
- **Accuracy**: 25.2%
- **Speed**: ~1.4 seconds per image
- **ETA**: ~7.5 hours remaining

### Per-Class Performance
- **IDLE**: 73.2% precision, 100% recall ✓
- **TURNOVER**: 0% (misclassified as IDLE)
- **PATIENT_IN_ROOM**: 0% (misclassified as SURGERY_ACTIVE)
- **SURGERY_ACTIVE**: 8.5% precision, 100% recall
- **UNKNOWN**: 0%

### Observations
As noted, the model shows clear bias patterns:
- PATIENT_IN_ROOM → SURGERY_ACTIVE (patient presence triggers surgery classification)
- TURNOVER → IDLE (empty room regardless of activity)

This suggests potential for a simplified 2-class system:
- **EMPTY_ROOM**: IDLE + TURNOVER
- **PATIENT_PRESENT**: PATIENT_IN_ROOM + SURGERY_ACTIVE

## File Structure

```
app/
  eval_data.py          # Evaluation data layer
  main.py               # Added eval endpoints

frontend/src/components/
  EvalDashboard.jsx     # Main evaluation view
  EvalDashboard.css
  ModelSelector.jsx     # Model selection UI
  ModelSelector.css
  MetricsDisplay.jsx    # Metrics cards and tables
  MetricsDisplay.css
  ConfusionMatrix.jsx   # Confusion matrix heatmap
  ConfusionMatrix.css

output/
  eval_metadata.json    # Model registry
  eval_results.csv      # Current evaluation
  eval_results_{id}.csv # Versioned results
  eval_metrics_{id}.json # Versioned metrics

register_eval.py        # CLI tool to register evaluations
```

## Next Steps

1. **Complete Current Evaluation**: Let moondream2_v1 finish (~7.5 hours)
2. **Implement 2-Class Classifier**: Merge classes based on observed patterns
3. **Fine-tune Moondream2**: Use labeled dataset to improve accuracy
4. **Register New Models**: Compare fine-tuned vs base model
5. **Add Visualization**: Plot accuracy over time, learning curves

## Technical Notes

- Evaluation results are stored in versioned CSV files
- Metadata stored in `output/eval_metadata.json`
- Frontend polls API for latest results (can add WebSocket for real-time updates)
- Confusion matrix uses color intensity to show prediction density
- Comparison view shows all models in a single table for easy comparison
