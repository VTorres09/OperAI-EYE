from pathlib import Path
from threading import Lock, Thread
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from hf_dataset import DatasetPreparationError, dataset_status, prepare_hf_dataset

from .data import get_filter_options, get_image_path, get_images, get_stats, reset_cache
from .eval_data import (
    compare_models,
    get_eval_filter_options,
    get_eval_results,
    list_models,
    register_model,
)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_dataset_download_lock = Lock()
_dataset_download_state = {
    "state": "idle",
    "message": "",
    "error": None,
}


def _set_dataset_download_state(state: str, message: str = "", error: str | None = None) -> None:
    with _dataset_download_lock:
        _dataset_download_state.update({"state": state, "message": message, "error": error})


def _get_dataset_download_state() -> dict:
    with _dataset_download_lock:
        return dict(_dataset_download_state)


def _download_dataset_job() -> None:
    _set_dataset_download_state("running", "Downloading dataset from Hugging Face")
    try:
        status = prepare_hf_dataset()
    except DatasetPreparationError as exc:
        _set_dataset_download_state("error", "Dataset download failed", str(exc))
        return
    except Exception as exc:
        _set_dataset_download_state("error", "Dataset download failed", str(exc))
        return

    reset_cache()
    _set_dataset_download_state(
        "complete",
        f"Dataset ready with {status['image_count']} images",
    )


@app.get("/api/dataset/status")
def get_dataset_status():
    status = dataset_status()
    status["download"] = _get_dataset_download_state()
    return status


@app.post("/api/dataset/download")
def download_dataset():
    status = dataset_status()
    download = _get_dataset_download_state()
    if status["ready"]:
        return {**status, "download": download}
    if download["state"] == "running":
        return {**status, "download": download}

    _set_dataset_download_state("running", "Starting dataset download")
    Thread(target=_download_dataset_job, daemon=True).start()
    status = dataset_status()
    status["download"] = _get_dataset_download_state()
    return status


@app.get("/images/{image_path:path}", include_in_schema=False)
def serve_image(image_path: str):
    requested_path = Path(image_path)
    if requested_path.is_absolute() or ".." in requested_path.parts:
        raise HTTPException(status_code=404, detail="Image not found")

    file_path = get_image_path(requested_path)
    if file_path is None:
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(file_path)


@app.get("/api/images")
def list_images(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    phase: Optional[str] = None,
    surgery_type: Optional[str] = None,
    camera: Optional[str] = None,
    procedure_id: Optional[int] = None,
    take_id: Optional[int] = None,
    model_id: Optional[str] = None,
):
    return get_images(
        page=page,
        page_size=page_size,
        phase=phase,
        surgery_type=surgery_type,
        camera=camera,
        procedure_id=procedure_id,
        take_id=take_id,
        model_id=model_id,
    )


@app.get("/api/filters")
def filters():
    return get_filter_options()


@app.get("/api/stats")
def stats(
    phase: Optional[str] = None,
    surgery_type: Optional[str] = None,
    camera: Optional[str] = None,
    procedure_id: Optional[int] = None,
    take_id: Optional[int] = None,
):
    return get_stats(
        phase=phase,
        surgery_type=surgery_type,
        camera=camera,
        procedure_id=procedure_id,
        take_id=take_id,
    )


@app.get("/api/eval/models")
def eval_list_models():
    return list_models()


@app.get("/api/eval/filters")
def eval_filters():
    return get_eval_filter_options()


@app.get("/api/eval/results/{model_id}")
def eval_results(
    model_id: str,
    phase: Optional[str] = None,
    surgery_type: Optional[str] = None,
    camera: Optional[str] = None,
    procedure_id: Optional[int] = None,
    take_id: Optional[int] = None,
    merge_patient_and_surgery: bool = False,
):
    result = get_eval_results(
        model_id,
        phase=phase,
        surgery_type=surgery_type,
        camera=camera,
        procedure_id=procedure_id,
        take_id=take_id,
        merge_patient_and_surgery=merge_patient_and_surgery,
    )
    if not result:
        raise HTTPException(status_code=404, detail="Model not found")
    return result


@app.get("/api/eval/compare")
def eval_compare(
    model_ids: str = Query(..., description="Comma-separated model IDs"),
    phase: Optional[str] = None,
    surgery_type: Optional[str] = None,
    camera: Optional[str] = None,
    procedure_id: Optional[int] = None,
    take_id: Optional[int] = None,
    merge_patient_and_surgery: bool = False,
):
    ids = [mid.strip() for mid in model_ids.split(",")]
    result = compare_models(
        ids,
        phase=phase,
        surgery_type=surgery_type,
        camera=camera,
        procedure_id=procedure_id,
        take_id=take_id,
        merge_patient_and_surgery=merge_patient_and_surgery,
    )
    if not result:
        raise HTTPException(status_code=404, detail="No models found")
    return result


@app.post("/api/eval/register")
def eval_register(
    model_id: str,
    model_name: str,
    prompt_file: str,
    description: str = "",
):
    return register_model(model_id, model_name, prompt_file, description)


if STATIC_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=str(STATIC_DIR / "assets")), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        if full_path.startswith("api"):
            raise HTTPException(status_code=404, detail="Not found")
        file_path = STATIC_DIR / full_path
        if full_path and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(STATIC_DIR / "index.html")
