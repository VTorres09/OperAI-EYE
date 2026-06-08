from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .data import DATA_DIR, get_filter_options, get_images, get_stats
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

if DATA_DIR.exists():
    app.mount("/images", StaticFiles(directory=str(DATA_DIR)), name="images")


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
):
    result = get_eval_results(
        model_id,
        phase=phase,
        surgery_type=surgery_type,
        camera=camera,
        procedure_id=procedure_id,
        take_id=take_id,
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
):
    ids = [mid.strip() for mid in model_ids.split(",")]
    result = compare_models(
        ids,
        phase=phase,
        surgery_type=surgery_type,
        camera=camera,
        procedure_id=procedure_id,
        take_id=take_id,
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
    labels_file: str = "test_labels.csv",
):
    return register_model(model_id, model_name, prompt_file, description, labels_file)


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
