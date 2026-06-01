from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .data import DATA_DIR, get_filter_options, get_images, get_stats

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
    confidence_min: Optional[float] = None,
    confidence_max: Optional[float] = None,
):
    return get_images(
        page=page,
        page_size=page_size,
        phase=phase,
        surgery_type=surgery_type,
        camera=camera,
        procedure_id=procedure_id,
        take_id=take_id,
        confidence_min=confidence_min,
        confidence_max=confidence_max,
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
    confidence_min: Optional[float] = None,
    confidence_max: Optional[float] = None,
):
    return get_stats(
        phase=phase,
        surgery_type=surgery_type,
        camera=camera,
        procedure_id=procedure_id,
        take_id=take_id,
        confidence_min=confidence_min,
        confidence_max=confidence_max,
    )


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
