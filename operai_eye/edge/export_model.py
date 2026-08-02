"""Export the fine-tuned LightlyTrain DINOv3 checkpoint to edge ONNX."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_REPO_ID = "OperAI-Research/operai-eye-dinov3-vitb16-exocentric-rgb"
DEFAULT_FILENAME = "exported_best.pt"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--filename", default=DEFAULT_FILENAME)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/operai_eye_dinov3.onnx"),
    )
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Also create a dynamic INT8 model next to the float model.",
    )
    return parser.parse_args(argv)


def resolve_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint:
        return args.checkpoint
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=args.repo_id,
            filename=args.filename,
            token=True,
        )
    )


def export_onnx(checkpoint_path: Path, output_path: Path, *, opset: int) -> None:
    try:
        import torch
        from torch import nn
    except ModuleNotFoundError as exc:
        raise RuntimeError("torch is required to export the edge model") from exc
    try:
        import lightly_train  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "lightly-train==0.15.1 is required to read the training export"
        ) from exc

    from operai_eye.training.dinov3.evaluate_lightly import load_model

    class LogitsOnly(nn.Module):
        def __init__(self, task_model: nn.Module) -> None:
            super().__init__()
            self.task_model = task_model

        def forward(self, image: torch.Tensor) -> torch.Tensor:
            return self.task_model.forward_backend(image)

    model = LogitsOnly(load_model(checkpoint_path, torch.device("cpu"))).eval()
    example = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            model,
            example,
            output_path,
            input_names=["images"],
            output_names=["logits"],
            dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=opset,
            do_constant_folding=True,
        )

    import onnx

    graph = onnx.load(output_path)
    onnx.checker.check_model(graph)
    verification_batch = torch.zeros((2, 3, 224, 224), dtype=torch.float32)
    with torch.inference_mode():
        expected = model(verification_batch).detach().numpy()
    actual = _run_onnx(output_path, verification_batch.numpy())
    import numpy as np

    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)


def quantize_model(source: Path) -> Path:
    import numpy as np
    from onnxruntime.quantization import QuantType, quantize_dynamic

    destination = source.with_name(f"{source.stem}.int8{source.suffix}")
    quantize_dynamic(
        model_input=source,
        model_output=destination,
        weight_type=QuantType.QInt8,
    )
    output = _run_onnx(
        destination,
        np.zeros((2, 3, 224, 224), dtype=np.float32),
    )
    if output.shape != (2, 4):
        raise RuntimeError(
            f"Unexpected quantized model output shape: {output.shape}; expected (2, 4)"
        )
    return destination


def _run_onnx(model_path: Path, images: Any) -> Any:
    import onnxruntime as ort

    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )
    return session.run(["logits"], {"images": images})[0]


def write_metadata(model_path: Path, checkpoint_path: Path) -> Path:
    metadata = {
        "format_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "model_family": "DINOv3 ViT-B/16",
        "source_checkpoint": checkpoint_path.name,
        "input_name": "images",
        "output_name": "logits",
        "image_size": 224,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "class_names": [
            "idle",
            "people_in_room",
            "surgery_inactive",
            "surgery_active",
        ],
        "phase_mapping": {
            "IDLE": "idle",
            "PATIENT_IN_ROOM": "people_in_room * surgery_inactive",
            "SURGERY_ACTIVE": "people_in_room * surgery_active",
        },
        "sha256": _sha256(model_path),
    }
    path = model_path.with_suffix(".json")
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    checkpoint = resolve_checkpoint(args)
    export_onnx(checkpoint, args.output, opset=args.opset)
    models = [args.output]
    if args.quantize:
        models.append(quantize_model(args.output))
    for model in models:
        metadata = write_metadata(model, checkpoint)
        print(f"model={model}")
        print(f"metadata={metadata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
