#!/usr/bin/env python3
"""Check that both pretrained models load and infer on a synthetic CPU frame."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights"
WEIGHTS.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache" / "ultralytics"))
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))

import numpy as np
import torch
import ultralytics
from ultralytics import YOLO


def main() -> int:
    print(f"Python: {__import__('sys').version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Ultralytics: {ultralytics.__version__}")
    print(f"CUDA disponível: {torch.cuda.is_available()}")
    sample = np.zeros((320, 320, 3), dtype=np.uint8)

    previous_directory = Path.cwd()
    try:
        os.chdir(WEIGHTS)
        for model_name in ("yolov8n.pt", "yolo11n.pt"):
            model = YOLO(model_name)
            result = model.predict(
                source=sample,
                imgsz=320,
                device="cpu",
                verbose=False,
                save=False,
            )[0]
            print(f"{model_name}: inferência OK; caixas={len(result.boxes)}; classes={len(model.names)}")
    finally:
        os.chdir(previous_directory)

    print("Smoke test concluído. A imagem sintética não é dado nem resultado experimental.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
