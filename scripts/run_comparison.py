#!/usr/bin/env python3
"""Train YOLOv8n and YOLO11n under one config and evaluate the held-out test set."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / ".cache" / "ultralytics"))
os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / ".cache" / "torch"))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import yaml

from validate_dataset import DatasetInfo, ManifestRecord, validate_dataset


def _load_model(weights_name: str, weights_dir: Path):
    from ultralytics import YOLO

    previous_directory = Path.cwd()
    try:
        os.chdir(weights_dir)
        return YOLO(weights_name)
    finally:
        os.chdir(previous_directory)


def _python_values(value) -> list[float]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        value = [value]
    return [float(item) for item in value]


def _metric_summary(metrics, model_names: dict[int, str]) -> dict:
    box = metrics.box
    maps = _python_values(getattr(box, "maps", None))
    map50_values = _python_values(getattr(box, "ap50", None))
    class_ids = [int(value) for value in _python_values(getattr(box, "ap_class_index", None))]
    precision_values = _python_values(getattr(box, "p", None))
    recall_values = _python_values(getattr(box, "r", None))
    precision_by_class = dict(zip(class_ids, precision_values))
    recall_by_class = dict(zip(class_ids, recall_values))

    per_class = {}
    for class_id, class_name in model_names.items():
        per_class[class_name] = {
            "class_id": class_id,
            "map_50_95": maps[class_id] if class_id < len(maps) else None,
            "map_50": map50_values[class_ids.index(class_id)] if class_id in class_ids and class_ids.index(class_id) < len(map50_values) else None,
            "precision": precision_by_class.get(class_id),
            "recall": recall_by_class.get(class_id),
        }

    speed_raw = getattr(metrics, "speed", {}) or {}
    speed = {str(key): float(value) for key, value in speed_raw.items()} if hasattr(speed_raw, "items") else {}
    total_speed = sum(speed.get(key, 0.0) for key in ("preprocess", "inference", "postprocess"))
    return {
        "map_50_95": float(box.map),
        "map_50": float(box.map50),
        "precision": float(box.mp),
        "recall": float(box.mr),
        "speed_ms_per_image": speed,
        "pipeline_fps_from_validation_speed": 1000.0 / total_speed if total_speed > 0 else None,
        "per_class": per_class,
    }


def _box_iou_xyxy(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_first = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_second = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = area_first + area_second - intersection
    return intersection / union if union > 0 else 0.0


def _read_class_boxes(label_path: Path, class_id: int, image_width: int, image_height: int):
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 5 or int(fields[0]) != class_id:
            continue
        x_center, y_center, width, height = (float(value) for value in fields[1:])
        boxes.append(
            (
                (x_center - width / 2) * image_width,
                (y_center - height / 2) * image_height,
                (x_center + width / 2) * image_width,
                (y_center + height / 2) * image_height,
            )
        )
    return boxes


def _count_video_recall(
    model,
    records: list[ManifestRecord],
    class_id: int,
    confidence: float,
    match_iou: float,
    prediction_iou: float,
    imgsz: int,
    batch: int,
    device: str,
) -> list[dict]:
    by_video: dict[str, list[ManifestRecord]] = defaultdict(list)
    for record in records:
        if record.split == "test":
            by_video[record.video_id].append(record)

    report = []
    for video_id in sorted(by_video):
        video_records = by_video[video_id]
        true_positives = 0
        ground_truth_count = 0
        for start in range(0, len(video_records), batch):
            chunk = video_records[start : start + batch]
            predictions = model.predict(
                source=[str(record.image_path) for record in chunk],
                imgsz=imgsz,
                batch=batch,
                device=device,
                conf=confidence,
                iou=prediction_iou,
                verbose=False,
                save=False,
            )
            for record, result in zip(chunk, predictions):
                image_height, image_width = result.orig_shape
                ground_truth = _read_class_boxes(
                    record.label_path, class_id, image_width, image_height
                )
                ground_truth_count += len(ground_truth)
                if not ground_truth or result.boxes is None or len(result.boxes) == 0:
                    continue

                predicted_boxes = result.boxes.xyxy.cpu().tolist()
                predicted_classes = result.boxes.cls.cpu().tolist()
                predicted_scores = result.boxes.conf.cpu().tolist()
                candidates = [
                    (float(predicted_scores[index]), tuple(float(v) for v in predicted_boxes[index]))
                    for index, predicted_class in enumerate(predicted_classes)
                    if int(predicted_class) == class_id and float(predicted_scores[index]) >= confidence
                ]
                candidates.sort(key=lambda item: item[0], reverse=True)
                matched: set[int] = set()
                for _, predicted_box in candidates:
                    best_index = None
                    best_iou = 0.0
                    for ground_truth_index, truth_box in enumerate(ground_truth):
                        if ground_truth_index in matched:
                            continue
                        overlap = _box_iou_xyxy(predicted_box, truth_box)
                        if overlap > best_iou:
                            best_index, best_iou = ground_truth_index, overlap
                    if best_index is not None and best_iou >= match_iou:
                        matched.add(best_index)
                true_positives += len(matched)

        report.append(
            {
                "video_id": video_id,
                "class_id": class_id,
                "ground_truth_objects": ground_truth_count,
                "true_positives": true_positives,
                "recall": true_positives / ground_truth_count if ground_truth_count else None,
            }
        )
    return report


def _percentile(sorted_values: list[float], quantile: float) -> float:
    position = (len(sorted_values) - 1) * quantile
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return sorted_values[low]
    fraction = position - low
    return sorted_values[low] * (1 - fraction) + sorted_values[high] * fraction


def _paired_video_bootstrap(first: list[dict], second: list[dict], samples: int, seed: int) -> dict:
    first_by_video = {row["video_id"]: row for row in first}
    second_by_video = {row["video_id"]: row for row in second}
    video_ids = sorted(
        video_id
        for video_id in first_by_video.keys() & second_by_video.keys()
        if first_by_video[video_id]["recall"] is not None
        and second_by_video[video_id]["recall"] is not None
    )
    if not video_ids:
        return {"videos": 0, "mean_recall_difference": None, "ci95": None}

    differences = [
        second_by_video[video_id]["recall"] - first_by_video[video_id]["recall"]
        for video_id in video_ids
    ]
    mean_difference = sum(differences) / len(differences)
    if len(video_ids) < 2 or samples < 1:
        return {
            "videos": len(video_ids),
            "mean_recall_difference": mean_difference,
            "ci95": None,
            "note": "São necessários ao menos dois vídeos com objetos sem capacete para estimar o intervalo.",
        }

    rng = random.Random(seed)
    boot_means = sorted(
        sum(rng.choice(differences) for _ in differences) / len(differences)
        for _ in range(samples)
    )
    return {
        "videos": len(video_ids),
        "mean_recall_difference": mean_difference,
        "ci95": [_percentile(boot_means, 0.025), _percentile(boot_means, 0.975)],
        "interpretation": "Diferença pareada de recall médio por vídeo: segundo modelo menos primeiro modelo.",
    }


def _runtime_info(torch, ultralytics) -> dict:
    cuda_available = bool(torch.cuda.is_available())
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "cuda_available": cuda_available,
        "gpu": torch.cuda.get_device_name(0) if cuda_available else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "experiment.yaml")
    parser.add_argument("--data", type=Path, help="Sobrescreve data_yaml da configuração")
    parser.add_argument("--device", help="Sobrescreve o dispositivo (ex.: 0 ou cpu)")
    args = parser.parse_args()

    config_path = args.config.resolve()
    if not config_path.is_file():
        parser.error(f"Configuração não encontrada: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if args.device:
        config.setdefault("train", {})["device"] = args.device
    train_config = config.get("train", {})
    evaluation_config = config.get("evaluation", {})
    paired_config = config.get("paired_analysis", {})
    model_specs = config.get("models", [])
    if len(model_specs) != 2:
        parser.error("A configuração precisa listar exatamente dois modelos")

    data_yaml = args.data or Path(config.get("data_yaml", ""))
    if not data_yaml.is_absolute():
        data_yaml = PROJECT_ROOT / data_yaml
    try:
        dataset = validate_dataset(data_yaml)
    except ValueError as exc:
        parser.error(str(exc))

    import torch
    import ultralytics

    device = str(train_config.get("device", "0"))
    if device.lower() != "cpu" and not torch.cuda.is_available():
        parser.error("A configuração pede GPU, mas CUDA não está disponível. Execute em um runtime NVIDIA.")

    output_base = Path(config.get("output_dir", "runs"))
    if not output_base.is_absolute():
        output_base = PROJECT_ROOT / output_base
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = output_base / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, run_root / "experiment.yaml")
    (run_root / "dataset_validation.json").write_text(
        json.dumps(dataset.report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    runtime = _runtime_info(torch, ultralytics)
    results: dict[str, dict] = {}
    best_weights: dict[str, Path] = {}
    weights_dir = PROJECT_ROOT / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    # Treine os dois modelos antes de consultar o conjunto reservado de teste.
    for model_spec in model_specs:
        model_name = str(model_spec["name"])
        weights_name = str(model_spec["weights"])
        print(f"\n=== Treinando {model_name} ===")
        model = _load_model(weights_name, weights_dir)
        train_args = dict(train_config)
        train_args.update(
            {
                "data": str(data_yaml.resolve()),
                "project": str(run_root),
                "name": f"{model_name}_train",
                "exist_ok": False,
                "save": True,
                "val": True,
            }
        )
        model.train(**train_args)
        trainer = getattr(model, "trainer", None)
        candidate = Path(trainer.best) if trainer and getattr(trainer, "best", None) else Path(trainer.save_dir) / "weights" / "best.pt"
        candidate = candidate.resolve()
        if not candidate.is_file():
            raise RuntimeError(f"Não encontrei o melhor checkpoint de {model_name}: {candidate}")
        best_weights[model_name] = candidate
        results[model_name] = {
            "initial_weights": weights_name,
            "best_weights": str(candidate),
            "training_run": str(Path(trainer.save_dir).resolve()),
        }
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    model_names = {index: name for index, name in enumerate(dataset.names)}
    test_records = [record for record in dataset.records if record.split == "test"]
    for model_spec in model_specs:
        model_name = str(model_spec["name"])
        print(f"\n=== Avaliando {model_name} no conjunto de teste ===")
        model = _load_model(str(best_weights[model_name]), weights_dir)
        metrics = model.val(
            data=str(data_yaml.resolve()),
            split=str(evaluation_config.get("split", "test")),
            imgsz=int(evaluation_config.get("imgsz", train_config.get("imgsz", 640))),
            batch=int(evaluation_config.get("batch", train_config.get("batch", 16))),
            device=device,
            conf=float(evaluation_config.get("conf", 0.001)),
            iou=float(evaluation_config.get("iou", 0.7)),
            max_det=int(evaluation_config.get("max_det", 300)),
            plots=bool(evaluation_config.get("plots", True)),
            project=str(run_root),
            name=f"{model_name}_test",
            exist_ok=False,
            verbose=True,
        )
        metric_record = _metric_summary(metrics, model_names)
        video_recall = _count_video_recall(
            model=model,
            records=test_records,
            class_id=int(paired_config.get("class_id", 1)),
            confidence=float(paired_config.get("confidence", 0.25)),
            match_iou=float(paired_config.get("iou_match", 0.5)),
            prediction_iou=float(evaluation_config.get("iou", 0.7)),
            imgsz=int(evaluation_config.get("imgsz", train_config.get("imgsz", 640))),
            batch=int(evaluation_config.get("batch", train_config.get("batch", 16))),
            device=device,
        )
        results[model_name]["test_metrics"] = metric_record
        results[model_name]["per_video_no_helmet_recall"] = video_recall
        results[model_name]["test_run"] = str(getattr(metrics, "save_dir", ""))
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    first_name, second_name = (str(spec["name"]) for spec in model_specs)
    paired = _paired_video_bootstrap(
        results[first_name]["per_video_no_helmet_recall"],
        results[second_name]["per_video_no_helmet_recall"],
        samples=int(paired_config.get("bootstrap_samples", 10_000)),
        seed=int(paired_config.get("seed", 42)),
    )
    results_summary = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": runtime,
        "device_used": device,
        "dataset": dataset.report,
        "paired_no_helmet_recall_by_video": {
            "first_model": first_name,
            "second_model": second_name,
            "confidence": float(paired_config.get("confidence", 0.25)),
            "iou_match": float(paired_config.get("iou_match", 0.5)),
            **paired,
        },
        "models": results,
    }
    (run_root / "summary.json").write_text(
        json.dumps(results_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (run_root / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ["model", "map_50_95", "map_50", "precision", "recall", "fps_from_validation_speed", "no_helmet_recall"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for model_name, model_result in results.items():
            test_metrics = model_result["test_metrics"]
            no_helmet = test_metrics["per_class"].get("sem_capacete", {})
            writer.writerow(
                {
                    "model": model_name,
                    "map_50_95": test_metrics["map_50_95"],
                    "map_50": test_metrics["map_50"],
                    "precision": test_metrics["precision"],
                    "recall": test_metrics["recall"],
                    "fps_from_validation_speed": test_metrics["pipeline_fps_from_validation_speed"],
                    "no_helmet_recall": no_helmet.get("recall"),
                }
            )

    print(f"\nComparação concluída. Relatórios: {run_root}")
    print(f"Bootstrap pareado de recall sem capacete por vídeo: {paired}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
