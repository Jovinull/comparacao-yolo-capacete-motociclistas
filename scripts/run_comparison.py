#!/usr/bin/env python3
"""Train YOLOv8n and YOLO11n under one config and evaluate the held-out test set."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import platform
import random
import shutil
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / ".cache" / "ultralytics"))
os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / ".cache" / "torch"))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import yaml
import psutil

from evaluation import calculate_detection_metrics, collect_detections, select_confidence_threshold
from validate_dataset import ManifestRecord, validate_dataset


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
            "precision_at_ultralytics_best_f1": precision_by_class.get(class_id),
            "recall_at_ultralytics_best_f1": recall_by_class.get(class_id),
        }

    speed_raw = getattr(metrics, "speed", {}) or {}
    speed = {str(key): float(value) for key, value in speed_raw.items()} if hasattr(speed_raw, "items") else {}
    inference_ms = speed.get("inference", 0.0)
    total_speed = sum(speed.get(key, 0.0) for key in ("preprocess", "inference", "postprocess"))
    return {
        "map_50_95": float(box.map),
        "map_50": float(box.map50),
        "precision_at_ultralytics_best_f1": float(box.mp),
        "recall_at_ultralytics_best_f1": float(box.mr),
        "speed_ms_per_image": speed,
        "detector_latency_ms_per_image": inference_ms or None,
        "detector_fps": 1000.0 / inference_ms if inference_ms > 0 else None,
        "pipeline_fps": 1000.0 / total_speed if total_speed > 0 else None,
        "per_class": per_class,
    }


class PeakRssMonitor:
    """Sample peak resident memory for the current process during one stage."""

    def __init__(self, interval_seconds: float = 0.05):
        self.process = psutil.Process()
        self.interval_seconds = interval_seconds
        self.peak_bytes = self.process.memory_info().rss
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.is_set():
            try:
                self.peak_bytes = max(self.peak_bytes, self.process.memory_info().rss)
            except psutil.Error:
                pass
            self._stop.wait(self.interval_seconds)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._stop.set()
        self._thread.join()


def _reset_cuda_peak(torch) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def _cuda_peak_memory_mb(torch) -> dict[str, float | None]:
    if not torch.cuda.is_available():
        return {"allocated_mb": None, "reserved_mb": None}
    torch.cuda.synchronize()
    return {
        "allocated_mb": torch.cuda.max_memory_allocated() / (1024**2),
        "reserved_mb": torch.cuda.max_memory_reserved() / (1024**2),
    }


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
            "note": "São necessários ao menos dois vídeos com objetos da classe alvo para estimar o intervalo.",
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
        "interpretation": "Diferença pareada de revocação média por vídeo: segundo modelo menos primeiro modelo.",
    }


def _runtime_info(torch, ultralytics) -> dict:
    cuda_available = bool(torch.cuda.is_available())
    return {
        "python": sys.version.split()[0],
        "operating_system": platform.platform(),
        "processor": platform.processor(),
        "host_memory_gb": psutil.virtual_memory().total / (1024**3),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "cuda_version": torch.version.cuda,
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
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        parser.error(f"Não foi possível ler a configuração: {exc}")
    if not isinstance(config, dict):
        parser.error("A configuração precisa ser um objeto YAML na raiz")
    if args.device:
        config.setdefault("train", {})["device"] = args.device
    train_config = config.get("train", {})
    evaluation_config = config.get("evaluation", {})
    threshold_config = config.get("threshold_selection", {})
    paired_config = config.get("paired_analysis", {})
    model_specs = config.get("models", [])
    if not isinstance(model_specs, list) or len(model_specs) != 2:
        parser.error("A configuração precisa listar exatamente dois modelos")
    model_names_in_config = [str(spec.get("name", "")) for spec in model_specs if isinstance(spec, dict)]
    if len(model_names_in_config) != 2 or not all(model_names_in_config) or len(set(model_names_in_config)) != 2:
        parser.error("Cada modelo precisa ter um nome único")

    confidence_floor = float(threshold_config.get("confidence_floor", 0.001))
    confidence_grid = [float(value) for value in threshold_config.get("confidence_grid", [])]
    match_iou = float(threshold_config.get("match_iou", 0.5))
    eval_imgsz = int(evaluation_config.get("imgsz", train_config.get("imgsz", 640)))
    eval_batch = int(evaluation_config.get("batch", train_config.get("batch", 1)))
    nms_iou = float(evaluation_config.get("iou", 0.7))
    max_det = int(evaluation_config.get("max_det", 300))
    target_class_name = str(paired_config.get("class_name", "motociclista_sem_capacete"))
    if not confidence_grid:
        parser.error("threshold_selection.confidence_grid não pode ficar vazia")
    if not 0 < confidence_floor <= min(confidence_grid) <= max(confidence_grid) <= 1:
        parser.error("A grade de limiares deve ficar entre confidence_floor e 1")
    if not 0 < match_iou <= 1 or not 0 < nms_iou <= 1:
        parser.error("Os limiares IoU precisam estar entre 0 e 1")
    if eval_imgsz < 1 or eval_batch < 1 or max_det < 1:
        parser.error("imgsz, batch e max_det precisam ser positivos")
    if str(evaluation_config.get("split", "test")) != "test":
        parser.error("evaluation.split deve ser test; o teste reservado é consultado só após escolher os limiares")

    data_yaml = args.data or Path(config.get("data_yaml", ""))
    if not data_yaml.is_absolute():
        data_yaml = PROJECT_ROOT / data_yaml
    try:
        dataset = validate_dataset(data_yaml)
    except ValueError as exc:
        parser.error(str(exc))
    if target_class_name not in dataset.names:
        parser.error(f"Classe para análise pareada não encontrada: {target_class_name}")

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
    archived_data_config = yaml.safe_load(data_yaml.resolve().read_text(encoding="utf-8")) or {}
    archived_data_config["path"] = str(dataset.root)
    (run_root / "data.yaml").write_text(
        yaml.safe_dump(archived_data_config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
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
        _reset_cuda_peak(torch)
        with PeakRssMonitor() as host_memory:
            training_started = time.perf_counter()
            model.train(**train_args)
            training_seconds = time.perf_counter() - training_started
        training_gpu_memory = _cuda_peak_memory_mb(torch)
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
            "training_cost": {
                "wall_time_seconds": training_seconds,
                "peak_host_rss_mb": host_memory.peak_bytes / (1024**2),
                "peak_gpu_memory": training_gpu_memory,
                "best_weights_size_mb": candidate.stat().st_size / (1024**2),
            },
        }
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    model_names = {index: name for index, name in enumerate(dataset.names)}
    validation_records = [record for record in dataset.records if record.split == "val"]
    test_records = [record for record in dataset.records if record.split == "test"]

    # Escolha de limiar usa apenas validação; todos os modelos são congelados antes do teste.
    for model_spec in model_specs:
        model_name = str(model_spec["name"])
        print(f"\n=== Escolhendo limiar de {model_name} na validação ===")
        model = _load_model(str(best_weights[model_name]), weights_dir)
        validation_predictions = collect_detections(
            model=model,
            records=validation_records,
            num_classes=len(dataset.names),
            confidence_floor=confidence_floor,
            prediction_iou=nms_iou,
            imgsz=eval_imgsz,
            batch=eval_batch,
            device=device,
            max_det=max_det,
        )
        selected_threshold, threshold_scores = select_confidence_threshold(
            validation_predictions,
            dataset.names,
            confidence_grid,
            match_iou,
        )
        results[model_name]["threshold_selection"] = {
            "split": "val",
            "method": "highest threshold maximizing macro-F1 over the configured grid",
            "selected_confidence": selected_threshold,
            "match_iou": match_iou,
            "grid_scores": threshold_scores,
        }
        del validation_predictions, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for model_spec in model_specs:
        model_name = str(model_spec["name"])
        print(f"\n=== Avaliando {model_name} no conjunto de teste ===")
        model = _load_model(str(best_weights[model_name]), weights_dir)
        _reset_cuda_peak(torch)
        with PeakRssMonitor() as host_memory:
            metrics = model.val(
                data=str(data_yaml.resolve()),
                split=str(evaluation_config.get("split", "test")),
                imgsz=eval_imgsz,
                batch=eval_batch,
                device=device,
                conf=float(evaluation_config.get("conf", confidence_floor)),
                iou=nms_iou,
                max_det=max_det,
                plots=bool(evaluation_config.get("plots", True)),
                project=str(run_root),
                name=f"{model_name}_test",
                exist_ok=False,
                verbose=True,
            )
            test_predictions = collect_detections(
                model=model,
                records=test_records,
                num_classes=len(dataset.names),
                confidence_floor=confidence_floor,
                prediction_iou=nms_iou,
                imgsz=eval_imgsz,
                batch=eval_batch,
                device=device,
                max_det=max_det,
            )
            operating_point = calculate_detection_metrics(
                test_predictions,
                dataset.names,
                results[model_name]["threshold_selection"]["selected_confidence"],
                match_iou,
            )
            condition_metrics = {}
            conditions_in_test = sorted(
                {condition for frame in test_predictions for condition in frame.record.conditions}
            )
            for condition in conditions_in_test:
                condition_frames = [
                    frame for frame in test_predictions if condition in frame.record.conditions
                ]
                condition_metrics[condition] = calculate_detection_metrics(
                    condition_frames,
                    dataset.names,
                    results[model_name]["threshold_selection"]["selected_confidence"],
                    match_iou,
                )

        metric_record = _metric_summary(metrics, model_names)
        metric_record["operating_point"] = operating_point
        metric_record["by_condition"] = condition_metrics
        results[model_name]["test_metrics"] = metric_record
        results[model_name]["test_cost"] = {
            "peak_host_rss_mb": host_memory.peak_bytes / (1024**2),
            "peak_gpu_memory": _cuda_peak_memory_mb(torch),
        }
        results[model_name]["test_run"] = str(getattr(metrics, "save_dir", ""))
        target_video_recall = [
            {"video_id": video_id, "recall": class_metrics[target_class_name]["recall"]}
            for video_id, class_metrics in operating_point["per_video"].items()
        ]
        results[model_name]["per_video_target_recall"] = target_video_recall
        del test_predictions, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    first_name, second_name = (str(spec["name"]) for spec in model_specs)
    paired = _paired_video_bootstrap(
        results[first_name]["per_video_target_recall"],
        results[second_name]["per_video_target_recall"],
        samples=int(paired_config.get("bootstrap_samples", 10_000)),
        seed=int(paired_config.get("seed", 42)),
    )
    results_summary = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": runtime,
        "device_used": device,
        "dataset": dataset.report,
        "paired_target_recall_by_video": {
            "first_model": first_name,
            "second_model": second_name,
            "target_class": target_class_name,
            **paired,
        },
        "models": results,
    }
    (run_root / "summary.json").write_text(
        json.dumps(results_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (run_root / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = [
            "model", "map_50_95", "map_50", "precision", "recall", "macro_f1",
            "confidence_threshold", "detector_latency_ms", "detector_fps", "pipeline_fps",
            "training_seconds", "best_weights_size_mb", "training_peak_host_rss_mb",
            "training_peak_gpu_allocated_mb", "test_peak_host_rss_mb", "test_peak_gpu_allocated_mb",
            "target_class_recall",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for model_name, model_result in results.items():
            test_metrics = model_result["test_metrics"]
            training_cost = model_result["training_cost"]
            test_cost = model_result["test_cost"]
            target_recall = test_metrics["operating_point"]["per_class"][target_class_name]["recall"]
            writer.writerow(
                {
                    "model": model_name,
                    "map_50_95": test_metrics["map_50_95"],
                    "map_50": test_metrics["map_50"],
                    "precision": test_metrics["operating_point"]["macro_precision"],
                    "recall": test_metrics["operating_point"]["macro_recall"],
                    "macro_f1": test_metrics["operating_point"]["macro_f1"],
                    "confidence_threshold": model_result["threshold_selection"]["selected_confidence"],
                    "detector_latency_ms": test_metrics["detector_latency_ms_per_image"],
                    "detector_fps": test_metrics["detector_fps"],
                    "pipeline_fps": test_metrics["pipeline_fps"],
                    "training_seconds": training_cost["wall_time_seconds"],
                    "best_weights_size_mb": training_cost["best_weights_size_mb"],
                    "training_peak_host_rss_mb": training_cost["peak_host_rss_mb"],
                    "training_peak_gpu_allocated_mb": training_cost["peak_gpu_memory"]["allocated_mb"],
                    "test_peak_host_rss_mb": test_cost["peak_host_rss_mb"],
                    "test_peak_gpu_allocated_mb": test_cost["peak_gpu_memory"]["allocated_mb"],
                    "target_class_recall": target_recall,
                }
            )

    with (run_root / "per_class.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = [
            "model", "class_name", "class_id", "map_50_95", "map_50", "true_positives", "false_positives",
            "false_negatives", "ground_truth_objects", "precision", "recall", "f1",
            "false_positives_per_100_frames", "false_negatives_per_100_objects", "confidence_threshold",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for model_name, model_result in results.items():
            test_metrics = model_result["test_metrics"]
            operating_classes = test_metrics["operating_point"]["per_class"]
            curve_classes = test_metrics["per_class"]
            for class_name, operating_class in operating_classes.items():
                writer.writerow(
                    {
                        "model": model_name,
                        "class_name": class_name,
                        "map_50_95": curve_classes[class_name]["map_50_95"],
                        "map_50": curve_classes[class_name]["map_50"],
                        **operating_class,
                        "confidence_threshold": model_result["threshold_selection"]["selected_confidence"],
                    }
                )

    print(f"\nComparação concluída. Relatórios: {run_root}")
    print(f"Bootstrap pareado de recall de {target_class_name} por vídeo: {paired}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
