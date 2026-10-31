#!/usr/bin/env python3
"""Run a time-bounded, CPU-only YOLO comparison on EdgeVision images."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _number_list(value) -> list[float]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (int, float)):
        return [float(value)]
    return [float(item) for item in value]


def _as_plain_metrics(metrics, class_names: list[str]) -> dict:
    output = {}
    for key, value in (getattr(metrics, "results_dict", {}) or {}).items():
        try:
            output[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    box = getattr(metrics, "box", None)
    if box is not None:
        for key in ("map", "map50", "map75", "mp", "mr"):
            value = getattr(box, key, None)
            if value is not None:
                output[key] = float(value)
        class_ids = [int(value) for value in _number_list(getattr(box, "ap_class_index", None))]
        positions = {class_id: index for index, class_id in enumerate(class_ids)}
        precisions = _number_list(getattr(box, "p", None))
        recalls = _number_list(getattr(box, "r", None))
        f1_scores = _number_list(getattr(box, "f1", None))
        ap50_values = _number_list(getattr(box, "ap50", None))
        maps = _number_list(getattr(box, "maps", None))
        output["per_class"] = {}
        for class_id, class_name in enumerate(class_names):
            position = positions.get(class_id)
            output["per_class"][class_name] = {
                "class_id": class_id,
                "precision_at_best_f1": precisions[position] if position is not None and position < len(precisions) else None,
                "recall_at_best_f1": recalls[position] if position is not None and position < len(recalls) else None,
                "f1_at_best_f1": f1_scores[position] if position is not None and position < len(f1_scores) else None,
                "map50": ap50_values[position] if position is not None and position < len(ap50_values) else None,
                "map50_95": maps[class_id] if class_id < len(maps) else None,
            }
    return output


def _load_yaml(path: Path) -> dict:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(f"Não foi possível ler {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"A raiz de {path} precisa ser um mapa YAML")
    return value


def _dataset_yaml_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "edgevision_12h.yaml")
    parser.add_argument(
        "--check-data-only", action="store_true", help="Confere imagens e rótulos sem iniciar o treino"
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = _load_yaml(config_path)
    dataset_dir = Path(config.get("dataset_dir", "data/edgevision"))
    if not dataset_dir.is_absolute():
        dataset_dir = PROJECT_ROOT / dataset_dir
    dataset_dir = dataset_dir.resolve()
    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.is_file():
        raise RuntimeError(f"Dados ausentes. Execute scripts/download_edgevision.py primeiro: {data_yaml}")

    data_config = _load_yaml(data_yaml)
    expected_classes = ["BikeWithRider", "NoHelmet", "Helmet"]
    names = data_config.get("names", {})
    if isinstance(names, dict):
        names = [names[index] for index in sorted(names, key=lambda item: int(item))]
    if list(names) != expected_classes:
        raise RuntimeError(f"Mapa de classes inesperado: {names}")
    for split in ("train", "val", "test"):
        split_file = dataset_dir / "splits" / f"{split}.txt"
        if not split_file.is_file() or not split_file.read_text(encoding="utf-8").strip():
            raise RuntimeError(f"Partição ausente ou vazia: {split_file}")
        entries = split_file.read_text(encoding="utf-8").splitlines()
        missing_images = [
            line
            for line in entries
            if not (split_file.parent / line).is_file()
        ]
        if missing_images:
            examples = ", ".join(missing_images[:3])
            raise RuntimeError(
                f"A partição {split} aponta para imagens inexistentes ({len(missing_images)}): {examples}"
            )
        print(f"Pré-voo {split}: conferindo {len(entries)} imagens e rótulos...", flush=True)
        for index, line in enumerate(entries, start=1):
            image_path = Path(line)
            if not image_path.is_absolute():
                image_path = (split_file.parent / image_path).resolve()
            label_path = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
            if not label_path.is_file():
                raise RuntimeError(f"Rótulo YOLO ausente para {image_path.name}: {label_path}")
            try:
                with image_path.open("rb") as image_stream:
                    image_stream.seek(-2, 2)
                    if image_stream.read(2) != b"\xff\xd9":
                        raise RuntimeError("marcador final JPEG ausente")
                with Image.open(image_path) as image:
                    orientation = image.getexif().get(274, 1)
                if orientation in (5, 6, 7, 8):
                    raise RuntimeError(f"rotação EXIF {orientation} ainda pendente")
            except Exception as exc:
                raise RuntimeError(f"Imagem não está pronta para treino: {image_path}: {exc}") from exc
            if index % 500 == 0 or index == len(entries):
                print(f"Pré-voo {split}: {index}/{len(entries)} OK", flush=True)

    print("Pré-voo concluído: imagens e rótulos encontrados.", flush=True)
    if args.check_data_only:
        return 0

    import torch
    import ultralytics
    from ultralytics import YOLO

    if torch.cuda.is_available():
        print("Aviso: CUDA está disponível, mas este perfil foi configurado para CPU.")

    budget = config.get("budget", {})
    total_budget_hours = float(budget.get("total_hours", 12))
    reserve_hours = float(budget.get("evaluation_reserve_hours", 3))
    per_model_cap_hours = float(budget.get("max_training_hours_per_model", 4.5))
    models = config.get("models", [])
    if len(models) != 2:
        raise RuntimeError("A configuração precisa listar exatamente dois modelos")
    training_budget_hours = total_budget_hours - reserve_hours
    if training_budget_hours <= 0 or 2 * per_model_cap_hours > training_budget_hours:
        raise RuntimeError("O orçamento por modelo precisa caber no orçamento total menos a reserva de avaliação")

    train_config = dict(config.get("train", {}))
    evaluation_config = dict(config.get("evaluation", {}))
    if str(train_config.get("device", "cpu")).lower() != "cpu":
        raise RuntimeError("Este perfil deve usar device: cpu")
    if int(train_config.get("epochs", 0)) < 1:
        raise RuntimeError("train.epochs precisa ser positivo")

    output_base = Path(config.get("output_dir", "runs/edgevision"))
    if not output_base.is_absolute():
        output_base = PROJECT_ROOT / output_base
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_base / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    (run_dir / "data.yaml").write_text(data_yaml.read_text(encoding="utf-8"), encoding="utf-8")
    for filename in (
        "split_summary.json", "annotation_conversion_report.json", "image_manifest.json", "preprocessing_report.json"
    ):
        source = dataset_dir / filename
        if source.is_file():
            shutil.copy2(source, run_dir / filename)
    shutil.copytree(dataset_dir / "splits", run_dir / "splits")

    try:
        split_summary = json.loads((dataset_dir / "split_summary.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        split_summary = {}

    run_started = time.monotonic()
    training_times: dict[str, float] = {}
    model_outputs: dict[str, dict] = {}
    best_weights: dict[str, Path] = {}
    train_deadline = run_started + training_budget_hours * 3600
    for index, model_spec in enumerate(models):
        model_name = str(model_spec["name"])
        weights_name = str(model_spec["weights"])
        remaining_models = len(models) - index
        remaining_training_hours = max(0.0, (train_deadline - time.monotonic()) / 3600)
        model_budget_hours = min(per_model_cap_hours, remaining_training_hours / remaining_models)
        if model_budget_hours <= 0:
            raise RuntimeError("O orçamento de treino acabou antes de iniciar o segundo modelo")

        print(f"\n=== Treino {model_name}: até {model_budget_hours:.2f} h em CPU ===", flush=True)
        model = YOLO(weights_name)
        model_started = time.monotonic()
        callback_state = {"budget_reached": False}

        def stop_at_budget(trainer, started=model_started, seconds=model_budget_hours * 3600, state=callback_state):
            if time.monotonic() - started >= seconds:
                trainer.stop = True
                state["budget_reached"] = True

        model.add_callback("on_train_epoch_end", stop_at_budget)
        train_args = dict(train_config)
        train_args.update(
            {
                "data": str(data_yaml),
                "project": str(run_dir),
                "name": f"{model_name}_train",
                "exist_ok": False,
                "save": True,
                "val": True,
                "verbose": True,
            }
        )
        model.train(**train_args)
        elapsed = time.monotonic() - model_started
        training_times[model_name] = elapsed
        trainer = getattr(model, "trainer", None)
        save_dir = Path(trainer.save_dir) if trainer and getattr(trainer, "save_dir", None) else run_dir / f"{model_name}_train"
        candidate = Path(trainer.best) if trainer and getattr(trainer, "best", None) else save_dir / "weights" / "best.pt"
        candidate = candidate.resolve()
        if not candidate.is_file():
            raise RuntimeError(f"Não encontrei o melhor checkpoint de {model_name}: {candidate}")
        completed_epochs = int(getattr(trainer, "epoch", -1)) + 1 if trainer else None
        best_weights[model_name] = candidate
        model_outputs[model_name] = {
            "initial_weights": weights_name,
            "best_weights": str(candidate),
            "training_seconds": elapsed,
            "training_hours": elapsed / 3600,
            "training_budget_hours": model_budget_hours,
            "budget_stop_requested": callback_state["budget_reached"],
            "completed_epochs": completed_epochs,
        }
        del model
        gc.collect()

    evaluation_results: dict[str, dict] = {}
    for model_spec in models:
        model_name = str(model_spec["name"])
        model = YOLO(str(best_weights[model_name]))
        evaluation_results[model_name] = {}
        for split in ("val", "test"):
            print(f"\n=== Avaliação {model_name} / {split} ===", flush=True)
            metrics = model.val(
                data=str(data_yaml),
                split=split,
                imgsz=int(evaluation_config.get("imgsz", train_config.get("imgsz", 640))),
                batch=int(evaluation_config.get("batch", 1)),
                device="cpu",
                workers=int(evaluation_config.get("workers", 2)),
                conf=float(evaluation_config.get("conf", 0.001)),
                iou=float(evaluation_config.get("iou", 0.7)),
                max_det=int(evaluation_config.get("max_det", 300)),
                plots=bool(evaluation_config.get("plots", True)),
                save_json=bool(evaluation_config.get("save_json", True)),
                project=str(run_dir / "evaluation"),
                name=f"{model_name}_{split}",
                exist_ok=False,
                verbose=False,
            )
            evaluation_results[model_name][split] = _as_plain_metrics(metrics, list(names))
        del model
        gc.collect()

    elapsed_total = time.monotonic() - run_started
    summary = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "python": sys.version.split()[0],
            "operating_system": platform.platform(),
            "processor": platform.processor(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
        },
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
        "device": "cpu",
        "dataset": split_summary,
        "total_budget_hours": total_budget_hours,
        "evaluation_reserve_hours": reserve_hours,
        "total_wall_time_seconds": elapsed_total,
        "models": model_outputs,
        "evaluation": evaluation_results,
        "interpretation": "Imagem-level pilot. This dataset has no video IDs and is separate from the four-class video protocol.",
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        fields = [
            "model", "epochs", "training_hours", "val_precision", "val_recall", "val_map50_95", "val_map50",
            "test_precision", "test_recall", "test_map50_95", "test_map50",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, values in model_outputs.items():
            val_metrics = evaluation_results[name].get("val", {})
            test_metrics = evaluation_results[name].get("test", {})
            writer.writerow(
                {
                    "model": name,
                    "epochs": values.get("completed_epochs"),
                    "training_hours": values.get("training_hours"),
                    "val_precision": val_metrics.get("metrics/precision(B)", val_metrics.get("mp")),
                    "val_recall": val_metrics.get("metrics/recall(B)", val_metrics.get("mr")),
                    "val_map50_95": val_metrics.get("metrics/mAP50-95(B)", val_metrics.get("map")),
                    "val_map50": val_metrics.get("metrics/mAP50(B)", val_metrics.get("map50")),
                    "test_precision": test_metrics.get("metrics/precision(B)", test_metrics.get("mp")),
                    "test_recall": test_metrics.get("metrics/recall(B)", test_metrics.get("mr")),
                    "test_map50_95": test_metrics.get("metrics/mAP50-95(B)", test_metrics.get("map")),
                    "test_map50": test_metrics.get("metrics/mAP50(B)", test_metrics.get("map50")),
                }
            )
    with (run_dir / "per_class.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        fields = [
            "model", "split", "class_name", "class_id", "precision_at_best_f1", "recall_at_best_f1",
            "f1_at_best_f1", "map50", "map50_95",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for model_name, splits in evaluation_results.items():
            for split, metrics in splits.items():
                for class_name, values in metrics.get("per_class", {}).items():
                    writer.writerow(
                        {
                            "model": model_name,
                            "split": split,
                            "class_name": class_name,
                            **values,
                        }
                    )
    print(f"\nComparação concluída em {elapsed_total / 3600:.2f} h. Relatórios: {run_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, yaml.YAMLError) as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        raise SystemExit(1)
