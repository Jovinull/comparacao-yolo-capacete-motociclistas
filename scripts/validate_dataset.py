#!/usr/bin/env python3
"""Validate the YOLO dataset layout and enforce video-level partitioning."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml


SPLITS = ("train", "val", "test")
EXPECTED_NAMES = ("com_capacete", "sem_capacete")


@dataclass(frozen=True)
class ManifestRecord:
    image_path: Path
    image_relative_path: Path
    label_path: Path
    video_id: str
    split: str


@dataclass(frozen=True)
class DatasetInfo:
    root: Path
    names: tuple[str, ...]
    records: tuple[ManifestRecord, ...]
    report: dict


class DatasetValidationError(ValueError):
    pass


def _normalize_names(raw_names: object) -> tuple[str, ...]:
    if isinstance(raw_names, list):
        return tuple(str(name) for name in raw_names)
    if isinstance(raw_names, dict):
        try:
            return tuple(str(raw_names[key]) for key in sorted(raw_names, key=lambda x: int(x)))
        except (TypeError, ValueError):
            return ()
    return ()


def _resolve_dataset_root(config: dict, data_yaml: Path) -> Path:
    raw_root = config.get("path", ".")
    root = Path(raw_root)
    if not root.is_absolute():
        root = data_yaml.parent / root
    return root.resolve()


def _label_path(root: Path, image_relative_path: Path) -> Path:
    if not image_relative_path.parts or image_relative_path.parts[0].lower() != "images":
        raise ValueError("image_path precisa começar com images/<split>/")
    return root / "labels" / Path(*image_relative_path.parts[1:]).with_suffix(".txt")


def _read_label_file(label_path: Path, num_classes: int, errors: list[str]) -> Counter:
    counts: Counter = Counter()
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        errors.append(f"Não foi possível ler {label_path}: {exc}")
        return counts

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        fields = line.split()
        prefix = f"{label_path}:{line_number}"
        if len(fields) != 5:
            errors.append(f"{prefix}: esperado class_id x_center y_center width height")
            continue
        try:
            class_id = int(fields[0])
            box = [float(value) for value in fields[1:]]
        except ValueError:
            errors.append(f"{prefix}: classe ou coordenada não numérica")
            continue
        if not 0 <= class_id < num_classes:
            errors.append(f"{prefix}: class_id {class_id} fora do intervalo 0..{num_classes - 1}")
            continue
        if not all(math.isfinite(value) for value in box):
            errors.append(f"{prefix}: coordenadas devem ser finitas")
            continue
        x_center, y_center, width, height = box
        if not all(0.0 <= value <= 1.0 for value in box):
            errors.append(f"{prefix}: coordenadas YOLO devem estar normalizadas entre 0 e 1")
            continue
        if width <= 0.0 or height <= 0.0:
            errors.append(f"{prefix}: width e height precisam ser maiores que zero")
            continue
        if x_center - width / 2 < 0 or x_center + width / 2 > 1:
            errors.append(f"{prefix}: caixa ultrapassa a largura da imagem")
            continue
        if y_center - height / 2 < 0 or y_center + height / 2 > 1:
            errors.append(f"{prefix}: caixa ultrapassa a altura da imagem")
            continue
        counts[class_id] += 1
    return counts


def validate_dataset(data_yaml_path: Path) -> DatasetInfo:
    data_yaml_path = data_yaml_path.resolve()
    if not data_yaml_path.is_file():
        raise DatasetValidationError(f"Arquivo de dados não encontrado: {data_yaml_path}")

    try:
        config = yaml.safe_load(data_yaml_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise DatasetValidationError(f"Não foi possível ler {data_yaml_path}: {exc}") from exc
    if not isinstance(config, dict):
        raise DatasetValidationError("O YAML do conjunto precisa conter um objeto na raiz")

    errors: list[str] = []
    warnings: list[str] = []
    root = _resolve_dataset_root(config, data_yaml_path)
    names = _normalize_names(config.get("names"))
    if names != EXPECTED_NAMES:
        errors.append(f"names precisa ser {list(EXPECTED_NAMES)} na ordem definida; recebido {list(names)}")
    if int(config.get("nc", len(names) or 0)) != 2:
        errors.append("nc precisa ser 2")

    declared_split_paths: dict[str, list[Path]] = {}
    for split in SPLITS:
        entries = config.get(split)
        if entries is None:
            errors.append(f"data.yaml não define a partição {split}")
            declared_split_paths[split] = []
            continue
        entries = entries if isinstance(entries, list) else [entries]
        declared_split_paths[split] = []
        for entry in entries:
            split_path = Path(str(entry))
            if not split_path.is_absolute():
                split_path = root / split_path
            split_path = split_path.resolve()
            declared_split_paths[split].append(split_path)
            if not split_path.exists():
                errors.append(f"Caminho de {split} não encontrado: {split_path}")

    manifest_path = root / "manifest.csv"
    if not manifest_path.is_file():
        errors.append(f"Manifesto não encontrado: {manifest_path}")
        if errors:
            raise DatasetValidationError("\n".join(errors))

    records: list[ManifestRecord] = []
    image_paths_seen: set[str] = set()
    video_splits: dict[str, str] = {}
    image_counts: Counter = Counter()
    video_counts: dict[str, set[str]] = {split: set() for split in SPLITS}
    class_counts: dict[str, Counter] = {split: Counter() for split in SPLITS}

    try:
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            required_columns = {"image_path", "video_id", "split"}
            if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
                errors.append("manifest.csv precisa ter as colunas image_path,video_id,split")
            else:
                for row_number, row in enumerate(reader, start=2):
                    raw_image = (row.get("image_path") or "").strip().replace("\\", "/")
                    video_id = (row.get("video_id") or "").strip()
                    split = (row.get("split") or "").strip().lower()
                    prefix = f"{manifest_path}:{row_number}"
                    if not raw_image or not video_id:
                        errors.append(f"{prefix}: image_path e video_id não podem ficar vazios")
                        continue
                    if split not in SPLITS:
                        errors.append(f"{prefix}: split precisa ser train, val ou test")
                        continue

                    relative_path = Path(raw_image)
                    if relative_path.is_absolute() or ".." in relative_path.parts:
                        errors.append(f"{prefix}: image_path precisa ser relativo e ficar dentro do conjunto")
                        continue
                    if len(relative_path.parts) < 3 or relative_path.parts[:2] != ("images", split):
                        errors.append(f"{prefix}: image_path precisa estar em images/{split}/")
                        continue
                    image_path = (root / relative_path).resolve()
                    if not image_path.is_relative_to(root):
                        errors.append(f"{prefix}: image_path sai da raiz do conjunto")
                        continue
                    if not any(
                        image_path.is_relative_to(split_path)
                        for split_path in declared_split_paths.get(split, [])
                    ):
                        errors.append(f"{prefix}: imagem não pertence ao caminho definido para {split} em data.yaml")
                        continue
                    normalized = relative_path.as_posix().casefold()
                    if normalized in image_paths_seen:
                        errors.append(f"{prefix}: image_path duplicado: {raw_image}")
                        continue
                    image_paths_seen.add(normalized)
                    if not image_path.is_file():
                        errors.append(f"{prefix}: imagem não encontrada: {image_path}")
                        continue

                    try:
                        label_path = _label_path(root, relative_path)
                    except ValueError as exc:
                        errors.append(f"{prefix}: {exc}")
                        continue
                    if not label_path.is_file():
                        errors.append(f"{prefix}: rótulo não encontrado: {label_path}")
                        continue

                    previous_split = video_splits.setdefault(video_id, split)
                    if previous_split != split:
                        errors.append(
                            f"{prefix}: vídeo {video_id!r} aparece em {previous_split} e {split}"
                        )
                    image_counts[split] += 1
                    video_counts[split].add(video_id)
                    class_counts[split].update(_read_label_file(label_path, len(names), errors))
                    records.append(ManifestRecord(image_path, relative_path, label_path, video_id, split))
    except OSError as exc:
        errors.append(f"Não foi possível ler {manifest_path}: {exc}")

    if not records:
        errors.append("manifest.csv não contém quadros válidos")
    for split in SPLITS:
        if image_counts[split] == 0:
            errors.append(f"A partição {split} não contém imagens válidas")
    for class_id, class_name in enumerate(names):
        if class_counts["train"][class_id] == 0:
            errors.append(f"A classe {class_name} não tem objetos no treino")
        for split in ("val", "test"):
            if image_counts[split] and class_counts[split][class_id] == 0:
                warnings.append(f"A classe {class_name} não tem objetos em {split}; métricas dessa classe serão limitadas")

    if errors:
        details = "\n".join(f"- {message}" for message in errors)
        if warnings:
            details += "\nAvisos:\n" + "\n".join(f"- {message}" for message in warnings)
        raise DatasetValidationError(details)

    report = {
        "data_yaml": str(data_yaml_path),
        "dataset_root": str(root),
        "images_by_split": dict(image_counts),
        "videos_by_split": {split: len(video_counts[split]) for split in SPLITS},
        "objects_by_split_and_class": {
            split: {name: class_counts[split][class_id] for class_id, name in enumerate(names)}
            for split in SPLITS
        },
        "warnings": warnings,
    }
    return DatasetInfo(root, names, tuple(records), report)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Caminho para data.yaml")
    args = parser.parse_args()
    try:
        dataset = validate_dataset(args.data)
    except DatasetValidationError as exc:
        print(f"ERRO: validação do conjunto falhou:\n{exc}")
        return 2

    print(json.dumps(dataset.report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
