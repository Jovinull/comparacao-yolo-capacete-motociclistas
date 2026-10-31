#!/usr/bin/env python3
"""Download and prepare the public EdgeVision dataset for an image-only pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml
from PIL import Image, ImageOps


DATASET_ID = "j82bnw7gsr"
DATASET_DOI = "10.17632/j82bnw7gsr.1"
API_URL = f"https://data.mendeley.com/public-api/datasets/{DATASET_ID}"
EXPECTED_CLASSES = ["BikeWithRider", "NoHelmet", "Helmet"]
SPLIT_RATIOS = {"train": 0.80, "val": 0.10, "test": 0.10}
USER_AGENT = "comparacao-yolo-capacete-motociclistas/1.0 (dataset setup)"


def fetch_json(url: str, *, attempts: int = 4) -> dict:
    for attempt in range(attempts):
        request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
        try:
            with urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError) as exc:
            if attempt + 1 == attempts:
                raise RuntimeError(f"Falha ao consultar {url}: {exc}") from exc
            time.sleep(2**attempt)
    raise RuntimeError(f"Falha ao consultar {url}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(record: dict, destination: Path) -> tuple[str, bool]:
    details = record.get("content_details") or {}
    expected_size = int(details.get("size", record.get("size", 0)))
    expected_hash = str(details.get("sha256_hash", ""))
    if destination.is_file() and destination.stat().st_size == expected_size:
        if not expected_hash or sha256_file(destination) == expected_hash:
            return record["filename"], False

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    download_url = details.get("download_url")
    if not download_url:
        raise RuntimeError(f"URL de download ausente para {record['filename']}")

    last_error: Exception | None = None
    for attempt in range(4):
        try:
            request = Request(download_url, headers={"User-Agent": USER_AGENT})
            digest = hashlib.sha256()
            received = 0
            with urlopen(request, timeout=120) as response, partial.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
                    digest.update(chunk)
                    received += len(chunk)
            if expected_size and received != expected_size:
                raise RuntimeError(
                    f"Tamanho incorreto em {record['filename']}: {received} != {expected_size}"
                )
            if expected_hash and digest.hexdigest() != expected_hash:
                raise RuntimeError(f"SHA-256 incorreto em {record['filename']}")
            partial.replace(destination)
            return record["filename"], True
        except (HTTPError, URLError, TimeoutError, OSError, RuntimeError) as exc:
            last_error = exc
            partial.unlink(missing_ok=True)
            if attempt < 3:
                time.sleep(2**attempt)
    raise RuntimeError(f"Falha ao baixar {record['filename']}: {last_error}")


def prepare_training_images(source_dir: Path, training_dir: Path, records: list[dict], root: Path) -> dict:
    """Keep downloaded originals intact and make clean JPEGs for Ultralytics to read."""
    training_dir.mkdir(parents=True, exist_ok=True)
    reencoded = []
    exif_normalized = []
    for record in records:
        filename = str(record["filename"])
        source_path = source_dir / filename
        training_path = training_dir / filename
        with source_path.open("rb") as stream:
            stream.seek(-2, 2)
            has_eoi_marker = stream.read(2) == b"\xff\xd9"
        with Image.open(source_path) as source_image:
            orientation = source_image.getexif().get(274, 1)
            if orientation not in (0, 1, 5, 6, 7, 8):
                raise RuntimeError(f"Orientação EXIF {orientation} não tratada em {filename}")
            rotate_for_display = orientation in (5, 6, 7, 8)
            if not has_eoi_marker or rotate_for_display:
                normalized = ImageOps.exif_transpose(source_image)
                normalized.save(training_path, "JPEG", subsampling=0, quality=100)
                reencoded.append(filename)
                if rotate_for_display:
                    exif_normalized.append(filename)
            else:
                shutil.copy2(source_path, training_path)

        with training_path.open("rb") as stream:
            stream.seek(-2, 2)
            if stream.read(2) != b"\xff\xd9":
                raise RuntimeError(f"JPEG preparado sem marcador final: {filename}")
        with Image.open(training_path) as prepared_image:
            prepared_orientation = prepared_image.getexif().get(274, 1)
            prepared_image.verify()
        if prepared_orientation in (5, 6, 7, 8):
            raise RuntimeError(f"A imagem preparada ainda requer rotação EXIF: {filename}")

    report = {
        "source_images": len(records),
        "training_images": len(records),
        "source_images_directory": str(source_dir.resolve()),
        "training_images_directory": str(training_dir.resolve()),
        "reencoded_to_repair_jpeg_or_apply_exif": reencoded,
        "exif_rotated_images_normalized": exif_normalized,
    }
    (root / "preprocessing_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def _split_stratum(items: list[dict], rng: random.Random) -> dict[str, list[dict]]:
    """Split equal class-presence combinations with deterministic largest remainders."""
    shuffled = list(items)
    rng.shuffle(shuffled)
    raw = {name: len(shuffled) * ratio for name, ratio in SPLIT_RATIOS.items()}
    counts = {name: math.floor(value) for name, value in raw.items()}
    left = len(shuffled) - sum(counts.values())
    order = sorted(SPLIT_RATIOS, key=lambda name: (raw[name] - counts[name], rng.random()), reverse=True)
    for name in order[:left]:
        counts[name] += 1
    output: dict[str, list[dict]] = {}
    cursor = 0
    for name in SPLIT_RATIOS:
        output[name] = shuffled[cursor : cursor + counts[name]]
        cursor += counts[name]
    return output


def make_splits(images: list[dict], annotations_by_image: dict[int, list[dict]], seed: int) -> dict[str, list[dict]]:
    strata: dict[tuple[int, ...], list[dict]] = defaultdict(list)
    for image in images:
        present = tuple(
            sorted({int(annotation["class_index"]) for annotation in annotations_by_image.get(int(image["id"]), [])})
        )
        strata[present].append(image)

    rng = random.Random(seed)
    output = {name: [] for name in SPLIT_RATIOS}
    for label_set in sorted(strata, key=lambda labels: (len(labels), labels)):
        parts = _split_stratum(strata[label_set], rng)
        for name, rows in parts.items():
            output[name].extend(rows)
    for rows in output.values():
        rng.shuffle(rows)
    if any(not output[name] for name in output):
        raise RuntimeError("A divisão train/val/test produziu uma partição vazia")
    return output


def write_yolo_labels(root: Path, images: list[dict], annotations: list[dict], categories: dict[int, str], class_names: list[str], source_image_dir: Path) -> tuple[dict[int, list[dict]], dict]:
    image_by_id = {int(image["id"]): image for image in images}
    class_index = {name: index for index, name in enumerate(class_names)}
    dimensions: dict[int, tuple[int, int]] = {}
    orientation_corrections = []
    for image in images:
        image_id = int(image["id"])
        training_image_path = root / "prepared" / "images" / str(image["file_name"])
        source_image_path = source_image_dir / str(image["file_name"])
        with Image.open(training_image_path) as training_image:
            width, height = training_image.size
        with Image.open(source_image_path) as source_image:
            orientation = source_image.getexif().get(274, 1)
        if orientation not in (0, 1, 5, 6, 7, 8):
            raise RuntimeError(f"Orientação EXIF {orientation} não tratada em {image['file_name']}")
        dimensions[image_id] = (width, height)
        if (width, height) != (int(image["width"]), int(image["height"])):
            orientation_corrections.append(
                {
                    "filename": image["file_name"],
                    "coco_dimensions": [int(image["width"]), int(image["height"])],
                    "oriented_dimensions": [width, height],
                    "exif_orientation": orientation,
                }
            )

    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    clipped_annotations = []
    skipped_annotations = []
    for annotation in annotations:
        image_id = int(annotation["image_id"])
        image = image_by_id.get(image_id)
        if image is None:
            raise RuntimeError(f"Anotação aponta para imagem inexistente: {image_id}")
        category_id = int(annotation["category_id"])
        category_name = categories.get(category_id)
        if category_name not in class_index:
            raise RuntimeError(f"Categoria inesperada no COCO: {category_name}")
        width, height = map(float, dimensions[image_id])
        x, y, box_width, box_height = map(float, annotation["bbox"])
        if width <= 0 or height <= 0 or box_width <= 0 or box_height <= 0:
            skipped_annotations.append(
                {
                    "annotation_id": int(annotation["id"]),
                    "filename": image["file_name"],
                    "reason": "nonpositive_image_or_box_dimensions",
                    "source_bbox": [x, y, box_width, box_height],
                }
            )
            continue
        x1, y1 = max(0.0, min(x, width)), max(0.0, min(y, height))
        x2, y2 = max(0.0, min(x + box_width, width)), max(0.0, min(y + box_height, height))
        if x2 <= x1 or y2 <= y1:
            skipped_annotations.append(
                {
                    "annotation_id": int(annotation["id"]),
                    "filename": image["file_name"],
                    "reason": "box_outside_image_after_clipping",
                    "source_bbox": [x, y, box_width, box_height],
                }
            )
            continue
        if (x1, y1, x2, y2) != (x, y, x + box_width, y + box_height):
            clipped_annotations.append(
                {
                    "annotation_id": int(annotation["id"]),
                    "filename": image["file_name"],
                    "source_bbox": [x, y, box_width, box_height],
                    "clipped_bbox": [x1, y1, x2 - x1, y2 - y1],
                }
            )
        converted = {
            "class_index": class_index[category_name],
            "cx": (x1 + (x2 - x1) / 2) / width,
            "cy": (y1 + (y2 - y1) / 2) / height,
            "width": (x2 - x1) / width,
            "height": (y2 - y1) / height,
        }
        annotations_by_image[image_id].append(converted)

    labels_dir = root / "prepared" / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    for image in images:
        label_path = labels_dir / (Path(image["file_name"]).stem + ".txt")
        lines = [
            f"{row['class_index']} {row['cx']:.6f} {row['cy']:.6f} {row['width']:.6f} {row['height']:.6f}"
            for row in annotations_by_image.get(int(image["id"]), [])
        ]
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    report = {
        "source_annotations": len(annotations),
        "converted_annotations": sum(len(rows) for rows in annotations_by_image.values()),
        "images_with_exif_dimension_correction": orientation_corrections,
        "annotations_clipped_to_image_bounds": clipped_annotations,
        "annotations_skipped": skipped_annotations,
    }
    (root / "annotation_conversion_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return annotations_by_image, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "edgevision")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 32:
        parser.error("--workers deve ficar entre 1 e 32")

    root = args.root.resolve()
    source_dir = root / "source"
    source_image_dir = source_dir / "images"
    image_dir = root / "prepared" / "images"
    source_dir.mkdir(parents=True, exist_ok=True)
    source_image_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    metadata = fetch_json(API_URL)
    if metadata.get("id") != DATASET_ID or int(metadata.get("version", 0)) != 1:
        raise RuntimeError("A API retornou um dataset/versão inesperado")
    records = metadata.get("files", [])
    files_by_name = {record["filename"]: record for record in records}
    required_source = ["annotations.json", "classes.txt", "LICENSE.txt", "README.txt", "annotations.csv"]
    missing = [name for name in required_source if name not in files_by_name]
    if missing:
        raise RuntimeError(f"Arquivos de metadados ausentes na versão publicada: {missing}")

    for name in required_source:
        download_file(files_by_name[name], source_dir / name)

    classes = [line.strip() for line in (source_dir / "classes.txt").read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if classes != EXPECTED_CLASSES:
        raise RuntimeError(f"Mapa de classes divergente: esperado {EXPECTED_CLASSES}, recebido {classes}")
    coco = json.loads((source_dir / "annotations.json").read_text(encoding="utf-8"))
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = {int(row["id"]): str(row["name"]) for row in coco.get("categories", [])}
    if len(images) != 2392:
        raise RuntimeError(f"Esperava 2.392 imagens e encontrei {len(images)}")
    if set(categories.values()) != set(classes):
        raise RuntimeError(f"Categorias COCO divergentes: {categories}")

    image_records = []
    for image in images:
        filename = str(image["file_name"])
        record = files_by_name.get(filename)
        if record is None or not filename.lower().endswith(".jpg"):
            raise RuntimeError(f"Arquivo de imagem não encontrado na versão Mendeley: {filename}")
        image_records.append(record)
    required_bytes = sum(int(row.get("size", 0)) for row in image_records)
    free_bytes = shutil.disk_usage(root.parent).free
    if free_bytes < required_bytes * 3 + 512 * 1024 * 1024:
        raise RuntimeError(
            f"Espaço insuficiente: precisa reservar até {required_bytes * 3 / 1e9:.2f} GB para fontes e cópias processadas"
        )

    print(
        f"Baixando {len(image_records)} imagens originais ({required_bytes / 1e9:.2f} GB) "
        f"para {source_image_dir}"
    )
    downloaded = 0
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_file, record, source_image_dir / record["filename"]): record["filename"]
            for record in image_records
        }
        for future in as_completed(futures):
            _, was_downloaded = future.result()
            downloaded += int(was_downloaded)
            completed += 1
            if completed % 200 == 0 or completed == len(futures):
                print(f"Imagens verificadas: {completed}/{len(futures)}")

    preprocessing_report = prepare_training_images(source_image_dir, image_dir, image_records, root)
    print(
        f"Cópias de treino preparadas: {len(image_records)} imagens; "
        f"{len(preprocessing_report['reencoded_to_repair_jpeg_or_apply_exif'])} regravadas sem alterar as originais."
    )
    annotations_by_image, conversion_report = write_yolo_labels(
        root, images, annotations, categories, classes, source_image_dir
    )
    splits = make_splits(images, annotations_by_image, args.seed)
    split_dir = root / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}
    for split_name, split_images in splits.items():
        # Absolute paths avoid Ultralytics' split-list relative-path rewriting on Windows.
        lines = [
            (root / "prepared" / "images" / str(image["file_name"])).resolve().as_posix()
            for image in split_images
        ]
        (split_dir / f"{split_name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        per_class_images = Counter()
        per_class_instances = Counter()
        for image in split_images:
            rows = annotations_by_image.get(int(image["id"]), [])
            per_class_images.update({row["class_index"] for row in rows})
            per_class_instances.update(row["class_index"] for row in rows)
        summary[split_name] = {
            "images": len(split_images),
            "images_with_class": {classes[index]: per_class_images[index] for index in range(len(classes))},
            "instances_by_class": {classes[index]: per_class_instances[index] for index in range(len(classes))},
        }

    dataset_yaml = {
        "path": root.as_posix(),
        "train": "splits/train.txt",
        "val": "splits/val.txt",
        "test": "splits/test.txt",
        "names": {index: name for index, name in enumerate(classes)},
    }
    (root / "data.yaml").write_text(yaml.safe_dump(dataset_yaml, allow_unicode=True, sort_keys=False), encoding="utf-8")
    split_summary = {
        "dataset": "EdgeVision Dataset",
        "dataset_id": DATASET_ID,
        "doi": DATASET_DOI,
        "version": 1,
        "license": "CC BY 4.0",
        "class_names": classes,
        "images": len(images),
        "instances": len(annotations),
        "converted_instances": conversion_report["converted_annotations"],
        "clipped_instances": len(conversion_report["annotations_clipped_to_image_bounds"]),
        "skipped_instances": len(conversion_report["annotations_skipped"]),
        "images_with_exif_dimension_correction": len(
            conversion_report["images_with_exif_dimension_correction"]
        ),
        "reencoded_training_images": len(preprocessing_report["reencoded_to_repair_jpeg_or_apply_exif"]),
        "seed": args.seed,
        "split_ratios": SPLIT_RATIOS,
        "splits": summary,
        "images_downloaded_this_run": downloaded,
        "image_bytes": required_bytes,
    }
    (root / "split_summary.json").write_text(json.dumps(split_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = [
        {
            "filename": row["filename"],
            "size": int(row.get("size", 0)),
            "sha256": row.get("content_details", {}).get("sha256_hash"),
        }
        for row in image_records
    ]
    (root / "image_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"Preparação concluída: {len(images)} imagens, "
        f"{conversion_report['converted_annotations']} caixas convertidas, divisão 80/10/10."
    )
    print(f"Configuração de dados: {root / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        raise SystemExit(1)
