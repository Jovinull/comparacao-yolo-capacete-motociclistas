#!/usr/bin/env python3
"""Download and convert the public OSF HELMET dataset for YOLO comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "data" / "helmet_osf"
NODE_ID = "4pwj8"
NODE_API = f"https://api.osf.io/v2/nodes/{NODE_ID}/"
FILES_API = f"https://api.osf.io/v2/nodes/{NODE_ID}/files/osfstorage/?page%5Bsize%5D=100"
DATASET_URL = f"https://osf.io/{NODE_ID}/"
API_USER_AGENT = "comparacao-yolo-capacete-motociclistas/1.0 (OSF dataset preparation)"
IMAGE_SIZE = (1920, 1080)
EXPECTED_VIDEO_COUNT = 910
EXPECTED_FRAMES_PER_VIDEO = 100
EXPECTED_FRAME_COUNT = EXPECTED_VIDEO_COUNT * EXPECTED_FRAMES_PER_VIDEO
CLASS_NAMES = [
    "motocicleta_todos_com_capacete",
    "motocicleta_com_ocupante_sem_capacete",
]
SPLIT_MAP = {"training": "train", "validation": "val", "test": "test"}
OCCUPANT_PATTERN = re.compile(r"(?:D|P[0-3])(?:NoHelmet|Helmet)")
PART_PATTERN = re.compile(r"part_[1-7]\.zip")


class DatasetSetupError(RuntimeError):
    pass


def _request_json(url: str) -> dict:
    request = Request(url, headers={"User-Agent": API_USER_AGENT, "Accept": "application/json"})
    try:
        with urlopen(request, timeout=60) as response:
            return json.load(response)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise DatasetSetupError(f"Falha ao consultar a API OSF ({url}): {exc}") from exc


def _list_osf_files() -> tuple[dict, list[dict]]:
    node = _request_json(NODE_API).get("data", {})
    attributes = node.get("attributes", {})
    if node.get("id") != NODE_ID or attributes.get("title") != "HELMET dataset":
        raise DatasetSetupError("A API OSF não confirmou o conjunto HELMET esperado.")
    if not attributes.get("public"):
        raise DatasetSetupError("O projeto OSF não está público; o download foi interrompido.")

    pending = [FILES_API]
    visited: set[str] = set()
    files: list[dict] = []
    while pending:
        url = pending.pop()
        if url in visited:
            continue
        visited.add(url)
        payload = _request_json(url)
        for item in payload.get("data", []):
            item_attributes = item.get("attributes", {})
            if item_attributes.get("kind") == "folder":
                child_url = (
                    item.get("relationships", {})
                    .get("files", {})
                    .get("links", {})
                    .get("related", {})
                    .get("href")
                )
                if child_url:
                    pending.append(child_url + ("&page%5Bsize%5D=100" if "?" in child_url else "?page%5Bsize%5D=100"))
            else:
                files.append(item)
        next_url = payload.get("links", {}).get("next")
        if next_url:
            pending.append(next_url)

    return node, files


def _select_dataset_files(files: list[dict]) -> dict[str, dict]:
    selected: dict[str, dict] = {}
    for item in files:
        attributes = item.get("attributes", {})
        materialized = str(attributes.get("materialized_path", "")).replace("\\", "/")
        name = attributes.get("name")
        if materialized.endswith("/HELMET_DATASET/annotation.zip"):
            selected["annotation.zip"] = item
        elif materialized.endswith("/HELMET_DATASET/data_split.csv"):
            selected["data_split.csv"] = item
        elif "/HELMET_DATASET/image/" in materialized and name and PART_PATTERN.fullmatch(str(name)):
            selected[str(name)] = item

    required = {"annotation.zip", "data_split.csv", *(f"part_{index}.zip" for index in range(1, 8))}
    missing = required - selected.keys()
    if missing:
        raise DatasetSetupError(f"A API OSF não listou os arquivos esperados: {', '.join(sorted(missing))}")
    return selected


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _download_file(file_record: dict, destination: Path) -> dict:
    attributes = file_record.get("attributes", {})
    expected_size = int(attributes.get("size") or 0)
    expected_hash = str(attributes.get("extra", {}).get("hashes", {}).get("sha256") or "")
    download_url = file_record.get("links", {}).get("download")
    file_name = str(attributes.get("name", destination.name))
    if not expected_size or not expected_hash or not download_url:
        raise DatasetSetupError(f"Metadados incompletos do arquivo OSF {file_name}.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size == expected_size and _sha256(destination) == expected_hash:
            print(f"Já baixado e verificado: {file_name}")
            return _source_file_record(file_record, destination)
        raise DatasetSetupError(
            f"O arquivo local {destination} existe, mas seu tamanho ou SHA-256 diverge do OSF. "
            "O arquivo foi preservado; mova-o manualmente se desejar baixá-lo novamente."
        )

    partial = destination.with_name(destination.name + ".partial")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > expected_size:
        raise DatasetSetupError(
            f"O parcial de {file_name} excede o tamanho publicado pelo OSF. "
            "O arquivo foi preservado; remova-o manualmente para reiniciar o download."
        )
    if offset == expected_size:
        actual_hash = _sha256(partial)
        if actual_hash.lower() != expected_hash.lower():
            raise DatasetSetupError(
                f"O parcial de {file_name} tem o tamanho esperado, mas o SHA-256 diverge do OSF. "
                "O arquivo foi preservado; remova-o manualmente para reiniciar o download."
            )
        os.replace(partial, destination)
        print(f"Download parcial finalizado e verificado: {file_name}")
        return _source_file_record(file_record, destination)

    headers = {"User-Agent": API_USER_AGENT}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = Request(download_url, headers=headers)
    started = time.monotonic()
    written = offset
    try:
        response = urlopen(request, timeout=120)
    except (HTTPError, URLError, TimeoutError) as exc:
        raise DatasetSetupError(f"Falha ao iniciar o download de {file_name}: {exc}") from exc

    with response:
        response_status = getattr(response, "status", 200)
        append = bool(offset and response_status == 206)
        if append:
            content_range = response.headers.get("Content-Range", "")
            if not content_range.startswith(f"bytes {offset}-"):
                raise DatasetSetupError(
                    f"O servidor retornou um intervalo inesperado para {file_name}: {content_range!r}."
                )
        if offset and not append:
            written = 0
        mode = "ab" if append else "wb"
        with partial.open(mode) as stream:
            while chunk := response.read(8 * 1024 * 1024):
                stream.write(chunk)
                written += len(chunk)
                if written == expected_size or written % (512 * 1024 * 1024) < len(chunk):
                    elapsed = max(time.monotonic() - started, 0.001)
                    print(
                        f"{file_name}: {written / (1024**3):.2f}/{expected_size / (1024**3):.2f} GiB "
                        f"({100 * written / expected_size:.1f}%), {written / elapsed / (1024**2):.1f} MiB/s"
                    )

    actual_size = partial.stat().st_size
    if actual_size != expected_size:
        raise DatasetSetupError(
            f"Download incompleto de {file_name}: esperado {expected_size} bytes, recebido {actual_size}. "
            "Execute novamente para retomar o arquivo parcial."
        )
    actual_hash = _sha256(partial)
    if actual_hash.lower() != expected_hash.lower():
        raise DatasetSetupError(
            f"SHA-256 não confere para {file_name}: esperado {expected_hash}, recebido {actual_hash}. "
            "O arquivo parcial foi preservado para inspeção."
        )
    os.replace(partial, destination)
    return _source_file_record(file_record, destination)


def _source_file_record(file_record: dict, local_path: Path) -> dict:
    attributes = file_record["attributes"]
    return {
        "name": attributes.get("name"),
        "materialized_path": attributes.get("materialized_path"),
        "size_bytes": int(attributes.get("size") or 0),
        "sha256_osf": attributes.get("extra", {}).get("hashes", {}).get("sha256"),
        "local_path": local_path.as_posix(),
    }


def _load_video_splits(path: Path) -> dict[str, str]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
    except OSError as exc:
        raise DatasetSetupError(f"Não foi possível ler a divisão oficial {path}: {exc}") from exc
    if not rows or not {"video_id", "Set"}.issubset(rows[0]):
        raise DatasetSetupError("data_split.csv não tem as colunas oficiais video_id,Set.")

    splits: dict[str, str] = {}
    for row in rows:
        video_id = (row.get("video_id") or "").strip()
        source_split = (row.get("Set") or "").strip().lower()
        if not video_id or not re.fullmatch(r"[A-Za-z0-9_-]+", video_id):
            raise DatasetSetupError(f"Identificador de vídeo inválido em data_split.csv: {video_id!r}")
        if source_split not in SPLIT_MAP:
            raise DatasetSetupError(f"Partição oficial desconhecida para {video_id}: {source_split!r}")
        if video_id in splits:
            raise DatasetSetupError(f"Vídeo repetido em data_split.csv: {video_id}")
        splits[video_id] = SPLIT_MAP[source_split]

    split_counts = Counter(splits.values())
    expected = {"train": 636, "val": 92, "test": 182}
    if len(splits) != EXPECTED_VIDEO_COUNT or dict(split_counts) != expected:
        raise DatasetSetupError(
            f"A divisão oficial mudou: {len(splits)} vídeos e {dict(split_counts)}; "
            f"esperado {EXPECTED_VIDEO_COUNT} vídeos e {expected}."
        )
    return splits


def _normalize_box(row: dict[str, str], video_id: str, frame_id: int) -> tuple[str, bool]:
    try:
        x, y, width, height = (float(row[key]) for key in ("x", "y", "w", "h"))
    except (KeyError, TypeError, ValueError) as exc:
        raise DatasetSetupError(f"Caixa inválida em {video_id}, quadro {frame_id}: {row}") from exc
    if not all(math.isfinite(value) for value in (x, y, width, height)) or width <= 0 or height <= 0:
        raise DatasetSetupError(f"Caixa não finita ou degenerada em {video_id}, quadro {frame_id}: {row}")

    image_width, image_height = IMAGE_SIZE
    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(float(image_width), x + width), min(float(image_height), y + height)
    clipped = (x1, y1, x2, y2) != (x, y, x + width, y + height)
    width, height = x2 - x1, y2 - y1
    if width <= 0 or height <= 0:
        raise DatasetSetupError(f"Caixa fora da imagem em {video_id}, quadro {frame_id}: {row}")

    label = (row.get("label") or "").strip()
    tokens = OCCUPANT_PATTERN.findall(label)
    if not tokens or "".join(tokens) != label:
        raise DatasetSetupError(f"Estado de capacete inesperado em {video_id}, quadro {frame_id}: {label!r}")
    class_id = 1 if any(token.endswith("NoHelmet") for token in tokens) else 0
    x_center = (x1 + width / 2) / image_width
    y_center = (y1 + height / 2) / image_height
    normalized_width = width / image_width
    normalized_height = height / image_height
    yolo_line = f"{class_id} {x_center:.8f} {y_center:.8f} {normalized_width:.8f} {normalized_height:.8f}"
    return yolo_line, clipped


def _read_annotations(
    archive_path: Path,
    splits: dict[str, str],
) -> tuple[dict[tuple[str, int], list[str]], dict]:
    labels_by_frame: dict[tuple[str, int], list[str]] = defaultdict(list)
    class_counts: Counter = Counter()
    original_label_counts: Counter = Counter()
    tracks_by_video: dict[str, set[str]] = defaultdict(set)
    rows_by_split: Counter = Counter()
    clipped_boxes = 0
    total_rows = 0
    seen_tracks_per_frame: set[tuple[str, int, str]] = set()

    try:
        with zipfile.ZipFile(archive_path) as archive:
            annotation_files = [
                info for info in archive.infolist()
                if not info.is_dir() and info.filename.lower().endswith(".csv")
            ]
            if len(annotation_files) != EXPECTED_VIDEO_COUNT:
                raise DatasetSetupError(
                    f"annotation.zip tem {len(annotation_files)} CSVs; esperados {EXPECTED_VIDEO_COUNT}."
                )
            annotated_video_ids: set[str] = set()
            for info in annotation_files:
                video_id = PurePosixPath(info.filename).stem
                if video_id not in splits:
                    raise DatasetSetupError(f"Anotação sem vídeo correspondente em data_split.csv: {video_id}")
                annotated_video_ids.add(video_id)
                text = archive.read(info).decode("utf-8-sig")
                reader = csv.DictReader(text.splitlines())
                required_columns = {"track_id", "frame_id", "x", "y", "w", "h", "label"}
                if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
                    raise DatasetSetupError(f"Cabeçalho inesperado em {info.filename}: {reader.fieldnames}")
                for row in reader:
                    total_rows += 1
                    try:
                        frame_id = int(row["frame_id"])
                    except (TypeError, ValueError) as exc:
                        raise DatasetSetupError(f"frame_id inválido em {info.filename}: {row}") from exc
                    if not 1 <= frame_id <= EXPECTED_FRAMES_PER_VIDEO:
                        raise DatasetSetupError(f"frame_id fora de 1..100 em {info.filename}: {frame_id}")
                    track_id = (row.get("track_id") or "").strip()
                    if not track_id:
                        raise DatasetSetupError(f"track_id vazio em {info.filename}, quadro {frame_id}")
                    track_key = (video_id, frame_id, track_id)
                    if track_key in seen_tracks_per_frame:
                        raise DatasetSetupError(f"Track duplicado no mesmo quadro: {track_key}")
                    seen_tracks_per_frame.add(track_key)
                    tracks_by_video[video_id].add(track_id)
                    line, clipped = _normalize_box(row, video_id, frame_id)
                    class_id = int(line.split(maxsplit=1)[0])
                    class_counts[(splits[video_id], class_id)] += 1
                    rows_by_split[splits[video_id]] += 1
                    original_label_counts[(splits[video_id], row["label"])] += 1
                    clipped_boxes += int(clipped)
                    labels_by_frame[(video_id, frame_id)].append(line)
            if annotated_video_ids != set(splits):
                raise DatasetSetupError("Os CSVs de anotação não correspondem aos 910 vídeos da divisão oficial.")
    except (OSError, zipfile.BadZipFile, UnicodeDecodeError) as exc:
        raise DatasetSetupError(f"Não foi possível ler as anotações {archive_path}: {exc}") from exc

    report = {
        "source_annotation_rows": total_rows,
        "unique_video_track_pairs": sum(len(track_ids) for track_ids in tracks_by_video.values()),
        "frames_with_annotated_motorcycles": len(labels_by_frame),
        "boxes_clipped_to_image_boundary": clipped_boxes,
        "boxes_by_split_and_class": {
            split: {
                CLASS_NAMES[class_id]: class_counts[(split, class_id)]
                for class_id in range(len(CLASS_NAMES))
            }
            for split in SPLIT_MAP.values()
        },
        "source_compound_labels_by_split": {
            split: {
                label: original_label_counts[(split, label)]
                for _split, label in sorted(original_label_counts)
                if _split == split
            }
            for split in SPLIT_MAP.values()
        },
        "interpretation": (
            "Cada caixa original representa uma motocicleta. A classe 0 significa que todos os ocupantes "
            "anotados usam capacete; a classe 1 significa que pelo menos um ocupante anotado não usa capacete. "
            "As anotações não incluem caixas individuais de motociclistas nem de capacetes."
        ),
    }
    return labels_by_frame, report


def _relative_frame_paths(video_id: str, split: str, frame_id: int) -> tuple[Path, Path]:
    filename = f"{video_id}_{frame_id:04d}.jpg"
    return Path("images") / split / filename, Path("labels") / split / f"{video_id}_{frame_id:04d}.txt"


def _prepare_part(
    archive_path: Path,
    dataset_root: Path,
    split_by_video: dict[str, str],
    labels_by_frame: dict[tuple[str, int], list[str]],
    expected_part_name: str,
) -> tuple[list[dict], int]:
    image_root = dataset_root / "images"
    label_root = dataset_root / "labels"
    manifest_rows: list[dict] = []
    image_count = 0
    required_extract_bytes = 0
    missing_entries: list[tuple[zipfile.ZipInfo, str, int, Path, Path]] = []

    try:
        with zipfile.ZipFile(archive_path) as archive:
            infos = [info for info in archive.infolist() if not info.is_dir()]
            for info in infos:
                source_parts = PurePosixPath(info.filename)
                if source_parts.is_absolute() or ".." in source_parts.parts:
                    raise DatasetSetupError(f"Caminho inseguro dentro do ZIP: {info.filename}")
                if len(source_parts.parts) != 3 or source_parts.parts[0] != expected_part_name.removesuffix(".zip"):
                    raise DatasetSetupError(f"Caminho inesperado dentro do ZIP: {info.filename}")
                video_id = source_parts.parts[1]
                frame_path = PurePosixPath(source_parts.parts[2])
                if frame_path.suffix.lower() not in {".jpg", ".jpeg"}:
                    continue
                if video_id not in split_by_video:
                    raise DatasetSetupError(f"Imagem sem vídeo correspondente em data_split.csv: {info.filename}")
                try:
                    frame_id = int(frame_path.stem)
                except ValueError as exc:
                    raise DatasetSetupError(f"Nome de quadro inesperado: {info.filename}") from exc
                if not 1 <= frame_id <= EXPECTED_FRAMES_PER_VIDEO:
                    raise DatasetSetupError(f"Quadro fora de 1..100 em {info.filename}")
                split = split_by_video[video_id]
                image_relative, label_relative = _relative_frame_paths(video_id, split, frame_id)
                image_destination = dataset_root / image_relative
                label_destination = dataset_root / label_relative
                if image_destination.exists() and image_destination.stat().st_size != info.file_size:
                    raise DatasetSetupError(f"Imagem parcial/incompatível já existe: {image_destination}")
                if not image_destination.exists():
                    required_extract_bytes += info.file_size
                    missing_entries.append((info, video_id, frame_id, image_destination, label_destination))
                else:
                    missing_entries.append((info, video_id, frame_id, image_destination, label_destination))
                image_count += 1
                manifest_rows.append(
                    {"image_path": image_relative.as_posix(), "video_id": video_id, "split": split}
                )

            if image_count == 0:
                raise DatasetSetupError(f"Não encontrei imagens no arquivo {archive_path.name}.")
            free_bytes = shutil.disk_usage(dataset_root).free
            safety_margin = max(1024**3, int(required_extract_bytes * 0.05))
            if free_bytes < required_extract_bytes + safety_margin:
                raise DatasetSetupError(
                    f"Espaço insuficiente para extrair {archive_path.name}: faltam aproximadamente "
                    f"{(required_extract_bytes + safety_margin - free_bytes) / (1024**3):.1f} GiB. "
                    "Libere espaço e execute novamente; o arquivo validado será mantido."
                )

            copied_bytes = 0
            for index, (info, video_id, frame_id, image_destination, label_destination) in enumerate(missing_entries, 1):
                image_destination.parent.mkdir(parents=True, exist_ok=True)
                label_destination.parent.mkdir(parents=True, exist_ok=True)
                if not image_destination.exists():
                    temporary = image_destination.with_name(image_destination.name + ".partial")
                    with archive.open(info) as source, temporary.open("wb") as destination:
                        shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
                    if temporary.stat().st_size != info.file_size:
                        raise DatasetSetupError(f"Extração incompleta: {info.filename}")
                    os.replace(temporary, image_destination)
                    copied_bytes += info.file_size

                lines = labels_by_frame.get((video_id, frame_id), [])
                label_content = "\n".join(lines)
                if label_content:
                    label_content += "\n"
                if label_destination.exists():
                    if label_destination.read_text(encoding="utf-8") != label_content:
                        raise DatasetSetupError(f"Rótulo preparado diverge das anotações oficiais: {label_destination}")
                else:
                    label_destination.write_text(label_content, encoding="utf-8", newline="\n")
                if index % 2000 == 0 or index == len(missing_entries):
                    print(
                        f"{archive_path.name}: {index}/{len(missing_entries)} quadros preparados "
                        f"({copied_bytes / (1024**3):.2f} GiB copiados)"
                    )
    except (OSError, zipfile.BadZipFile) as exc:
        raise DatasetSetupError(f"Falha ao preparar {archive_path}: {exc}") from exc

    return manifest_rows, image_count


def _write_outputs(
    dataset_root: Path,
    split_by_video: dict[str, str],
    manifest_rows: list[dict],
    image_count: int,
    annotation_report: dict,
    source_records: list[dict],
) -> None:
    if image_count != EXPECTED_FRAME_COUNT or len(manifest_rows) != EXPECTED_FRAME_COUNT:
        raise DatasetSetupError(
            f"Foram preparados {image_count} quadros; o conjunto oficial deve ter {EXPECTED_FRAME_COUNT}."
        )
    unique_images = {row["image_path"] for row in manifest_rows}
    per_video_frames = Counter(row["video_id"] for row in manifest_rows)
    if len(unique_images) != EXPECTED_FRAME_COUNT or len(per_video_frames) != EXPECTED_VIDEO_COUNT:
        raise DatasetSetupError("Há quadros duplicados ou identificadores de vídeo ausentes no conjunto preparado.")
    incomplete_videos = [video_id for video_id, count in per_video_frames.items() if count != EXPECTED_FRAMES_PER_VIDEO]
    if incomplete_videos:
        raise DatasetSetupError(f"Nem todos os vídeos têm 100 quadros: {incomplete_videos[:10]}")
    manifest_path = dataset_root / "manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_rows.sort(key=lambda row: (row["split"], row["video_id"], row["image_path"]))
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image_path", "video_id", "split"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    data_config = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(CLASS_NAMES),
        "names": {index: name for index, name in enumerate(CLASS_NAMES)},
    }
    (dataset_root / "data.yaml").write_text(
        yaml.safe_dump(data_config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    split_video_counts = Counter(split_by_video.values())
    frame_counts = Counter(row["split"] for row in manifest_rows)
    report = {
        "dataset": "HELMET dataset",
        "osf_node": NODE_ID,
        "source_url": DATASET_URL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "license": "CC BY 4.0",
        "official_split_video_counts": dict(split_video_counts),
        "prepared_frame_counts": dict(frame_counts),
        "prepared_frame_count": image_count,
        "image_size_pixels": {"width": IMAGE_SIZE[0], "height": IMAGE_SIZE[1]},
        "classes": {str(index): name for index, name in enumerate(CLASS_NAMES)},
        "class_mapping": {
            "motocicleta_todos_com_capacete": "Caixa de motocicleta; todos os ocupantes anotados usam capacete.",
            "motocicleta_com_ocupante_sem_capacete": (
                "Caixa de motocicleta; pelo menos um ocupante anotado não usa capacete."
            ),
        },
        "official_annotations": annotation_report,
        "source_files": source_records,
        "reproducibility_note": (
            "A partição por vídeo foi mantida da data_split.csv oficial. Os arquivos de imagem não são "
            "sobrescritos nem recompactados; cada arquivo baixado foi verificado por SHA-256 antes da conversão."
        ),
    }
    (dataset_root / "preparation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Raiz do conjunto preparado")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Usa arquivos já existentes em source/downloads, sem acessar o OSF",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Mantém os sete ZIPs de imagens após extração; exige espaço adicional",
    )
    args = parser.parse_args()
    dataset_root = args.root.resolve()
    download_root = dataset_root / "source" / "downloads"
    download_root.mkdir(parents=True, exist_ok=True)

    try:
        if args.prepare_only:
            source_files = {}
            metadata_records = []
            for name in ("annotation.zip", "data_split.csv"):
                local_path = download_root / name
                if not local_path.is_file():
                    raise DatasetSetupError(f"Arquivo necessário ausente para --prepare-only: {local_path}")
                source_files[name] = local_path
                metadata_records.append(
                    {
                        "name": name,
                        "local_path": local_path.as_posix(),
                        "size_bytes": local_path.stat().st_size,
                        "sha256_local": _sha256(local_path),
                    }
                )
            selected = {}
            node = {}
        else:
            node, files = _list_osf_files()
            selected = _select_dataset_files(files)
            source_files = {}
            metadata_records = []
            for name in ("data_split.csv", "annotation.zip"):
                destination = download_root / name
                record = _download_file(selected[name], destination)
                metadata_records.append(record)
                source_files[name] = destination
            (dataset_root / "source" / "osf_node.json").write_text(
                json.dumps(node, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        manifest_file = dataset_root / "source" / "download_manifest.json"
        if manifest_file.is_file():
            source_records = json.loads(manifest_file.read_text(encoding="utf-8")).get("files", [])
        else:
            source_records = metadata_records
        known_source_names = {record.get("name") for record in source_records}
        for record in metadata_records:
            if record.get("name") not in known_source_names:
                source_records.append(record)
                known_source_names.add(record.get("name"))
        manifest_file.parent.mkdir(parents=True, exist_ok=True)
        manifest_file.write_text(
            json.dumps(
                {
                    "dataset_url": DATASET_URL,
                    "osf_node_id": NODE_ID,
                    "files": source_records,
                    "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        split_by_video = _load_video_splits(source_files["data_split.csv"])
        labels_by_frame, annotation_report = _read_annotations(source_files["annotation.zip"], split_by_video)
        state_path = dataset_root / "source" / "preparation_state.json"
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"processed_parts": {}}
        manifest_rows: list[dict] = []
        total_images = 0
        for part_index in range(1, 8):
            part_name = f"part_{part_index}.zip"
            if part_name in state.get("processed_parts", {}):
                print(f"Parte já preparada; mantendo: {part_name}")
                processed_video_ids = state["processed_parts"][part_name].get("video_ids", [])
                if not processed_video_ids:
                    raise DatasetSetupError(f"O estado de preparação não registra os vídeos de {part_name}.")
                for video_id in processed_video_ids:
                    if video_id not in split_by_video:
                        raise DatasetSetupError(f"Vídeo desconhecido no estado de preparação: {video_id}")
                    split = split_by_video[video_id]
                    for frame_id in range(1, EXPECTED_FRAMES_PER_VIDEO + 1):
                        image_relative, label_relative = _relative_frame_paths(video_id, split, frame_id)
                        if not (dataset_root / image_relative).is_file() or not (dataset_root / label_relative).is_file():
                            raise DatasetSetupError(f"Parte marcada como concluída, mas falta {image_relative}")
                        manifest_rows.append(
                            {"image_path": image_relative.as_posix(), "video_id": video_id, "split": split}
                        )
                        total_images += 1
                continue

            part_path = download_root / part_name
            if not args.prepare_only:
                record = _download_file(selected[part_name], part_path)
            elif not part_path.is_file():
                raise DatasetSetupError(f"Arquivo necessário para --prepare-only ausente: {part_path}")
            else:
                record = {
                    "name": part_name,
                    "local_path": part_path.as_posix(),
                    "size_bytes": part_path.stat().st_size,
                    "sha256_local": _sha256(part_path),
                }
            if part_name not in known_source_names:
                record["retained_after_preparation"] = bool(args.keep_archives)
                source_records.append(record)
                known_source_names.add(part_name)
            else:
                for source_record in source_records:
                    if source_record.get("name") == part_name:
                        source_record["retained_after_preparation"] = bool(args.keep_archives)
                        break

            print(f"\nPreparando {part_name}...")
            rows, count = _prepare_part(
                part_path,
                dataset_root,
                split_by_video,
                labels_by_frame,
                part_name,
            )
            manifest_rows.extend(rows)
            total_images += count
            state.setdefault("processed_parts", {})[part_name] = {
                "frame_count": count,
                "video_ids": sorted({row["video_id"] for row in rows}),
                "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
            manifest_file.parent.mkdir(parents=True, exist_ok=True)
            manifest_file.write_text(
                json.dumps(
                    {
                        "dataset_url": DATASET_URL,
                        "osf_node_id": NODE_ID,
                        "files": source_records,
                        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            if not args.keep_archives:
                part_path.unlink()

        if total_images != EXPECTED_FRAME_COUNT:
            raise DatasetSetupError(f"Total preparado: {total_images} quadros; esperado: {EXPECTED_FRAME_COUNT}.")

        _write_outputs(
            dataset_root,
            split_by_video,
            manifest_rows,
            total_images,
            annotation_report,
            source_records,
        )
    except (DatasetSetupError, OSError, ValueError, zipfile.BadZipFile) as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        return 2

    print(f"\nConjunto HELMET preparado em: {dataset_root}")
    print(f"Quadros: {total_images}; caixas de motocicletas anotadas: {annotation_report['source_annotation_rows']}.")
    print("Divisão oficial por vídeo mantida: 636 treino, 92 validação e 182 teste.")
    print("Execute scripts/validate_dataset.py --data data/helmet_osf/data.yaml antes do treino.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
