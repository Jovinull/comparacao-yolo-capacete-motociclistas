"""Collect detections and calculate class-wise operating-point metrics."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from validate_dataset import ManifestRecord


Box = tuple[float, float, float, float]
Detection = tuple[int, float, Box]


@dataclass(frozen=True)
class ImageDetections:
    record: ManifestRecord
    ground_truth: dict[int, tuple[Box, ...]]
    predictions: tuple[Detection, ...]


def collect_detections(
    model,
    records: list[ManifestRecord],
    num_classes: int,
    confidence_floor: float,
    prediction_iou: float,
    imgsz: int,
    batch: int,
    device: str,
    max_det: int,
) -> list[ImageDetections]:
    """Run one low-confidence prediction pass and retain boxes for thresholding."""
    collected: list[ImageDetections] = []
    for start in range(0, len(records), batch):
        chunk = records[start : start + batch]
        results = model.predict(
            source=[str(record.image_path) for record in chunk],
            imgsz=imgsz,
            batch=batch,
            device=device,
            conf=confidence_floor,
            iou=prediction_iou,
            max_det=max_det,
            verbose=False,
            save=False,
        )
        if len(results) != len(chunk):
            raise RuntimeError("O modelo retornou uma quantidade inesperada de predições")

        for record, result in zip(chunk, results):
            image_height, image_width = result.orig_shape
            ground_truth: dict[int, list[Box]] = {class_id: [] for class_id in range(num_classes)}
            for line in record.label_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                class_id, x_center, y_center, width, height = line.split()
                class_index = int(class_id)
                ground_truth[class_index].append(
                    (
                        (float(x_center) - float(width) / 2) * image_width,
                        (float(y_center) - float(height) / 2) * image_height,
                        (float(x_center) + float(width) / 2) * image_width,
                        (float(y_center) + float(height) / 2) * image_height,
                    )
                )

            predictions: list[Detection] = []
            boxes = result.boxes
            if boxes is not None and len(boxes):
                xyxy = boxes.xyxy.cpu().tolist()
                classes = boxes.cls.cpu().tolist()
                scores = boxes.conf.cpu().tolist()
                predictions = [
                    (
                        int(classes[index]),
                        float(scores[index]),
                        tuple(float(value) for value in xyxy[index]),
                    )
                    for index in range(len(xyxy))
                ]
            collected.append(
                ImageDetections(
                    record=record,
                    ground_truth={key: tuple(value) for key, value in ground_truth.items()},
                    predictions=tuple(predictions),
                )
            )
    return collected


def _box_iou(first: Box, second: Box) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _counts_for_class(
    frames: list[ImageDetections], class_id: int, confidence: float, match_iou: float
) -> tuple[int, int, int, dict[str, dict[str, int]]]:
    true_positives = false_positives = false_negatives = 0
    by_video: dict[str, dict[str, int]] = defaultdict(
        lambda: {"true_positives": 0, "false_positives": 0, "false_negatives": 0, "ground_truth": 0, "frames": 0}
    )

    for frame in frames:
        truth = frame.ground_truth[class_id]
        candidates = sorted(
            (
                (score, box)
                for predicted_class, score, box in frame.predictions
                if predicted_class == class_id and score >= confidence
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        matched: set[int] = set()
        frame_tp = frame_fp = 0
        for _, predicted_box in candidates:
            best_index = None
            best_iou = 0.0
            for truth_index, truth_box in enumerate(truth):
                if truth_index in matched:
                    continue
                overlap = _box_iou(predicted_box, truth_box)
                if overlap > best_iou:
                    best_index, best_iou = truth_index, overlap
            if best_index is not None and best_iou >= match_iou:
                matched.add(best_index)
                frame_tp += 1
            else:
                frame_fp += 1
        frame_fn = len(truth) - frame_tp
        true_positives += frame_tp
        false_positives += frame_fp
        false_negatives += frame_fn
        row = by_video[frame.record.video_id]
        row["true_positives"] += frame_tp
        row["false_positives"] += frame_fp
        row["false_negatives"] += frame_fn
        row["ground_truth"] += len(truth)
        row["frames"] += 1

    return true_positives, false_positives, false_negatives, dict(by_video)


def _confusion_matrix(
    frames: list[ImageDetections], num_classes: int, confidence: float, match_iou: float
) -> list[list[int]]:
    """Build a detection confusion matrix; the final row/column represent background."""
    matrix = [[0 for _ in range(num_classes + 1)] for _ in range(num_classes + 1)]
    for frame in frames:
        truth = [
            (class_id, box)
            for class_id, boxes in frame.ground_truth.items()
            for box in boxes
        ]
        predictions = sorted(
            (
                (class_id, score, box)
                for class_id, score, box in frame.predictions
                if score >= confidence
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        matched: set[int] = set()
        for predicted_class, _, predicted_box in predictions:
            best_index = None
            best_iou = 0.0
            for truth_index, (_, truth_box) in enumerate(truth):
                if truth_index in matched:
                    continue
                overlap = _box_iou(predicted_box, truth_box)
                if overlap > best_iou:
                    best_index, best_iou = truth_index, overlap
            if best_index is not None and best_iou >= match_iou:
                matched.add(best_index)
                true_class = truth[best_index][0]
                matrix[true_class][predicted_class] += 1
            else:
                matrix[num_classes][predicted_class] += 1
        for truth_index, (true_class, _) in enumerate(truth):
            if truth_index not in matched:
                matrix[true_class][num_classes] += 1
    return matrix


def calculate_detection_metrics(
    frames: list[ImageDetections],
    class_names: tuple[str, ...],
    confidence: float,
    match_iou: float,
) -> dict:
    """Calculate count-based metrics and FP/FN rates for a fixed threshold."""
    per_class = {}
    video_data: dict[str, dict] = defaultdict(dict)
    precision_values = []
    recall_values = []
    f1_values = []

    for class_id, class_name in enumerate(class_names):
        tp, fp, fn, by_video = _counts_for_class(frames, class_id, confidence, match_iou)
        ground_truth_count = tp + fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / ground_truth_count if ground_truth_count else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        precision_values.append(precision)
        recall_values.append(recall)
        per_class[class_name] = {
            "class_id": class_id,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "ground_truth_objects": ground_truth_count,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_positives_per_100_frames": 100 * fp / len(frames) if frames else None,
            "false_negatives_per_100_objects": 100 * fn / ground_truth_count if ground_truth_count else None,
        }
        for video_id, counts in by_video.items():
            video_data[video_id][class_name] = {
                **counts,
                "recall": counts["true_positives"] / counts["ground_truth"]
                if counts["ground_truth"]
                else None,
            }

    return {
        "confidence_threshold": confidence,
        "match_iou": match_iou,
        "frames": len(frames),
        "macro_precision": sum(precision_values) / len(precision_values) if precision_values else None,
        "macro_recall": sum(recall_values) / len(recall_values) if recall_values else None,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else None,
        "per_class": per_class,
        "confusion_matrix": {
            "labels": [*class_names, "background"],
            "counts": _confusion_matrix(frames, len(class_names), confidence, match_iou),
        },
        "per_video": dict(video_data),
    }


def select_confidence_threshold(
    validation_frames: list[ImageDetections],
    class_names: tuple[str, ...],
    confidence_grid: list[float],
    match_iou: float,
) -> tuple[float, list[dict]]:
    """Choose the highest threshold among those tied for best validation macro-F1."""
    if not confidence_grid:
        raise ValueError("A grade de limiares de confiança está vazia")
    scores = []
    for threshold in sorted(set(float(value) for value in confidence_grid)):
        if not 0 < threshold <= 1:
            raise ValueError("Os limiares de confiança devem estar entre 0 e 1")
        metrics = calculate_detection_metrics(validation_frames, class_names, threshold, match_iou)
        scores.append({"confidence_threshold": threshold, "macro_f1": metrics["macro_f1"]})
    best_score = max(row["macro_f1"] for row in scores)
    selected = max(row["confidence_threshold"] for row in scores if row["macro_f1"] == best_score)
    return selected, scores
