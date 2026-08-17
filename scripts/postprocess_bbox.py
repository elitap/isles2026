#!/usr/bin/env python3
"""
postprocess_bbox.py — Generate bounding-box-cropped dataset from Stage 1 predictions or GT labels.

By default (--bbox_source predictions) bounding boxes are derived from Stage 1
predictions — cases with empty predictions are skipped and fall back to Stage 1
output at inference.

With --bbox_source gt, bounding boxes are derived from ground-truth labels
instead. Use this for the GT-crop ablation study (Dataset054_ISLESGTCrop) to
measure the upper-bound benefit of bbox cropping independently of Stage 1
quality. Cases with empty GT masks (no lesion) are skipped.

Important: after nnUNetv2_preprocess on the new dataset, copy
  preprocessed/Dataset053_ISLESFull/splits_final.json
into
  preprocessed/Dataset054_ISLESGTCrop/splits_final.json
so that the CV folds are identical to the Stage 1 training.

Usage:
    # GT-crop ablation
    python postprocess_bbox.py --bbox_source gt \
        --images <imagesTr_dir> --labels <labelsTr_dir> \
        --output <Dataset054_dir> [--margin 15]

    # Stage 2 (prediction-based, default)
    python postprocess_bbox.py --bbox_source predictions \
        --predictions <stage1_pred_dir> --images <imagesTr_dir> \
        --labels <labelsTr_dir> --output <Dataset055_dir> [--margin 15]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bounding-box helpers
# ---------------------------------------------------------------------------

BBox = Tuple[int, int, int, int, int, int]  # (z0,z1, y0,y1, x0,x1)


def label_bbox(mask: np.ndarray, label: int) -> BBox:
    """Tight axis-aligned bounding box for a single label (voxel coords)."""
    coords = np.argwhere(mask == label)
    assert coords.size, f"Label {label} not found in mask"
    z0, y0, x0 = coords.min(axis=0).tolist()
    z1, y1, x1 = coords.max(axis=0).tolist()
    return z0, z1, y0, y1, x0, x1


def expand_bbox(bb: BBox, shape: Tuple[int, int, int], margin: int) -> BBox:
    """Add margin, clamped to image boundary."""
    z0, z1, y0, y1, x0, x1 = bb
    Z, Y, X = shape
    return (
        max(0, z0 - margin), min(Z - 1, z1 + margin),
        max(0, y0 - margin), min(Y - 1, y1 + margin),
        max(0, x0 - margin), min(X - 1, x1 + margin),
    )


def bbox_union(a: BBox, b: BBox) -> BBox:
    return (
        min(a[0], b[0]), max(a[1], b[1]),
        min(a[2], b[2]), max(a[3], b[3]),
        min(a[4], b[4]), max(a[5], b[5]),
    )


def bbox_touches(a: BBox, b: BBox) -> bool:
    """True when two bounding boxes share or overlap at least one voxel."""
    return (
        a[0] <= b[1] and b[0] <= a[1] and
        a[2] <= b[3] and b[2] <= a[3] and
        a[4] <= b[5] and b[4] <= a[5]
    )


def iterative_bbox_expansion(
    mask: np.ndarray,
    labels: List[int],
    margin: int,
) -> Tuple[Dict[int, BBox], int]:
    """
    Expand per-label bboxes until no bbox absorbs a bordering label.
    Returns (per-label BBox dict, number of expansion iterations).
    """
    shape = mask.shape
    bboxes: Dict[int, BBox] = {
        lbl: expand_bbox(label_bbox(mask, lbl), shape, margin) for lbl in labels
    }
    n_iter = 0
    changed = True
    while changed:
        changed = False
        n_iter += 1
        for lbl in labels:
            for other in labels:
                if lbl == other:
                    continue
                if bbox_touches(bboxes[lbl], bboxes[other]):
                    merged = bbox_union(bboxes[lbl], bboxes[other])
                    if merged != bboxes[lbl]:
                        bboxes[lbl] = merged
                        changed = True
    return bboxes, n_iter


def merged_bbox(bboxes: Dict[int, BBox]) -> BBox:
    bb = list(bboxes.values())
    result = bb[0]
    for b in bb[1:]:
        result = bbox_union(result, b)
    return result


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def crop_sitk(img: sitk.Image, bb: BBox) -> sitk.Image:
    z0, z1, y0, y1, x0, x1 = bb
    # SimpleITK uses (x, y, z) order for Extract
    size = [int(x1 - x0 + 1), int(y1 - y0 + 1), int(z1 - z0 + 1)]
    index = [int(x0), int(y0), int(z0)]
    return sitk.Extract(img, size, index)


def _derive_dataset_name(output: Path) -> str:
    return output.name if output.name.startswith("Dataset") else output.name


# ---------------------------------------------------------------------------
# Per-case worker (runs in subprocess)
# ---------------------------------------------------------------------------

def _process_case(
    source_path: Path,
    images_dir: Path,
    labels_dir: Path,
    images_out: Path,
    labels_out: Path,
    output_root: Path,
    bbox_source: str,
    margin: int,
) -> Optional[Tuple[dict, dict]]:
    """
    Returns (sidecar_dict, training_entry_dict) or None if the case is skipped.
    Writes cropped image/label files and the per-case *_bbox.json sidecar.
    """
    case_id = source_path.name.replace(".nii.gz", "")

    source_sitk = sitk.ReadImage(str(source_path))
    source_np = sitk.GetArrayFromImage(source_sitk)
    labels = [int(l) for l in np.unique(source_np) if l != 0]

    if not labels:
        return None  # caller logs the skip

    bboxes, n_iter = iterative_bbox_expansion(source_np, labels, margin)
    bb = merged_bbox(bboxes)

    z0, z1, y0, y1, x0, x1 = bb
    crop_np = source_np[z0:z1+1, y0:y1+1, x0:x1+1]
    assert all(lbl in np.unique(crop_np) for lbl in labels), \
        f"Label(s) missing after crop for {case_id}"

    img_candidates = sorted(images_dir.glob(f"{case_id}*.nii.gz"))
    assert img_candidates, f"No image found for {case_id} in {images_dir}"

    ref_img = None
    for img_path in img_candidates:
        cropped_img = crop_sitk(sitk.ReadImage(str(img_path)), bb)
        if ref_img is None:
            ref_img = cropped_img  # geometry reference for label
        sitk.WriteImage(cropped_img, str(images_out / img_path.name))

    # Copy metadata sidecar if present (needed by nnUNetTrainerMetaBottleneck).
    # prepare_dataset.py always writes it one channel past the last image
    # channel, e.g. {case_id}_0001.json alongside {case_id}_0000.nii.gz — it
    # is per-case, not per-image-channel, so this is independent of how many
    # image candidates were found above.
    sidecar_src = images_dir / f"{case_id}_{len(img_candidates):04d}.json"
    if sidecar_src.exists():
        shutil.copy2(sidecar_src, images_out / sidecar_src.name)

    gt_path = labels_dir / f"{case_id}.nii.gz"
    assert gt_path.exists(), f"Ground-truth label not found: {gt_path}"
    cropped_gt = crop_sitk(sitk.ReadImage(str(gt_path)), bb)
    # Ensure exact geometry match — original image/seg may differ by floating-point noise
    cropped_gt.CopyInformation(ref_img)
    sitk.WriteImage(cropped_gt, str(labels_out / f"{case_id}.nii.gz"))

    channel_images = [f"./imagesTr/{p.name}" for p in img_candidates]
    training_entry = {
        "image": channel_images[0] if len(channel_images) == 1 else channel_images,
        "label": f"./labelsTr/{case_id}.nii.gz",
    }
    sidecar = {
        "case_id": case_id,
        "bbox_source": bbox_source,
        "original_image": str(img_candidates[0]),
        "original_shape": list(source_np.shape),
        "bbox": {"z0": z0, "z1": z1, "y0": y0, "y1": y1, "x0": x0, "x1": x1},
        "labels": labels,
        "expansion_iterations": n_iter,
    }
    (output_root / f"{case_id}_bbox.json").write_text(json.dumps(sidecar, indent=2))
    return sidecar, training_entry


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bbox_source", choices=["predictions", "gt"], default="predictions",
                   help="Source for bounding-box computation: Stage 1 predictions or GT labels")
    p.add_argument("--predictions", type=Path, default=None,
                   help="Stage 1 predictions dir (required when --bbox_source predictions)")
    p.add_argument("--images", required=True, type=Path,
                   help="Original imagesTr dir (case_id_0000.nii.gz)")
    p.add_argument("--labels", required=True, type=Path,
                   help="Ground-truth labelsTr dir (case_id.nii.gz)")
    p.add_argument("--output", required=True, type=Path,
                   help="Output dataset root (Dataset054_… or Dataset055_…)")
    p.add_argument("--dataset_id", type=int, default=None,
                   help="nnUNet dataset ID (optional, inferred from output name if omitted)")
    p.add_argument("--margin", type=int, default=10,
                   help="Voxel margin around bbox (default 10)")
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() // 2),
                   help="Parallel worker processes (default: half of CPU count)")
    args = p.parse_args()
    if args.bbox_source == "predictions" and args.predictions is None:
        p.error("--predictions is required when --bbox_source predictions")
    return args


def main() -> None:
    args = parse_args()
    t0 = time.time()
    log.info("Bounding-box post-processing  bbox_source=%s  margin=%d  workers=%d",
             args.bbox_source, args.margin, args.workers)

    images_out = args.output / "imagesTr"
    labels_out = args.output / "labelsTr"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    if args.bbox_source == "predictions":
        source_files = sorted(args.predictions.glob("*.nii.gz"))
        log.info("Found %d prediction files", len(source_files))
    else:
        source_files = sorted(args.labels.glob("*.nii.gz"))
        log.info("Found %d GT label files (GT-crop ablation)", len(source_files))

    task_args = [
        (f, args.images, args.labels, images_out, labels_out,
         args.output, args.bbox_source, args.margin)
        for f in source_files
    ]

    sidecars: List[dict] = []
    training_entries: List[dict] = []
    skipped_empty: List[str] = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_process_case, *t): t[0] for t in task_args}
        done = 0
        for fut in as_completed(futures):
            source_path = futures[fut]
            result = fut.result()
            done += 1
            if result is None:
                case_id = source_path.name.replace(".nii.gz", "")
                skipped_empty.append(case_id)
                if args.bbox_source == "predictions":
                    log.warning("%s — no foreground in prediction; skipping", case_id)
                else:
                    log.warning("%s — empty GT mask; skipping", case_id)
            else:
                sidecar, entry = result
                sidecars.append(sidecar)
                training_entries.append(entry)
            if done % 50 == 0:
                log.info("  %d / %d done", done, len(task_args))

    # Sort for deterministic dataset.json
    sidecars.sort(key=lambda s: s["case_id"])
    training_entries.sort(key=lambda e: e["label"])

    log.info("Processed %d cases, skipped %d (no foreground) in %.1f s",
             len(sidecars), len(skipped_empty), time.time() - t0)
    if skipped_empty:
        log.warning("Skipped: %s", ", ".join(sorted(skipped_empty)))

    dataset_name = _derive_dataset_name(args.output)
    dataset_json = {
        "name": dataset_name,
        "description": "ISLES 2026 ischemic stroke lesion segmentation — bbox-cropped",
        "reference": "https://isles-26.grand-challenge.org/",
        "licence": "see challenge website",
        "release": "1.0",
        "channel_names": {"0": "T1w"},
        "labels": {"background": 0, "lesion": 1},
        "numTraining": len(sidecars),
        "file_ending": ".nii.gz",
        "training": training_entries,
    }
    (args.output / "dataset.json").write_text(json.dumps(dataset_json, indent=2))
    log.info("Wrote dataset.json with %d training cases", len(sidecars))


if __name__ == "__main__":
    main()
