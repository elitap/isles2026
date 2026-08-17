#!/usr/bin/env python3
"""
inference.py — Two-stage ISLES inference pipeline.

1. Stage 1: 5-fold ensemble on full-FOV T1w images (Dataset053_ISLESFull).
2. Crop:    derive bbox from Stage 1 prediction + margin, crop each input image.
3. Stage 2: 5-fold ensemble on cropped images (Dataset055_ISLESFullStage2).
4. Paste:   place Stage 2 output back into original image space (zeros outside bbox).

Fallback: cases where Stage 1 predicts nothing (empty mask) keep Stage 1 output.

Environment variables nnUNet_raw, nnUNet_preprocessed, and nnUNet_results must be
set before calling this script (same as for nnUNetv2_train / nnUNetv2_predict).

Usage:
    python inference.py \\
        --input   /input \\
        --output  /output \\
        --stage1_dataset  53 \\
        --stage2_dataset  55 \\
        --trainer  nnUNetTrainerMetaBottleneck \\
        --plans    nnUNetResEncUNetLPlans \\
        [--folds 0 1 2 3 4] \\
        [--margin 15] \\
        [--tmp_dir /tmp/isles_infer] \\
        [--workers N]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import SimpleITK as sitk

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

BBox = Tuple[int, int, int, int, int, int]  # (z0,z1, y0,y1, x0,x1)


# ---------------------------------------------------------------------------
# Bounding-box helpers  (mirrors postprocess_bbox.py)
# ---------------------------------------------------------------------------

def _expand_bbox(bb: BBox, shape: Tuple[int, int, int], margin: int) -> BBox:
    z0, z1, y0, y1, x0, x1 = bb
    Z, Y, X = shape
    return (
        max(0, z0 - margin), min(Z - 1, z1 + margin),
        max(0, y0 - margin), min(Y - 1, y1 + margin),
        max(0, x0 - margin), min(X - 1, x1 + margin),
    )


def bbox_from_mask(mask_np: np.ndarray, margin: int) -> Optional[BBox]:
    """Return expanded bbox for the foreground of a binary mask, or None if empty."""
    coords = np.argwhere(mask_np > 0)
    if coords.size == 0:
        return None
    z0, y0, x0 = coords.min(axis=0).tolist()
    z1, y1, x1 = coords.max(axis=0).tolist()
    return _expand_bbox((z0, z1, y0, y1, x0, x1), mask_np.shape, margin)


def crop_sitk(img: sitk.Image, bb: BBox) -> sitk.Image:
    z0, z1, y0, y1, x0, x1 = bb
    size  = [int(x1 - x0 + 1), int(y1 - y0 + 1), int(z1 - z0 + 1)]
    index = [int(x0), int(y0), int(z0)]
    return sitk.Extract(img, size, index)


# ---------------------------------------------------------------------------
# nnUNetv2_predict wrapper
# ---------------------------------------------------------------------------

def _run_nnunet_predict(
    dataset_id: int,
    input_dir: Path,
    output_dir: Path,
    trainer: str,
    plans: str,
    folds: List[int],
    config: str = "3d_fullres",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "nnUNetv2_predict",
        "-d", str(dataset_id),
        "-i", str(input_dir),
        "-o", str(output_dir),
        "-tr", trainer,
        "-p", plans,
        "-c", config,
        "-f", *[str(f) for f in folds],
    ]
    log.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=True, text=True)
    if result.returncode != 0:
        log.error("nnUNetv2_predict failed with exit code %d", result.returncode)
        sys.exit(result.returncode)


# ---------------------------------------------------------------------------
# Per-case crop worker
# ---------------------------------------------------------------------------

def _crop_case(
    case_id: str,
    input_dir: Path,
    stage1_dir: Path,
    cropped_dir: Path,
    bbox_json_dir: Path,
    margin: int,
) -> Optional[str]:
    """
    Crop all input channels for case_id to the Stage 1 bbox.
    Returns case_id if cropped, None if Stage 1 prediction was empty (fallback).
    Writes <cropped_dir>/<case_id>_0000.nii.gz and a bbox JSON sidecar.
    """
    pred_path = stage1_dir / f"{case_id}.nii.gz"
    if not pred_path.exists():
        log.warning("%s — Stage 1 prediction not found; skipping crop", case_id)
        return None

    pred_np = sitk.GetArrayFromImage(sitk.ReadImage(str(pred_path)))
    bb = bbox_from_mask(pred_np, margin)
    if bb is None:
        log.warning("%s — Stage 1 prediction empty; will use Stage 1 fallback", case_id)
        return None

    z0, z1, y0, y1, x0, x1 = bb
    channel_paths = sorted(input_dir.glob(f"{case_id}_*.nii.gz"))
    if not channel_paths:
        log.error("%s — no input channels found in %s", case_id, input_dir)
        return None

    for ch_path in channel_paths:
        cropped = crop_sitk(sitk.ReadImage(str(ch_path)), bb)
        sitk.WriteImage(cropped, str(cropped_dir / ch_path.name))

    bbox_json_dir.mkdir(parents=True, exist_ok=True)
    (bbox_json_dir / f"{case_id}_bbox.json").write_text(
        json.dumps({
            "case_id": case_id,
            "original_shape": list(pred_np.shape),
            "bbox": {"z0": z0, "z1": z1, "y0": y0, "y1": y1, "x0": x0, "x1": x1},
        }, indent=2)
    )
    return case_id


# ---------------------------------------------------------------------------
# Per-case paste-back worker
# ---------------------------------------------------------------------------

def _paste_case(
    case_id: str,
    stage2_dir: Path,
    bbox_json_dir: Path,
    stage1_dir: Path,
    final_dir: Path,
) -> str:
    """
    Paste Stage 2 prediction back into original image space.
    Falls back to Stage 1 prediction if no bbox JSON exists (Stage 1 was empty).
    """
    bbox_json = bbox_json_dir / f"{case_id}_bbox.json"
    stage2_pred = stage2_dir / f"{case_id}.nii.gz"
    stage1_pred = stage1_dir / f"{case_id}.nii.gz"
    out_path = final_dir / f"{case_id}.nii.gz"

    if not bbox_json.exists() or not stage2_pred.exists():
        # Stage 1 predicted nothing — copy Stage 1 output as fallback
        shutil.copy2(str(stage1_pred), str(out_path))
        return f"{case_id} [fallback→stage1]"

    meta = json.loads(bbox_json.read_text())
    bb = meta["bbox"]
    z0, z1 = bb["z0"], bb["z1"]
    y0, y1 = bb["y0"], bb["y1"]
    x0, x1 = bb["x0"], bb["x1"]
    orig_shape = tuple(meta["original_shape"])  # (Z, Y, X)

    stage2_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(stage2_pred)))

    # Verify crop dimensions match bbox (nnUNet output is in input space)
    expected = (z1 - z0 + 1, y1 - y0 + 1, x1 - x0 + 1)
    if stage2_arr.shape != expected:
        log.warning(
            "%s — Stage 2 shape %s != expected %s; resampling paste region",
            case_id, stage2_arr.shape, expected,
        )

    result_arr = np.zeros(orig_shape, dtype=np.uint8)
    # Clamp paste region to array bounds (safeguard against off-by-one)
    pz1 = min(z0 + stage2_arr.shape[0], orig_shape[0])
    py1 = min(y0 + stage2_arr.shape[1], orig_shape[1])
    px1 = min(x0 + stage2_arr.shape[2], orig_shape[2])
    result_arr[z0:pz1, y0:py1, x0:px1] = stage2_arr[:pz1-z0, :py1-y0, :px1-x0]

    ref = sitk.ReadImage(str(stage1_pred))
    out_img = sitk.GetImageFromArray(result_arr)
    out_img.CopyInformation(ref)
    sitk.WriteImage(out_img, str(out_path))
    return case_id


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",            required=True, type=Path,
                   help="Directory of test images (*_0000.nii.gz)")
    p.add_argument("--output",           required=True, type=Path,
                   help="Directory for final predictions")
    p.add_argument("--stage1_dataset",   required=True, type=int,
                   help="nnUNet dataset ID for Stage 1 (e.g. 53)")
    p.add_argument("--stage2_dataset",   required=True, type=int,
                   help="nnUNet dataset ID for Stage 2 (e.g. 55)")
    p.add_argument("--trainer",          default="nnUNetTrainerMetaBottleneck")
    p.add_argument("--plans",            default="nnUNetResEncUNetLPlans")
    p.add_argument("--config",           default="3d_fullres")
    p.add_argument("--folds",            nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--margin",           type=int, default=15,
                   help="Voxel margin around Stage 1 bbox (default 15)")
    p.add_argument("--tmp_dir",          type=Path, default=None,
                   help="Temp directory for intermediate files (default: system tmp)")
    p.add_argument("--workers",          type=int, default=max(1, os.cpu_count() // 2))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()

    args.output.mkdir(parents=True, exist_ok=True)

    own_tmp = args.tmp_dir is None
    tmp_root = Path(tempfile.mkdtemp(prefix="isles_infer_")) if own_tmp else args.tmp_dir
    tmp_root.mkdir(parents=True, exist_ok=True)
    log.info("Tmp dir: %s", tmp_root)

    stage1_out   = tmp_root / "stage1_predictions"
    cropped_dir  = tmp_root / "stage2_inputs"
    bbox_dir     = tmp_root / "bbox_json"
    stage2_out   = tmp_root / "stage2_predictions"

    cropped_dir.mkdir(parents=True, exist_ok=True)

    try:
        # ------------------------------------------------------------------ #
        # Stage 1 — full-FOV ensemble
        # ------------------------------------------------------------------ #
        log.info("=== Stage 1 inference (dataset %d, folds %s) ===",
                 args.stage1_dataset, args.folds)
        _run_nnunet_predict(
            args.stage1_dataset, args.input, stage1_out,
            args.trainer, args.plans, args.folds, args.config,
        )

        case_ids = sorted(
            p.name.replace(".nii.gz", "")
            for p in stage1_out.glob("*.nii.gz")
            if not p.name.startswith(".")
        )
        log.info("Stage 1 produced %d predictions", len(case_ids))

        # ------------------------------------------------------------------ #
        # Crop — derive bbox from Stage 1, crop input images
        # ------------------------------------------------------------------ #
        log.info("=== Cropping inputs to Stage 1 bboxes (margin=%d) ===", args.margin)
        cropped_ids: List[str] = []
        fallback_ids: List[str] = []

        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    _crop_case, cid,
                    args.input, stage1_out, cropped_dir, bbox_dir, args.margin,
                ): cid
                for cid in case_ids
            }
            for fut in as_completed(futures):
                result = fut.result()
                if result is not None:
                    cropped_ids.append(result)
                else:
                    fallback_ids.append(futures[fut])

        log.info("Cropped %d cases; %d fallback to Stage 1 (empty prediction)",
                 len(cropped_ids), len(fallback_ids))

        # ------------------------------------------------------------------ #
        # Stage 2 — ensemble on crops
        # ------------------------------------------------------------------ #
        if cropped_ids:
            log.info("=== Stage 2 inference (dataset %d, folds %s) ===",
                     args.stage2_dataset, args.folds)
            _run_nnunet_predict(
                args.stage2_dataset, cropped_dir, stage2_out,
                args.trainer, args.plans, args.folds, args.config,
            )
        else:
            log.warning("No cases to run through Stage 2; all fall back to Stage 1.")

        # ------------------------------------------------------------------ #
        # Paste back — Stage 2 → original space
        # ------------------------------------------------------------------ #
        log.info("=== Pasting Stage 2 predictions back into original space ===")
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    _paste_case, cid,
                    stage2_out, bbox_dir, stage1_out, args.output,
                ): cid
                for cid in case_ids
            }
            for fut in as_completed(futures):
                log.debug("Done: %s", fut.result())

        n_final = len(list(args.output.glob("*.nii.gz")))
        log.info(
            "=== Done — %d final predictions written to %s  (%.1f s total) ===",
            n_final, args.output, time.time() - t0,
        )
        if fallback_ids:
            log.info("Fallback cases (Stage 1 output used): %s",
                     ", ".join(sorted(fallback_ids)))

    finally:
        if own_tmp:
            shutil.rmtree(tmp_root, ignore_errors=True)
            log.info("Cleaned up tmp dir %s", tmp_root)


if __name__ == "__main__":
    main()
