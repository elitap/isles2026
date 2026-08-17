#!/usr/bin/env python3
"""
crop_coverage.py — For each fold-0 validation case, derive the Stage 2 bounding
box from the Stage 1 (MetaBottleneck) prediction using the exact same
label_bbox/expand_bbox logic as postprocess_bbox.py, then measure what
percentage of the *ground-truth* lesion voxels fall inside that box.

This answers: if we crop at inference using Stage 1's own prediction (not the
GT, which postprocess_bbox.py's GT-crop ablation uses), how much of the true
lesion does the crop actually retain? A case where Stage 1 predicts a box
that's off-target (non-empty prediction, zero overlap with GT) will show
genuine low/zero coverage — a real crop-induced loss. A case where Stage 1
predicts nothing at all is different: per AGENTS.md, Stage 2 skips cases with
an empty Stage 1 prediction and falls back to the uncropped Stage 1 output —
no crop is ever applied, so nothing is lost to cropping and the full volume
remains available downstream to "find the lesion again." Those are logged as
warnings (DSC=0 either way) but scored as 100% coverage, not 0%, since 0%
would misrepresent them as a cropping failure when they're a segmentation
failure with no cropping involved.

Usage:
    python isles/scripts/crop_coverage.py \
        --pred <stage1_pred_dir> --gt <gt_segmentations_dir> \
        --output <output_dir> [--margin 15] [--label 1]
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).parent))
from postprocess_bbox import label_bbox, expand_bbox, BBox  # noqa: E402  (reuse production bbox logic)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def crop_mask(bb: BBox) -> tuple[slice, slice, slice]:
    z0, z1, y0, y1, x0, x1 = bb
    return slice(z0, z1 + 1), slice(y0, y1 + 1), slice(x0, x1 + 1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pred", required=True, type=Path, help="Stage 1 prediction dir (*.nii.gz)")
    p.add_argument("--gt", required=True, type=Path, help="Ground-truth segmentation dir (*.nii.gz)")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--margin", type=int, default=15, help="Voxel margin around bbox (default 15, matches AGENTS.md Step 4b)")
    p.add_argument("--label", type=int, default=1, help="Foreground label (default 1 = lesion)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    pred_files = sorted(args.pred.glob("*.nii.gz"))
    log.info("Found %d prediction files in %s", len(pred_files), args.pred)

    rows = []
    for pred_path in pred_files:
        case_id = pred_path.name.replace(".nii.gz", "")
        gt_path = args.gt / pred_path.name
        if not gt_path.exists():
            log.warning("%s — no GT found, skipping", case_id)
            continue

        pred_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(pred_path))).astype(np.int16)
        gt_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(gt_path))).astype(np.int16)

        gt_mask = gt_arr == args.label
        gt_voxels = int(gt_mask.sum())
        pred_mask = pred_arr == args.label
        empty_prediction = not pred_mask.any()
        gt_empty = gt_voxels == 0

        if gt_empty:
            # Nothing to capture — coverage is undefined, not zero.
            rows.append({
                "case_id": case_id, "gt_voxels": 0, "gt_voxels_in_crop": 0,
                "coverage_pct": "", "empty_prediction": empty_prediction, "zero_dsc": False, "gt_empty": True,
                "bbox": "",
            })
            continue

        overlap_voxels = int((pred_mask & gt_mask).sum())
        zero_dsc = overlap_voxels == 0  # true for both empty and non-empty-but-mislocated predictions

        if empty_prediction:
            # Stage 1 predicted nothing anywhere -> no bbox can be formed, so
            # no crop is ever applied — Stage 2 falls back to the uncropped
            # Stage 1 output (AGENTS.md). The full volume is still available
            # downstream, so this is scored as 100% coverage, not 0%.
            log.warning(
                "%s — Stage-1 prediction is empty (DSC=0); no crop applied, falls back to "
                "uncropped volume -> scored as 100%% coverage, not a cropping failure.",
                case_id,
            )
            rows.append({
                "case_id": case_id, "gt_voxels": gt_voxels, "gt_voxels_in_crop": gt_voxels,
                "coverage_pct": 100.0, "empty_prediction": True, "zero_dsc": True, "gt_empty": False,
                "bbox": "",
            })
            continue

        bb = expand_bbox(label_bbox(pred_arr, args.label), pred_arr.shape, args.margin)
        sl = crop_mask(bb)
        gt_in_crop = int(gt_mask[sl].sum())
        coverage_pct = 100.0 * gt_in_crop / gt_voxels

        if zero_dsc:
            log.warning(
                "%s — Stage-1 prediction is non-empty but has zero voxel overlap with GT (DSC=0); "
                "bbox crop still captures %.1f%% of GT by spatial proximity.",
                case_id, coverage_pct,
            )

        rows.append({
            "case_id": case_id, "gt_voxels": gt_voxels, "gt_voxels_in_crop": gt_in_crop,
            "coverage_pct": round(coverage_pct, 2), "empty_prediction": False, "zero_dsc": zero_dsc, "gt_empty": False,
            "bbox": json.dumps({"z0": bb[0], "z1": bb[1], "y0": bb[2], "y1": bb[3], "x0": bb[4], "x1": bb[5]}),
        })

    rows.sort(key=lambda r: r["case_id"])
    csv_path = args.output / "crop_coverage.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    valid = [r["coverage_pct"] for r in rows if r["coverage_pct"] != "" and not r["gt_empty"]]
    n_empty_pred = sum(1 for r in rows if r["empty_prediction"])
    n_zero_dsc = sum(1 for r in rows if r["zero_dsc"])
    n_zero_dsc_mislocated = n_zero_dsc - n_empty_pred  # non-empty prediction, still zero overlap
    n_gt_empty = sum(1 for r in rows if r["gt_empty"])
    n_full_100 = sum(1 for v in valid if v >= 99.995)

    summary = {
        "n_cases_total": len(rows),
        "n_gt_empty_excluded": n_gt_empty,
        "n_empty_prediction_fallback_100pct": n_empty_pred,
        "n_zero_dsc_total": n_zero_dsc,
        "n_zero_dsc_mislocated_real_loss": n_zero_dsc_mislocated,
        "n_evaluated_for_coverage": len(valid),
        "coverage_pct_mean": float(np.mean(valid)) if valid else None,
        "coverage_pct_std": float(np.std(valid)) if valid else None,
        "coverage_pct_median": float(np.median(valid)) if valid else None,
        "coverage_pct_min": float(np.min(valid)) if valid else None,
        "coverage_pct_max": float(np.max(valid)) if valid else None,
        "n_cases_100pct_coverage": n_full_100,
        "margin_voxels": args.margin,
    }
    (args.output / "crop_coverage_summary.json").write_text(json.dumps(summary, indent=2))

    log.info(
        "n=%d evaluated (excluded %d GT-empty); %d cases had DSC=0 (%d empty-prediction "
        "-> scored 100%%, %d non-empty-but-mislocated -> real crop loss)",
        len(valid), n_gt_empty, n_zero_dsc, n_empty_pred, n_zero_dsc_mislocated,
    )
    log.info("coverage_pct: mean=%.2f std=%.2f median=%.2f min=%.2f max=%.2f",
              summary["coverage_pct_mean"], summary["coverage_pct_std"],
              summary["coverage_pct_median"], summary["coverage_pct_min"], summary["coverage_pct_max"])
    log.info("Wrote %s and crop_coverage_summary.json", csv_path)


if __name__ == "__main__":
    main()
