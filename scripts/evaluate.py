#!/usr/bin/env python3
"""
evaluate.py — Compute challenge metrics: DSC, HD, HD95, ASSD, RVE, NSD (1 mm).

Usage:
    python evaluate.py --pred <pred_dir> --gt <gt_dir> --output <results_dir> \
                       [--classes 1 2 3] [--nsd_tolerance 1.0] [--workers N]
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
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
# Metric helpers
# ---------------------------------------------------------------------------

def compute_dsc(pred: np.ndarray, gt: np.ndarray, label: int) -> float:
    p = (pred == label).astype(np.uint8)
    g = (gt == label).astype(np.uint8)
    intersection = (p & g).sum()
    denom = p.sum() + g.sum()
    if denom == 0:
        return 1.0
    return float(2 * intersection / denom)


def compute_rve(pred: np.ndarray, gt: np.ndarray, label: int) -> float:
    p_vol = (pred == label).sum()
    g_vol = (gt == label).sum()
    if g_vol == 0:
        return 0.0 if p_vol == 0 else 1.0
    return float(abs(p_vol - g_vol) / g_vol)


def compute_lesion_miss_stats(
    pred: np.ndarray, gt: np.ndarray, label: int, min_voxels: int = 3,
) -> Tuple[int, int]:
    """
    Connected-component GT lesion instances vs. prediction overlap.

    Returns (n_lesions, n_missed) where a lesion counts as missed if the
    prediction has zero overlapping voxels anywhere in that component —
    relevant for the bbox-crop pipeline, since a lesion with no Stage 1
    overlap has no ROI and cannot be recovered by Stage 2 cropping.
    Uses 26-connectivity (fullyConnected=True) so diagonally touching voxels
    count as one lesion instance. Components smaller than `min_voxels` are
    dropped before counting — 1-3 voxel islands are typically mask
    digitization/partial-volume noise rather than distinct lesions.
    """
    g = (gt == label).astype(np.uint8)
    if not g.any():
        return 0, 0
    cc = sitk.ConnectedComponent(sitk.GetImageFromArray(g), True)
    cc_arr = sitk.GetArrayFromImage(cc)
    p = pred == label
    n_lesions = 0
    n_missed = 0
    for i in range(1, int(cc_arr.max()) + 1):
        comp = cc_arr == i
        if comp.sum() < min_voxels:
            continue
        n_lesions += 1
        if not p[comp].any():
            n_missed += 1
    return n_lesions, n_missed


def _surface_distances(
    pred_arr: np.ndarray,
    gt_arr: np.ndarray,
    label: int,
    sitk_spacing: Tuple[float, ...],
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return (d_pred→gt, d_gt→pred) surface distance arrays in mm via SimpleITK."""
    p = (pred_arr == label).astype(np.uint8)
    g = (gt_arr == label).astype(np.uint8)
    if not p.any() or not g.any():
        return None

    p_img = sitk.GetImageFromArray(p)
    p_img.SetSpacing(sitk_spacing)
    g_img = sitk.GetImageFromArray(g)
    g_img.SetSpacing(sitk_spacing)

    p_surf = sitk.GetArrayFromImage(sitk.LabelContour(p_img, fullyConnected=False)).astype(bool)
    g_surf = sitk.GetArrayFromImage(sitk.LabelContour(g_img, fullyConnected=False)).astype(bool)

    if not p_surf.any() or not g_surf.any():
        return None

    # SignedMaurerDistanceMap measures distance to nearest boundary — abs gives surface→surface dist
    p_dt = sitk.GetArrayFromImage(
        sitk.SignedMaurerDistanceMap(p_img, squaredDistance=False, useImageSpacing=True)
    )
    g_dt = sitk.GetArrayFromImage(
        sitk.SignedMaurerDistanceMap(g_img, squaredDistance=False, useImageSpacing=True)
    )

    return np.abs(g_dt[p_surf]), np.abs(p_dt[g_surf])


def compute_surface_metrics(
    pred_arr: np.ndarray,
    gt_arr: np.ndarray,
    label: int,
    sitk_spacing: Tuple[float, ...],
    tolerance: float = 1.0,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Return (HD, HD95, ASSD, NSD) from a single surface distance computation pass."""
    dists = _surface_distances(pred_arr, gt_arr, label, sitk_spacing)
    if dists is None:
        return None, None, None, None
    d_p2g, d_g2p = dists
    all_dists = np.concatenate([d_p2g, d_g2p])
    hd   = float(np.max(all_dists))
    hd95 = float(np.percentile(all_dists, 95))
    assd = float((d_p2g.mean() + d_g2p.mean()) / 2)
    nsd  = float(
        ((d_p2g <= tolerance).sum() + (d_g2p <= tolerance).sum())
        / (len(d_p2g) + len(d_g2p))
    )
    return hd, hd95, assd, nsd


# ---------------------------------------------------------------------------
# Case-level evaluation (runs in worker process)
# ---------------------------------------------------------------------------

def _evaluate_case(
    pred_path: Path,
    gt_path: Path,
    classes: Optional[List[int]],
    nsd_tolerance: float,
    min_lesion_voxels: int,
) -> Dict:
    pred_sitk = sitk.ReadImage(str(pred_path))
    gt_sitk   = sitk.ReadImage(str(gt_path))
    pred_arr  = sitk.GetArrayFromImage(pred_sitk).astype(np.int16)
    gt_arr    = sitk.GetArrayFromImage(gt_sitk).astype(np.int16)
    spacing   = pred_sitk.GetSpacing()  # (x, y, z) — correct order for sitk SetSpacing

    cls_list = classes if classes is not None else [
        int(l) for l in np.unique(gt_arr) if l != 0
    ]
    row: Dict = {"case_id": pred_path.name.replace(".nii.gz", "")}
    for cls in cls_list:
        hd, hd95, assd, nsd = compute_surface_metrics(
            pred_arr, gt_arr, cls, spacing, nsd_tolerance
        )
        row[f"DSC_{cls}"]  = round(compute_dsc(pred_arr, gt_arr, cls), 4)
        row[f"HD_{cls}"]   = round(hd,   2) if hd   is not None else None
        row[f"HD95_{cls}"] = round(hd95, 2) if hd95 is not None else None
        row[f"ASSD_{cls}"] = round(assd, 2) if assd is not None else None
        row[f"RVE_{cls}"]  = round(compute_rve(pred_arr, gt_arr, cls), 4)
        row[f"NSD_{cls}"]  = round(nsd,  4) if nsd  is not None else None
        n_lesions, n_missed = compute_lesion_miss_stats(pred_arr, gt_arr, cls, min_lesion_voxels)
        row[f"NLesions_{cls}"]       = n_lesions
        row[f"MissedLesions_{cls}"]  = n_missed
    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pred", required=True, type=Path)
    p.add_argument("--gt", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--classes", nargs="+", type=int, default=None,
                   help="Label indices to evaluate (default: all foreground labels found in GT)")
    p.add_argument("--nsd_tolerance", type=float, default=1.0)
    p.add_argument("--min_lesion_voxels", type=int, default=3,
                   help="Drop GT connected components smaller than this many voxels "
                        "before lesion-miss counting (digitization/partial-volume noise, default 3)")
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4),
                   help="Parallel worker processes (default: min(8, cpu_count))")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()
    args.output.mkdir(parents=True, exist_ok=True)

    pred_files = sorted(args.pred.glob("*.nii.gz"))
    log.info("Found %d prediction files", len(pred_files))

    task_args: List[Tuple] = []
    for pred_path in pred_files:
        gt_path = args.gt / pred_path.name
        if not gt_path.exists():
            log.warning("GT not found for %s — skipping", pred_path.stem)
            continue
        task_args.append((pred_path, gt_path, args.classes, args.nsd_tolerance, args.min_lesion_voxels))

    rows: List[Dict] = []

    if args.workers > 1:
        log.info("Evaluating %d cases with %d workers", len(task_args), args.workers)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_evaluate_case, *t): t[0] for t in task_args}
            for future in as_completed(futures):
                case_path = futures[future]
                try:
                    rows.append(future.result())
                    log.info("Done: %s", case_path.stem)
                except Exception as exc:
                    log.error("Failed %s: %s", case_path.stem, exc)
    else:
        for t in task_args:
            try:
                rows.append(_evaluate_case(*t))
                log.info("Done: %s", t[0].stem)
            except Exception as exc:
                log.error("Failed %s: %s", t[0].stem, exc)

    rows.sort(key=lambda r: r["case_id"])

    if not rows:
        log.error("No cases evaluated.")
        return

    metric_cols = [
        c for c in rows[0]
        if c.startswith(("DSC_", "HD_", "HD95_", "ASSD_", "RVE_", "NSD_",
                          "NLesions_", "MissedLesions_"))
    ]
    for col in sorted(metric_cols):
        vals = [r[col] for r in rows if r.get(col) is not None]
        if vals:
            log.info("%-10s  mean=%.4f  std=%.4f  (n=%d)", col, np.mean(vals), np.std(vals), len(vals))

    csv_path = args.output / "results.csv"
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        col: {
            "mean": float(np.mean([r[col] for r in rows if r.get(col) is not None])),
            "std":  float(np.std ([r[col] for r in rows if r.get(col) is not None])),
        }
        for col in metric_cols
        if any(r.get(col) is not None for r in rows)
    }
    lesion_cls = sorted({
        c.split("_", 1)[1] for c in metric_cols if c.startswith("NLesions_")
    })
    for cls in lesion_cls:
        n_col, m_col = f"NLesions_{cls}", f"MissedLesions_{cls}"
        total = sum(r[n_col] for r in rows if r.get(n_col) is not None)
        missed = sum(r[m_col] for r in rows if r.get(m_col) is not None)
        rate = (missed / total) if total else 0.0
        summary[f"lesion_miss_rate_{cls}"] = {
            "n_lesions": total, "n_missed": missed, "miss_rate": rate,
        }
        log.info("LesionMiss_%-3s  %d / %d missed  (%.1f%%)", cls, missed, total, rate * 100)

    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Results written to %s  (%.1f s total)", args.output, time.time() - t0)


if __name__ == "__main__":
    main()
