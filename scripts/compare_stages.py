#!/usr/bin/env python3
"""
compare_stages.py — Paired comparison of Stage 1 (full-volume) vs Stage 2
(ROI-crop) cross-validated predictions, restricted to the case-ID
intersection of both evaluate.py results.csv files ("matched" cohort in the
paper). Computes per-metric mean+-std for both stages on that intersection,
a paired Wilcoxon signed-rank test on DSC, and pooled lesion-component
miss-rate stats (before/after, and the count "recovered": components missed
by Stage 1 that are no longer missed by Stage 2, restricted to the same
matched cases).

Usage:
    python isles/scripts/compare_stages.py \
        --stage1 isles/evaluation/paper_dataset063_5fold/results.csv \
        --stage2 isles/evaluation/paper_dataset064_5fold/results.csv \
        --output isles/evaluation/paper_dataset063_5fold/matched_cohort.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


def load(path: Path) -> dict[str, dict[str, float]]:
    with path.open() as f:
        return {r["case_id"]: r for r in csv.DictReader(f)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage1", required=True, type=Path)
    p.add_argument("--stage2", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()

    s1 = load(args.stage1)
    s2 = load(args.stage2)
    matched_ids = sorted(set(s1) & set(s2))
    print(f"Stage 1: {len(s1)} cases  Stage 2: {len(s2)} cases  Matched: {len(matched_ids)} cases")

    metrics = ["DSC_1", "NSD_1", "HD95_1", "ASSD_1"]
    result: dict = {"n_stage1": len(s1), "n_stage2": len(s2), "n_matched": len(matched_ids)}

    for stage_name, rows in [("stage1", s1), ("stage2", s2)]:
        stage_stats = {}
        for m in metrics:
            vals = [float(rows[cid][m]) for cid in matched_ids if rows[cid][m] not in ("", "None")]
            stage_stats[m] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
        result[stage_name] = stage_stats

    # Paired Wilcoxon on DSC — every matched case has a DSC value (never undefined).
    dsc1 = np.array([float(s1[cid]["DSC_1"]) for cid in matched_ids])
    dsc2 = np.array([float(s2[cid]["DSC_1"]) for cid in matched_ids])
    stat, pval = wilcoxon(dsc2, dsc1)
    result["wilcoxon_dsc"] = {
        "n": len(matched_ids), "statistic": float(stat), "p_value": float(pval),
        "mean_paired_delta": float(np.mean(dsc2 - dsc1)),
    }

    # Pooled lesion-component miss-rate, restricted to the matched cases.
    n_lesions_1 = sum(int(s1[cid]["NLesions_1"]) for cid in matched_ids)
    n_missed_1 = sum(int(s1[cid]["MissedLesions_1"]) for cid in matched_ids)
    n_lesions_2 = sum(int(s2[cid]["NLesions_1"]) for cid in matched_ids)
    n_missed_2 = sum(int(s2[cid]["MissedLesions_1"]) for cid in matched_ids)

    if n_lesions_1 == n_lesions_2:
        # Holds for a GT-derived crop (e.g. Dataset064): built to fully contain every GT
        # lesion component, so cropping cannot change the component count, and "recovered"
        # is a clean like-for-like count.
        recovered = n_missed_1 - n_missed_2
        result["lesion_components_matched"] = {
            "note": "n_total identical between stages (GT-derived crop) -- recovered is like-for-like.",
            "n_total": n_lesions_1,
            "n_missed_stage1": n_missed_1,
            "n_missed_stage2": n_missed_2,
            "miss_rate_stage1_pct": round(100 * n_missed_1 / n_lesions_1, 1),
            "miss_rate_stage2_pct": round(100 * n_missed_2 / n_lesions_2, 1),
            "n_recovered": recovered,
            "recovered_pct_of_stage1_missed": round(100 * recovered / n_missed_1, 1) if n_missed_1 else None,
        }
    else:
        # Does NOT hold for a prediction-derived crop (e.g. Dataset065): the crop box comes
        # from Stage 1's own (imperfect) prediction, so it can physically clip GT lesion
        # components that fall entirely outside it -- they never appear in Stage 2's cropped
        # GT at all, so n_total differs by construction. Report both denominators separately
        # rather than a "recovered" count, which would silently misattribute clipped-away
        # components as "still missed" or hide them entirely.
        n_clipped = n_lesions_1 - n_lesions_2
        result["lesion_components_matched"] = {
            "note": (
                "n_total DIFFERS between stages -- Stage 2's crop is derived from its own "
                "prediction (not GT), so it can clip GT lesion components entirely outside the "
                "crop box before evaluation ever sees them. miss rates below use each stage's "
                "own denominator; not a like-for-like 'recovered' count."
            ),
            "n_total_stage1": n_lesions_1,
            "n_total_stage2": n_lesions_2,
            "n_clipped_by_crop": n_clipped,
            "n_missed_stage1": n_missed_1,
            "n_missed_stage2": n_missed_2,
            "miss_rate_stage1_pct": round(100 * n_missed_1 / n_lesions_1, 1),
            "miss_rate_stage2_pct": round(100 * n_missed_2 / n_lesions_2, 1) if n_lesions_2 else None,
        }

    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
