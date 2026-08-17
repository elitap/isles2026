#!/usr/bin/env python3
"""
paste_back_stage2.py — Reconstruct the deployed two-stage pipeline's actual
output: start from Stage 1's own full-volume prediction and, where a Stage 2
(ROI-crop) prediction exists for that case, overwrite the crop region with
Stage 2's refined prediction (pasted back using the *_bbox.json sidecar
postprocess_bbox.py wrote). Cases with an empty Stage 1 prediction have no
ROI and keep Stage 1's prediction unchanged -- this is the fallback the paper
describes at inference time.

Cases that have a bbox (a crop was prepared) but no Stage 2 prediction yet
(e.g. that case's fold hasn't finished training) are SKIPPED, not silently
treated as a Stage-1 fallback -- there is no real pipeline answer for them
yet, and pretending Stage 1 alone is "the pipeline's answer" for a case
that's actually just missing data would misrepresent the pipeline's own
end-to-end performance.

The output is evaluable against the FULL (uncropped) ground truth, unlike
evaluating Stage 2's raw crop output against the cropped GT -- no clipping/
selection confound, this is what a deployed system would actually produce.

Usage:
    python isles/scripts/paste_back_stage2.py \
        --stage1_pred isles/evaluation/paper_dataset063_5fold/all_folds_pred \
        --stage2_pred isles/evaluation/paper_dataset065_partial_cv/all_folds_pred \
        --bbox_dir $nnUNet_raw/Dataset065_ISLESAtlas3Stage2 \
        --output isles/evaluation/paper_dataset065_partial_cv/pasted_back
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import SimpleITK as sitk

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage1_pred", required=True, type=Path, help="Dir of Stage 1 full-volume predictions (*.nii.gz)")
    p.add_argument("--stage2_pred", required=True, type=Path, help="Dir of Stage 2 cropped predictions (*.nii.gz)")
    p.add_argument("--bbox_dir", required=True, type=Path, help="Dataset065 raw root, containing *_bbox.json sidecars")
    p.add_argument("--output", required=True, type=Path)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    stage1_files = sorted(args.stage1_pred.glob("*.nii.gz"))
    log.info("Found %d Stage 1 predictions", len(stage1_files))

    n_pasted = n_fallback_no_roi = n_skipped_no_stage2 = 0
    for s1_path in stage1_files:
        case_id = s1_path.name.replace(".nii.gz", "")
        bbox_path = args.bbox_dir / f"{case_id}_bbox.json"
        s2_path = args.stage2_pred / f"{case_id}.nii.gz"

        if not bbox_path.exists():
            # Stage 1 predicted nothing for this case -> no ROI was ever formed ->
            # true inference-time fallback, matches the paper's stated behavior.
            s1_img = sitk.ReadImage(str(s1_path))
            sitk.WriteImage(s1_img, str(args.output / f"{case_id}.nii.gz"))
            n_fallback_no_roi += 1
            continue

        if not s2_path.exists():
            # A crop exists but this case's fold hasn't produced a Stage 2
            # prediction yet -- no real pipeline answer to reconstruct.
            n_skipped_no_stage2 += 1
            continue

        s1_img = sitk.ReadImage(str(s1_path))
        s1_arr = sitk.GetArrayFromImage(s1_img)
        bbox = json.loads(bbox_path.read_text())["bbox"]
        z0, z1 = bbox["z0"], bbox["z1"]
        y0, y1 = bbox["y0"], bbox["y1"]
        x0, x1 = bbox["x0"], bbox["x1"]

        s2_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(s2_path)))
        expected_shape = (z1 - z0 + 1, y1 - y0 + 1, x1 - x0 + 1)
        if s2_arr.shape != expected_shape:
            raise ValueError(
                f"{case_id}: Stage 2 prediction shape {s2_arr.shape} does not match "
                f"bbox-implied shape {expected_shape} -- geometry mismatch, investigate "
                f"before trusting any paste-back output."
            )

        out_arr = s1_arr.copy()
        out_arr[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1] = s2_arr
        out_img = sitk.GetImageFromArray(out_arr)
        out_img.CopyInformation(s1_img)
        sitk.WriteImage(out_img, str(args.output / f"{case_id}.nii.gz"))
        n_pasted += 1

    log.info(
        "Pasted Stage 2 into %d cases; %d true Stage-1 fallbacks (no ROI); "
        "%d skipped (crop exists but Stage 2 prediction not available yet)",
        n_pasted, n_fallback_no_roi, n_skipped_no_stage2,
    )
    log.info("Reconstructed pipeline output written for %d cases total", n_pasted + n_fallback_no_roi)


if __name__ == "__main__":
    main()
