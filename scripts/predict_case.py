#!/usr/bin/env python3
"""
predict_case.py — Two-stage ISLES inference for a single unseen test case,
with metadata conditioning actually applied at inference time.

Input is a case in the *raw* ATLAS layout (image + `*_metadata.csv`), the same
format found under isles/data/raw/*/Training_Raw. The raw CSV fields
(DAYS_POST_STROKE, CHRONICITY, SITE) are normalised using the training
dataset's metadata_stats.json (same encoding as prepare_dataset.py) before
being fed to the network.

IMPORTANT — why this script does not just shell out to nnUNetv2_predict:
nnUNetTrainerMetaBottleneck's MetaConditionedStage only receives metadata via
an explicit set_meta() call, which the trainer issues from train_step /
validation_step. The standard nnUNetPredictor / nnUNetv2_predict CLI path
never calls set_meta(), so it always runs the network in pass-through mode
(conditioning disabled) — this is intentional for the Docker submission,
where no metadata is available at all. To actually condition on metadata for
a case where we do have it, this script drives nnUNetPredictor directly and
calls set_meta() on the (single, shared) network instance before triggering
prediction, then predict_from_files_sequential() runs that one case,
single-threaded, across all requested folds while the metadata stays set.

Pipeline:
  1. Discover/parse the raw case (image + metadata.csv) and normalise its
     metadata against metadata_stats.json.
  2. Stage 1: metadata-conditioned ensemble on the full-FOV image
     (Dataset063_ISLESFullAtlas3).
  3. Crop the original image to the Stage 1 prediction's bounding box
     (+ margin).
  4. Stage 2: metadata-conditioned ensemble on the crop
     (Dataset064_ISLESAtlas3GTCrop).
  5. Paste the Stage 2 prediction back into the original image space.
     Falls back to the Stage 1 prediction if Stage 1 predicted nothing.

Environment variables nnUNet_raw, nnUNet_preprocessed, and nnUNet_results
must be set exactly as for nnUNetv2_train / nnUNetv2_predict — in particular
nnUNet_preprocessed is required because nnUNetTrainerMetaBottleneck's
build_network_architecture reads n_sites from
nnUNet_preprocessed/<dataset>/metadata_stats.json to size the site embedding.

Usage:
    python isles/scripts/predict_case.py \\
        --case_dir isles/data/raw/ATLAS3_Training_Raw/R005/sub-r005s086/ses-1/anat \\
        --output   isles/predictions/single_case/r005s086.nii.gz \\
        --stage1_dataset 63 \\
        --stage2_dataset 64
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import SimpleITK as sitk
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

BBox = Tuple[int, int, int, int, int, int]  # (z0,z1, y0,y1, x0,x1)


# ---------------------------------------------------------------------------
# Raw case discovery + metadata normalisation (mirrors prepare_dataset.py)
# ---------------------------------------------------------------------------

def discover_raw_case(case_dir: Path) -> Dict[str, Any]:
    """
    Parse a raw ATLAS-layout case directory
    (<site>/<subject>/ses-1/anat/), same layout as Training_Raw.
    """
    images = sorted(case_dir.glob("*_space-orig_desc-brain_T1w.nii.gz"))
    if not images:
        raise SystemExit(f"No *_space-orig_desc-brain_T1w.nii.gz found in {case_dir}")

    days_raw, chronicity_raw, site = "", "", ""
    meta_files = sorted(case_dir.glob("*_metadata.csv"))
    if meta_files:
        with meta_files[0].open() as f:
            row = next(csv.DictReader(f), None)
        if row:
            days_raw = row.get("DAYS_POST_STROKE", "").strip()
            chronicity_raw = row.get("CHRONICITY", "").strip()
            site = row.get("SITE", "").strip()
    else:
        log.warning("No *_metadata.csv found in %s — proceeding with empty metadata", case_dir)

    # ses-1/anat -> subject -> site, mirrors Training_Raw/<site>/<subject>/ses-1/anat
    subject_dir = case_dir.parent.parent
    site_dir = subject_dir.parent
    subject = subject_dir.name
    if not site:
        site = site_dir.name

    return {
        "image": images[0],
        "days_raw": days_raw,
        "chronicity_raw": chronicity_raw,
        "site": site,
        "subject": subject,
        "case_id": subject.removeprefix("sub-"),
    }


def encode_metadata(days_raw: str, chronicity_raw: str, site: str, stats: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise raw metadata fields exactly as prepare_dataset.py's build_encoding does,
    but using stats already computed over the training set (no recomputation)."""
    days_stats = stats["days_post_stroke"]
    if days_raw in ("", None):
        days_norm = days_stats["sentinel"]
    else:
        d_min, d_max = days_stats["min_raw"], days_stats["max_raw"]
        d_range = d_max - d_min if d_max != d_min else 1.0
        days_norm = round((float(days_raw) - d_min) / d_range, 6)

    chron_map = stats["chronicity"]["encoding"]
    if chronicity_raw not in chron_map:
        log.warning("Unknown CHRONICITY value %r — mapping to 0 (unknown)", chronicity_raw)
    chronicity = chron_map.get(chronicity_raw, 0)

    site_vocab = stats["site"]["vocab"]
    site_to_idx = {s: i for i, s in enumerate(site_vocab)}
    if site not in site_to_idx:
        log.warning(
            "Site %r not in training vocabulary (%d sites) — mapping to unknown bucket (index 0)",
            site, len(site_vocab),
        )
    site_idx = site_to_idx.get(site, 0)

    return {"days_norm": days_norm, "chronicity": chronicity, "site_idx": site_idx}


# ---------------------------------------------------------------------------
# Model / dataset resolution
# ---------------------------------------------------------------------------

def resolve_model_dir(nnunet_results: Path, dataset_id: int, trainer: str, plans: str, config: str) -> Path:
    candidates = sorted(nnunet_results.glob(f"Dataset{dataset_id:03d}_*/{trainer}__{plans}__{config}"))
    if not candidates:
        raise SystemExit(
            f"No results folder for dataset {dataset_id} matching "
            f"Dataset{dataset_id:03d}_*/{trainer}__{plans}__{config} under {nnunet_results}"
        )
    if len(candidates) > 1:
        log.warning("Multiple results folders match dataset %d; using %s", dataset_id, candidates[0])
    return candidates[0]


def resolve_metadata_stats(nnunet_raw: Path, dataset_id: int) -> Path:
    candidates = sorted(nnunet_raw.glob(f"Dataset{dataset_id:03d}_*/metadata_stats.json"))
    if not candidates:
        raise SystemExit(f"metadata_stats.json not found for dataset {dataset_id} under {nnunet_raw}")
    return candidates[0]


# ---------------------------------------------------------------------------
# Bounding-box helpers (mirrors inference.py / postprocess_bbox.py)
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
    coords = np.argwhere(mask_np > 0)
    if coords.size == 0:
        return None
    z0, y0, x0 = coords.min(axis=0).tolist()
    z1, y1, x1 = coords.max(axis=0).tolist()
    return _expand_bbox((z0, z1, y0, y1, x0, x1), mask_np.shape, margin)


def crop_sitk(img: sitk.Image, bb: BBox) -> sitk.Image:
    z0, z1, y0, y1, x0, x1 = bb
    size = [int(x1 - x0 + 1), int(y1 - y0 + 1), int(z1 - z0 + 1)]
    index = [int(x0), int(y0), int(z0)]
    return sitk.Extract(img, size, index)


# ---------------------------------------------------------------------------
# nnUNetPredictor wrapper with metadata conditioning
# ---------------------------------------------------------------------------

def _find_meta_stage(network: torch.nn.Module):
    from nnunetv2.training.nnUNetTrainer.meta_conditioning import MetaConditionedStage
    base = getattr(network, "_orig_mod", network)
    for module in base.modules():
        if isinstance(module, MetaConditionedStage):
            return module
    return None


def run_stage(
    model_dir: Path,
    folds: Optional[list],
    checkpoint: str,
    device: torch.device,
    image_path: Path,
    output_prefix: Path,
    meta: Dict[str, Any],
) -> Path:
    """Run a metadata-conditioned ensemble prediction for one case. Returns the
    written segmentation file path."""
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=(device.type == "cuda"),
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    predictor.initialize_from_trained_model_folder(str(model_dir), use_folds=folds, checkpoint_name=checkpoint)

    meta_stage = _find_meta_stage(predictor.network)
    if meta_stage is not None:
        days = torch.tensor([meta["days_norm"]], dtype=torch.float32)
        chron = torch.tensor([meta["chronicity"]], dtype=torch.long)
        site_idx = torch.tensor([meta["site_idx"]], dtype=torch.long)
        meta_stage.set_meta(days, chron, site_idx)
        log.info(
            "Metadata conditioning active for %s: days_norm=%.4f chronicity=%d site_idx=%d",
            model_dir.parent.name, meta["days_norm"], meta["chronicity"], meta["site_idx"],
        )
    else:
        log.warning("No MetaConditionedStage found in %s — running unconditioned", model_dir)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    predictor.predict_from_files_sequential(
        [[str(image_path)]], [str(output_prefix)],
        save_probabilities=False, overwrite=True,
    )

    if meta_stage is not None:
        meta_stage.clear_meta()

    return Path(str(output_prefix) + predictor.dataset_json["file_ending"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case_dir", type=Path, default=None,
                   help="Raw case dir, e.g. Training_Raw/<SITE>/<subject>/ses-1/anat "
                        "(auto-discovers image + metadata.csv)")
    p.add_argument("--image", type=Path, default=None,
                   help="Explicit path to the T1w image (overrides --case_dir discovery)")
    p.add_argument("--metadata_csv", type=Path, default=None,
                   help="Explicit path to the case's *_metadata.csv (overrides --case_dir discovery)")
    p.add_argument("--site", default=None, help="Override SITE value")
    p.add_argument("--days_post_stroke", default=None, help="Override DAYS_POST_STROKE value")
    p.add_argument("--chronicity", default=None, help="Override CHRONICITY value")
    p.add_argument("--case_id", default=None, help="Override case ID used for output filenames")

    p.add_argument("--output", required=True, type=Path, help="Final prediction output path (.nii.gz)")

    p.add_argument("--stage1_dataset", required=True, type=int, help="nnUNet dataset ID for Stage 1 (e.g. 63)")
    p.add_argument("--stage2_dataset", required=True, type=int, help="nnUNet dataset ID for Stage 2 (e.g. 64)")
    p.add_argument("--metadata_stats", type=Path, default=None,
                   help="Path to metadata_stats.json (default: auto-resolved from nnUNet_raw + --stage1_dataset)")
    p.add_argument("--trainer", default="nnUNetTrainerMetaBottleneck")
    p.add_argument("--plans", default="nnUNetResEncUNetLPlans")
    p.add_argument("--config", default="3d_fullres")
    p.add_argument("--checkpoint", default="checkpoint_final.pth")
    p.add_argument("--stage1_folds", nargs="+", type=int, default=None,
                   help="Folds to ensemble for Stage 1 (default: auto-detect available checkpoints)")
    p.add_argument("--stage2_folds", nargs="+", type=int, default=None,
                   help="Folds to ensemble for Stage 2 (default: auto-detect available checkpoints)")
    p.add_argument("--margin", type=int, default=15, help="Voxel margin around Stage 1 bbox (default 15)")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--tmp_dir", type=Path, default=None, help="Temp dir for intermediates (default: system tmp)")
    return p.parse_args()


def main() -> None:
    import os

    args = parse_args()
    t0 = time.time()

    nnunet_raw = os.environ.get("nnUNet_raw")
    nnunet_results = os.environ.get("nnUNet_results")
    if not nnunet_raw or not os.environ.get("nnUNet_preprocessed") or not nnunet_results:
        raise SystemExit(
            "nnUNet_raw, nnUNet_preprocessed, and nnUNet_results must all be set "
            "(nnUNet_preprocessed is required at inference too — "
            "nnUNetTrainerMetaBottleneck reads metadata_stats.json from there to size the site embedding)."
        )
    nnunet_raw = Path(nnunet_raw)
    nnunet_results = Path(nnunet_results)

    # --- Resolve raw case -------------------------------------------------
    if args.case_dir is not None:
        case = discover_raw_case(args.case_dir)
    elif args.image is not None:
        case = {"image": args.image, "days_raw": "", "chronicity_raw": "", "site": "",
                "subject": args.image.stem, "case_id": args.case_id or args.image.stem}
        if args.metadata_csv is not None:
            with args.metadata_csv.open() as f:
                row = next(csv.DictReader(f), None)
            if row:
                case["days_raw"] = row.get("DAYS_POST_STROKE", "").strip()
                case["chronicity_raw"] = row.get("CHRONICITY", "").strip()
                case["site"] = row.get("SITE", "").strip()
    else:
        raise SystemExit("Provide either --case_dir or --image")

    if args.site is not None:
        case["site"] = args.site
    if args.days_post_stroke is not None:
        case["days_raw"] = args.days_post_stroke
    if args.chronicity is not None:
        case["chronicity_raw"] = args.chronicity
    if args.case_id is not None:
        case["case_id"] = args.case_id

    log.info("Case %s: site=%r days_raw=%r chronicity_raw=%r image=%s",
             case["case_id"], case["site"], case["days_raw"], case["chronicity_raw"], case["image"])

    # --- Normalise metadata -------------------------------------------------
    stats_path = args.metadata_stats or resolve_metadata_stats(nnunet_raw, args.stage1_dataset)
    stats = json.loads(stats_path.read_text())
    meta = encode_metadata(case["days_raw"], case["chronicity_raw"], case["site"], stats)
    log.info("Normalised metadata: %s (stats from %s)", meta, stats_path)

    stage1_dir = resolve_model_dir(nnunet_results, args.stage1_dataset, args.trainer, args.plans, args.config)
    stage2_dir = resolve_model_dir(nnunet_results, args.stage2_dataset, args.trainer, args.plans, args.config)
    device = torch.device(args.device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    own_tmp = args.tmp_dir is None
    tmp_root = Path(tempfile.mkdtemp(prefix="isles_predict_case_")) if own_tmp else args.tmp_dir
    tmp_root.mkdir(parents=True, exist_ok=True)
    log.info("Tmp dir: %s", tmp_root)

    try:
        # --- Stage 1: full-FOV ensemble ------------------------------------
        log.info("=== Stage 1 (dataset %d) ===", args.stage1_dataset)
        stage1_seg_path = run_stage(
            stage1_dir, args.stage1_folds, args.checkpoint, device,
            case["image"], tmp_root / "stage1" / case["case_id"], meta,
        )
        stage1_img = sitk.ReadImage(str(stage1_seg_path))
        stage1_np = sitk.GetArrayFromImage(stage1_img)

        bb = bbox_from_mask(stage1_np, args.margin)
        if bb is None:
            log.warning("Stage 1 prediction empty for %s — using Stage 1 output as final result", case["case_id"])
            shutil.copy2(str(stage1_seg_path), str(args.output))
        else:
            # --- Crop original image to Stage 1 bbox -----------------------
            z0, z1, y0, y1, x0, x1 = bb
            log.info("=== Cropping to bbox z[%d:%d] y[%d:%d] x[%d:%d] (margin=%d) ===",
                     z0, z1, y0, y1, x0, x1, args.margin)
            orig_img = sitk.ReadImage(str(case["image"]))
            cropped_img = crop_sitk(orig_img, bb)
            cropped_path = tmp_root / "stage2_input" / f"{case['case_id']}_0000.nii.gz"
            cropped_path.parent.mkdir(parents=True, exist_ok=True)
            sitk.WriteImage(cropped_img, str(cropped_path))

            # --- Stage 2: ensemble on the crop ------------------------------
            log.info("=== Stage 2 (dataset %d) ===", args.stage2_dataset)
            stage2_seg_path = run_stage(
                stage2_dir, args.stage2_folds, args.checkpoint, device,
                cropped_path, tmp_root / "stage2" / case["case_id"], meta,
            )
            stage2_np = sitk.GetArrayFromImage(sitk.ReadImage(str(stage2_seg_path)))

            # --- Paste back into original space -----------------------------
            log.info("=== Pasting Stage 2 prediction back into original space ===")
            orig_shape = sitk.GetArrayFromImage(orig_img).shape
            expected = (z1 - z0 + 1, y1 - y0 + 1, x1 - x0 + 1)
            if stage2_np.shape != expected:
                log.warning("Stage 2 shape %s != expected crop shape %s", stage2_np.shape, expected)

            result_np = np.zeros(orig_shape, dtype=np.uint8)
            pz1 = min(z0 + stage2_np.shape[0], orig_shape[0])
            py1 = min(y0 + stage2_np.shape[1], orig_shape[1])
            px1 = min(x0 + stage2_np.shape[2], orig_shape[2])
            result_np[z0:pz1, y0:py1, x0:px1] = stage2_np[:pz1 - z0, :py1 - y0, :px1 - x0]

            result_img = sitk.GetImageFromArray(result_np)
            result_img.CopyInformation(orig_img)
            sitk.WriteImage(result_img, str(args.output))

        log.info("=== Done — final prediction written to %s (%.1f s total) ===",
                 args.output, time.time() - t0)

    finally:
        if own_tmp:
            shutil.rmtree(tmp_root, ignore_errors=True)
            log.info("Cleaned up tmp dir %s", tmp_root)


if __name__ == "__main__":
    main()
