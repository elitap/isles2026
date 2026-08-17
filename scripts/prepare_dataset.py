"""
Convert ISLES Training_Raw data into an nnUNet-formatted dataset.

Per-case metadata (days_post_stroke, chronicity, site_idx) is stored as
ISLES_XXXX_0001.json alongside the image ISLES_XXXX_0000.nii.gz in imagesTr/.
This follows nnUNet's modality naming convention; the custom trainer reads
these JSON files instead of a second NIfTI channel.

A global metadata_stats.json with normalization parameters and the site
vocabulary is written to the dataset root for use at inference time.

Sampling modes:
    --subset one-per-site   1 random Training case per site (ablation, default)
    --subset all            all Training cases (full dataset)

Usage examples:
    # Ablation dataset (33 cases, one per site):
    python isles/scripts/prepare_dataset.py \\
        --input  isles/data/raw/Training_Raw \\
        --output $nnUNet_raw/Dataset051_ISLESAblation \\
        --subset one-per-site

    # Full dataset:
    python isles/scripts/prepare_dataset.py \\
        --input  isles/data/raw/Training_Raw \\
        --output $nnUNet_raw/Dataset053_ISLESFull \\
        --subset all
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import shutil
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Chronicity encoding: extend this map if Batch 2 introduces new values.
# Empty string (no information) → 0.
# "1.0" appears in ATLAS3/SOOP cases and is equivalent to "1" (chronic).
CHRONICITY_MAP: dict[str, int] = {"": 0, "1": 1, "1.0": 1, "2": 2, "3": 3}


# ---------------------------------------------------------------------------
# Data discovery
# ---------------------------------------------------------------------------

def discover_cases(training_raw: Path, allowed_splits: frozenset[str]) -> dict[str, list[dict[str, Any]]]:
    """Return {site_id: [case_dict, ...]} for every Training case found."""
    sites: dict[str, list[dict[str, Any]]] = {}

    for site_dir in sorted(training_raw.iterdir()):
        if not site_dir.is_dir():
            continue
        site_id = site_dir.name
        cases: list[dict[str, Any]] = []

        for subject_dir in sorted(site_dir.iterdir()):
            if not subject_dir.is_dir():
                continue
            anat_dir = subject_dir / "ses-1" / "anat"
            if not anat_dir.is_dir():
                continue

            meta_files = list(anat_dir.glob("*_metadata.csv"))
            if not meta_files:
                continue

            with meta_files[0].open() as f:
                row = next(csv.DictReader(f), None)

            if row is None:
                log.warning(
                    "Empty metadata CSV (header-only) for %s — including with sentinel values",
                    subject_dir,
                )
            else:
                atlas_split = row.get("ATLAS2_DATASET", "").strip()
                if atlas_split not in allowed_splits:
                    continue

            images = list(anat_dir.glob("*_space-orig_desc-brain_T1w.nii.gz"))
            labels = list(anat_dir.glob("*_space-orig_label-lesion_desc-T1lesion_mask.nii.gz"))
            if not images or not labels:
                log.warning("Missing image or label for %s — skipping", subject_dir)
                continue

            cases.append({
                "site": site_id,
                "subject": subject_dir.name,
                "image": images[0],
                "label": labels[0],
                "days_post_stroke_raw": row.get("DAYS_POST_STROKE", "").strip() if row else "",
                "chronicity_raw": row.get("CHRONICITY", "").strip() if row else "",
            })

        sites[site_id] = cases

    return sites


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_cases(
    sites: dict[str, list[dict[str, Any]]],
    mode: str,
    seed: int,
) -> list[dict[str, Any]]:
    """
    'one-per-site' → 1 random Training case per site (sites with 0 cases skipped).
    'all'          → all Training cases.
    """
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []

    for site_id, cases in sites.items():
        if not cases:
            log.warning("Site %s has no Training cases — skipping", site_id)
            continue
        if mode == "one-per-site":
            selected.append(rng.choice(cases))
        else:
            selected.extend(cases)

    return selected


# ---------------------------------------------------------------------------
# Metadata encoding
# ---------------------------------------------------------------------------

def build_encoding(
    cases: list[dict[str, Any]],
    all_site_ids: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Encode metadata for each case and compute normalisation statistics.

    Returns (cases_with_meta, stats) where cases_with_meta is the input list
    augmented with 'days_norm', 'chronicity', 'site_idx' fields.

    DAYS_POST_STROKE
        Empty field → sentinel –1 (no information / no stroke).
        Non-empty   → min-max normalised to [0, 1] over non-sentinel training values.

    CHRONICITY
        Empty string → 0 (unknown/missing). See CHRONICITY_MAP.

    SITE
        Sorted vocabulary over *all* training sites so the embedding table
        size is stable between ablation and full-dataset runs.
    """
    # Index 0 is reserved as the permanent unknown/OOV bucket so that sites
    # not seen at training time can be mapped to it without index errors.
    UNKNOWN_SITE = "__unknown__"
    site_vocab = [UNKNOWN_SITE] + sorted(all_site_ids)
    site_to_idx = {s: i for i, s in enumerate(site_vocab)}

    # Collect valid days values for min-max normalisation
    raw_days: list[float] = []
    for case in cases:
        raw = case["days_post_stroke_raw"]
        if raw != "":
            raw_days.append(float(raw))

    days_min = min(raw_days) if raw_days else 0.0
    days_max = max(raw_days) if raw_days else 1.0
    days_range = days_max - days_min if days_max != days_min else 1.0

    for case in cases:
        raw = case["days_post_stroke_raw"]
        case["days_norm"] = -1.0 if raw == "" else round((float(raw) - days_min) / days_range, 6)

        chron_raw = case["chronicity_raw"]
        if chron_raw not in CHRONICITY_MAP:
            log.warning(
                "Unknown CHRONICITY value %r for %s — mapped to 0 (unknown)",
                chron_raw, case["subject"],
            )
        case["chronicity"] = CHRONICITY_MAP.get(chron_raw, 0)

        case["site_idx"] = site_to_idx.get(case["site"], 0)

    stats: dict[str, Any] = {
        "days_post_stroke": {
            "min_raw": days_min,
            "max_raw": days_max,
            "sentinel": -1.0,
            "note": "empty field → sentinel –1; others min-max normalised to [0, 1]",
        },
        "chronicity": {
            "encoding": CHRONICITY_MAP,
            "note": "0=unknown/missing; extend for Batch 2 values if needed",
        },
        "site": {
            "vocab": site_vocab,
            "note": "site_idx = position in this sorted list; used for learned embedding table",
        },
    }

    return cases, stats


# ---------------------------------------------------------------------------
# nnUNet dataset construction
# ---------------------------------------------------------------------------

def write_dataset(
    cases: list[dict[str, Any]],
    output: Path,
    dataset_name: str,
    stats: dict[str, Any],
) -> None:
    """
    Write nnUNet-formatted dataset to output/.

    imagesTr/
        ISLES_XXXX_0000.nii.gz   — T1w image (channel 0)
        ISLES_XXXX_0001.json     — encoded metadata (follows modality naming)
    labelsTr/
        ISLES_XXXX.nii.gz        — binary lesion mask
    dataset.json                 — nnUNet dataset descriptor
    metadata_stats.json          — normalisation params + site vocabulary
    """
    images_dir = output / "imagesTr"
    labels_dir = output / "labelsTr"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    training_entries = []

    for i, case in enumerate(cases):
        # Derive case ID from the original subject name (strip "sub-" prefix).
        # Result e.g. "r031s033" → files: r031s033_0000.nii.gz, r031s033_0001.json
        case_id = case["subject"].removeprefix("sub-")
        case["case_id"] = case_id

        # Channel 0000: T1w image
        shutil.copy2(case["image"], images_dir / f"{case_id}_0000.nii.gz")

        # Channel 0001: per-case metadata JSON (follows nnUNet modality naming)
        meta_payload = {
            "subject":    case["subject"],
            "site":       case["site"],
            "days_norm":  case["days_norm"],
            "chronicity": case["chronicity"],
            "site_idx":   case["site_idx"],
        }
        with (images_dir / f"{case_id}_0001.json").open("w") as f:
            json.dump(meta_payload, f)

        # Label
        shutil.copy2(case["label"], labels_dir / f"{case_id}.nii.gz")

        training_entries.append({
            "image": f"./imagesTr/{case_id}_0000.nii.gz",
            "label": f"./labelsTr/{case_id}.nii.gz",
        })

        log.info("  [%d/%d] %s → %s", i + 1, len(cases), case["subject"], case_id)

    # dataset.json
    dataset_json = {
        "name": dataset_name,
        "description": "ISLES 2026 ischemic stroke lesion segmentation",
        "reference": "https://isles-26.grand-challenge.org/",
        "licence": "see challenge website",
        "release": "1.0",
        "channel_names": {
            "0": "T1w",
        },
        "labels": {
            "background": 0,
            "lesion": 1,
        },
        "numTraining": len(cases),
        "file_ending": ".nii.gz",
        "training": training_entries,
    }
    with (output / "dataset.json").open("w") as f:
        json.dump(dataset_json, f, indent=2)

    # Global normalization stats + site vocabulary
    with (output / "metadata_stats.json").open("w") as f:
        json.dump(stats, f, indent=2)

    log.info("dataset.json and metadata_stats.json written to %s", output)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, type=Path, help="Path to Training_Raw/")
    p.add_argument(
        "--output", required=True, type=Path,
        help="Output nnUNet dataset dir (e.g. $nnUNet_raw/Dataset051_ISLESAblation)",
    )
    p.add_argument(
        "--subset", default="one-per-site", choices=["one-per-site", "all"],
        help="Sampling strategy (default: one-per-site)",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed for sampling (default: 42)")
    p.add_argument(
        "--atlas-splits", default="Training",
        help="Comma-separated ATLAS2_DATASET values to include (default: 'Training'; use 'Training,Testing' for all 955 cases)",
    )
    p.add_argument(
        "--dataset-name", default=None,
        help="Override dataset name in dataset.json (default: derived from --output dirname)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    training_raw = args.input.resolve()
    output = args.output.resolve()
    dataset_name = args.dataset_name or output.name

    if not training_raw.is_dir():
        raise SystemExit(f"--input does not exist: {training_raw}")

    allowed_splits = frozenset(s.strip() for s in args.atlas_splits.split(","))
    log.info("Discovering cases in %s (ATLAS2_DATASET filter: %s)", training_raw, allowed_splits)
    sites = discover_cases(training_raw, allowed_splits)
    all_site_ids = sorted(sites.keys())
    n_nonempty = sum(1 for v in sites.values() if v)
    total_found = sum(len(v) for v in sites.values())
    log.info("Found %d sites with training cases (%d total cases)", n_nonempty, total_found)

    log.info("Sampling with strategy '%s' (seed=%d)", args.subset, args.seed)
    cases = sample_cases(sites, mode=args.subset, seed=args.seed)
    log.info("Selected %d cases from %d sites", len(cases), len({c["site"] for c in cases}))

    log.info("Encoding metadata")
    cases, stats = build_encoding(cases, all_site_ids)

    log.info("Writing nnUNet dataset to %s", output)
    write_dataset(cases, output, dataset_name, stats)

    log.info(
        "Done — %d cases, %d sites represented",
        len(cases), len({c["site"] for c in cases}),
    )


if __name__ == "__main__":
    main()
