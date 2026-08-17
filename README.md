# Metadata-Informed Two-Stage nnU-Net Cascade for Ischemic Stroke Lesion Segmentation

Our submission to the [ISLES'26 Challenge](https://isles-26.grand-challenge.org/) (binary
ischemic stroke lesion segmentation in native-space T1-weighted MRI).

This repository contains the parts of our pipeline that are specific to this submission: the
metadata-conditioning trainer, dataset preparation, the ROI-cropping second stage, inference, and
evaluation. It does **not** include the nnU-Net framework itself (install it from
[github.com/MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet)) or trained model weights —
everything here is meant to be retrained from the released ISLES'26 data using the standard
`nnUNetv2_*` CLI plus the scripts below.

## Method

A two-stage coarse-to-fine cascade, both stages an nnU-Net `3d_fullres` residual-encoder-large
configuration:

1. **Stage 1** segments the lesion on the full-resolution volume (six encoder stages,
   160x192x160 patch, five-fold cross-validation).
2. The Stage 1 prediction is reduced to an axis-aligned bounding box (15-voxel margin) and used to
   crop the image.
3. **Stage 2**, re-planned from scratch on the cropped geometry (five encoder stages, 80x96x80
   patch — ROI volumes are much smaller than whole brains, so the standard nnU-Net planner picks a
   shallower network), refines the segmentation inside the crop. Its output is pasted back into
   the original volume; cases where Stage 1 predicts nothing fall back to the Stage 1 output.

Both stages are trained on identical five-fold splits so no case crosses between training and
validation. During training, Stage 2's crop is derived from the **ground-truth** mask so that all
cases remain usable regardless of Stage 1 quality; at test time the crop instead comes from the
Stage 1 prediction. This train/test mismatch is the main limitation of the cascade — see
[Results and caveats](#results-and-caveats).

### Metadata-Informed Bottleneck (MIB)

ISLES'26 ships per-case metadata (`DAYS_POST_STROKE`, `CHRONICITY`, `SITE`). We condition Stage 1
on it through a small module fused into the deepest encoder stage (`trainer/meta_conditioning.py`,
wired in by `trainer/nnUNetTrainerMetaBottleneck.py`):

```
[days_norm, chronicity_onehot, site_embedding]  ->  Linear -> ReLU -> Linear  ->  conditioning vector
conditioning vector broadcast spatially, concatenated to the bottleneck feature map,
projected back to the original channel width by a 1x1x1 convolution
```

`SITE` uses a learned embedding table with one row reserved as an explicit "unknown-site" index —
used whenever a case's site metadata is present but does not match any center in the training
vocabulary (e.g. a center that only appears in the hidden test set). During training, 5% of cases
per epoch (resampled every step) have their site identity remapped to this index, so the row
actually receives gradient rather than sitting at its random initialization. If no metadata is
supplied at all, the module degenerates to an identity pass-through, so the same checkpoint also
runs unconditioned.

## Repository layout

```
trainer/
  meta_conditioning.py            MetaConditionedStage — the MIB module
  nnUNetTrainerMetaBottleneck.py  nnU-Net trainer subclass wiring the MIB into the bottleneck

scripts/
  prepare_dataset.py     raw ATLAS layout -> nnU-Net Dataset (Stage 1), incl. metadata sidecars
  postprocess_bbox.py    Stage 1 predictions (or GT) -> bounding-box crop -> Stage 2 dataset
  evaluate.py             DSC / NSD@1mm / HD95 / ASSD / lesion-component miss rate
  crop_coverage.py        Stage 1 -> Stage 2 error-propagation audit (bbox lesion coverage)
  plot_crop_coverage.py   histogram of crop coverage
  compare_stages.py       paired Stage 1 vs Stage 2 statistics (Wilcoxon signed-rank, miss rates)
  paste_back_stage2.py    reconstructs the deployed pipeline's output (Stage 2 crop pasted
                           into Stage 1's full-volume prediction)
  predict_case.py         single-case two-stage inference that actually applies metadata
                           conditioning at inference time (drives nnUNetPredictor directly and
                           calls the trainer's set_meta() before each fold's prediction)
  inference.py             batch two-stage inference (Stage 1 ensemble -> crop -> Stage 2
                           ensemble -> paste back) via the standard nnUNetv2_predict CLI, which
                           never calls set_meta() and therefore always runs the MIB unconditioned
                           (pass-through), even though ISLES'26 provides metadata for test cases
                           too — use predict_case.py's approach instead if metadata-conditioned
                           inference is desired end-to-end

evaluation/
  ablation_fold0/         Stage-component ablation, ATLAS v2.0 cohort, fold 0 (524 train / 131 val)
  stage1_5fold_cv/         Stage 1, full ISLES'26 training release, 5-fold CV (n=1453)
  stage2_5fold_cv/         Stage 2 (GT-crop trained), same cohort, 5-fold CV (n=1448)
  figures/                 crop-coverage figure
```

## Reproducing

```bash
conda activate <your nnU-Net env>
export nnUNet_raw=...
export nnUNet_preprocessed=...
export nnUNet_results=...

# make the trainer discoverable by the nnU-Net CLI, e.g.
cp trainer/*.py <path-to-nnUNet>/nnunetv2/training/nnUNetTrainer/

# Stage 1
python scripts/prepare_dataset.py --input <ATLAS_Training_Raw> --output $nnUNet_raw/DatasetXXX_ISLES --subset all
nnUNetv2_plan_and_preprocess -d XXX --verify_dataset_integrity -pl nnUNetPlannerResEncL
for fold in 0 1 2 3 4; do
  nnUNetv2_train XXX 3d_fullres $fold -p nnUNetResEncUNetLPlans -tr nnUNetTrainerMetaBottleneck --npz
done

# Stage 2 dataset (crop from GT for training)
python scripts/postprocess_bbox.py --bbox_source gt \
  --images $nnUNet_raw/DatasetXXX_ISLES/imagesTr --labels $nnUNet_raw/DatasetXXX_ISLES/labelsTr \
  --output $nnUNet_raw/DatasetYYY_ISLESStage2 --margin 15
nnUNetv2_plan_and_preprocess -d YYY --verify_dataset_integrity -pl nnUNetPlannerResEncL
for fold in 0 1 2 3 4; do
  nnUNetv2_train YYY 3d_fullres $fold -p nnUNetResEncUNetLPlans -tr nnUNetTrainerMetaBottleneck --npz
done

# Inference on new cases (Stage 1 -> crop -> Stage 2 -> paste back)
python scripts/inference.py --input <cases> --output <predictions> \
  --stage1_dataset XXX --stage2_dataset YYY -tr nnUNetTrainerMetaBottleneck

# Evaluation
python scripts/evaluate.py --pred <predictions> --gt <gt_segmentations> --output <eval_dir> \
  --classes 1 --nsd_tolerance 1.0
```

## Results and caveats

Ablation on the ATLAS v2.0 cohort (524 train / 131 val, fold 0):

| Configuration            | DSC        | NSD@1mm    | HD95 (mm)  | ASSD (mm) |
|---------------------------|-----------|-----------|-----------|-----------|
| nnU-Net baseline          | 0.608     | 0.635     | 16.08     | 5.33      |
| + Metadata-Informed Bottleneck | 0.620 | 0.659     | 16.72     | 5.01      |
| + MIB + ROI crop          | **0.714** | **0.728** | **8.82**  | **1.88**  |

5-fold cross-validation on the full ISLES'26 training release (1453 cases; pooled out-of-fold
predictions):

| Stage              | n    | DSC       | NSD@1mm   | HD95 (mm) | ASSD (mm) |
|---------------------|------|-----------|-----------|-----------|-----------|
| Stage 1 (full volume) | 1453 | 0.651     | 0.643     | 18.46     | 5.83      |
| Stage 2 (ROI-refined) | 1448 | **0.730** | **0.720** | **9.28**  | **2.25**  |

Full per-case results, summary statistics and reproduction commands are in `evaluation/`.

**Caveat, stated explicitly rather than left implicit:** Stage 2 above is trained on
ground-truth-derived crops but, at deployment, receives crops from the Stage 1 prediction instead
— so the 0.730 DSC is an optimistic estimate of end-to-end performance, not a guarantee. We quantify
the risk directly: pooled over the full cohort, the Stage 1 bounding box retains 91.7% of the
ground-truth lesion volume on average (median 100%) and loses the lesion entirely in only 3.5% of
cases — all of which the single-stage model had already mislocalized (DSC 0), so this is not a
*new* failure mode introduced by the cascade. See `evaluation/stage1_5fold_cv/results_table.md` for
the full crop-coverage breakdown, and `scripts/paste_back_stage2.py` / `scripts/compare_stages.py`
for the tooling used to measure it.

## Citation

If you use this code, please cite the ISLES'26 data descriptor (once available) and, if relevant,
our ISLES'26 SWITCH+ workshop submission describing this method in full.
