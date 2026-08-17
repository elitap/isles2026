# ISLES — Stage 1 (full volume) final 5-fold cross-validation

`Dataset063_ISLESFullAtlas3`, `nnUNetTrainerMetaBottleneck`, ResEnc-L 3d_fullres. All 5 folds
complete as of 2026-07-13 (folds 3-4 finished after `paper_dataset063_partial_cv/` — that folder is
now superseded, kept for the record only). 292+291+290+290+290 = 1453 out-of-fold predictions,
exactly matching `dataset.json`'s `numTraining`.

DSC and NSD (1 mm tolerance): higher is better. HD95 and ASSD in mm: lower is better.
n varies below 1453 for HD95/NSD/ASSD where prediction or GT is entirely empty.

| Config                        | DSC                     | NSD@1mm                 | HD95 (mm)                 | ASSD (mm)                 |
|---------------------------------|---------------------------|---------------------------|------------------------------|------------------------------|
| Stage 1 (full volume, 5-fold) | 0.651 ± 0.285 (n=1453)  | 0.643 ± 0.260 (n=1411)  | 18.46 ± 23.21 (n=1411)    | 5.83 ± 11.73 (n=1411)     |

## Matched-cohort comparison with Stage 2 (Dataset064)

Stage 2's ROI-crop dataset excludes 5 cases with an empty GT mask (no lesion to crop), so the
case-ID intersection between Stage 1 (1453) and Stage 2 (1448) is exactly Stage 2's own cohort
(n=1448). Computed by `isles/scripts/compare_stages.py`:

| Stage | n | DSC | NSD@1mm | HD95 (mm) | ASSD (mm) |
|---|---|---|---|---|---|
| Stage 1 (matched) | 1448 | 0.651 ± 0.284 | 0.643 ± 0.260 (n=1411) | 18.46 ± 23.21 (n=1411) | 5.83 ± 11.73 (n=1411) |
| Stage 2 (matched) | 1448 | 0.730 ± 0.199 | 0.720 ± 0.210 (n=1442) | 9.28 ± 14.80 (n=1442) | 2.25 ± 4.60 (n=1442) |

Paired Wilcoxon signed-rank test on per-case DSC (matched, n=1448): statistic=299287,
**p = 2.15e-43**, mean paired delta = **+0.0789**.

Pooled lesion-component miss rate (matched, n=1448; 4313 total GT components — same total for both
stages, since Dataset064's GT-crop is built to fully contain each GT lesion, so cropping cannot
change the component count — this equality is asserted in the script and passed):

| | n missed | miss rate |
|---|---|---|
| Stage 1 | 1853 | 43.0% |
| Stage 2 | 1327 | 30.8% |

Recovered: 526 components (28.4% of Stage 1's 1853 missed) that Stage 2 no longer misses.

## Crop coverage (Stage 2 relevance)

| Metric | Value |
|---|---|
| n evaluated | 1448 (5 GT-empty cases excluded) |
| Mean coverage | 91.7% |
| Median coverage | 100.0% |
| Cases at 100% coverage | 939 / 1448 (64.8%) |
| Cases with DSC=0 | 91 total — 40 empty-prediction (scored 100%, no crop applied) + 51 non-empty but mislocated (real crop loss, 3.5%) |

See `crop_coverage/crop_coverage.csv` for full per-case detail and
`crop_coverage/crop_coverage_hist_log.png` for the histogram. The paper's Fig. 2(b) is generated
from this same directory by `isles/paper/make_figs.py`.

## Reproduce

```bash
conda activate cuda12.9
export nnUNet_results="/root/miccai26/isles/data/nnunet/results"
export nnUNet_preprocessed="/root/miccai26/isles/data/nnunet/preprocessed"
D63="$nnUNet_results/Dataset063_ISLESFullAtlas3/nnUNetTrainerMetaBottleneck__nnUNetResEncUNetLPlans__3d_fullres"

mkdir -p isles/evaluation/paper_dataset063_5fold/all_folds_pred
for fold in 0 1 2 3 4; do
  cp "$D63/fold_${fold}/validation/"*.nii.gz isles/evaluation/paper_dataset063_5fold/all_folds_pred/
done

python isles/scripts/evaluate.py \
  --pred isles/evaluation/paper_dataset063_5fold/all_folds_pred \
  --gt "$nnUNet_preprocessed/Dataset063_ISLESFullAtlas3/gt_segmentations" \
  --output isles/evaluation/paper_dataset063_5fold \
  --classes 1 --nsd_tolerance 1.0

python isles/scripts/_strip_lesion_cols.py isles/evaluation/paper_dataset063_5fold

python isles/scripts/crop_coverage.py \
  --pred isles/evaluation/paper_dataset063_5fold/all_folds_pred \
  --gt "$nnUNet_preprocessed/Dataset063_ISLESFullAtlas3/gt_segmentations" \
  --output isles/evaluation/paper_dataset063_5fold/crop_coverage \
  --margin 15 --label 1

python isles/scripts/plot_crop_coverage.py \
  --csv isles/evaluation/paper_dataset063_5fold/crop_coverage/crop_coverage.csv \
  --output isles/evaluation/paper_dataset063_5fold/crop_coverage \
  --label "5-fold CV (full cohort)"

python isles/scripts/compare_stages.py \
  --stage1 isles/evaluation/paper_dataset063_5fold/results.csv \
  --stage2 isles/evaluation/paper_dataset064_5fold/results.csv \
  --output isles/evaluation/paper_dataset063_5fold/matched_cohort.json

python isles/paper/make_figs.py --input isles/evaluation --output isles/paper/figs
```
