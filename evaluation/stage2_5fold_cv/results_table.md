# ISLES — Stage 2 (GT-crop) final 5-fold cross-validation

`Dataset064_ISLESAtlas3GTCrop`, `nnUNetTrainerMetaBottleneck`, ResEnc-L 3d_fullres, trained
from scratch on each fold (no weight transfer from Stage 1 — see AGENTS.md Step 4a). All 5 folds
complete. Each of the 1448 registered cases appears in exactly one fold's validation split, so
pooling the 5 out-of-fold prediction sets gives one clean cross-validated estimate over the full
dataset — not an average of 5 separate numbers.

DSC and NSD (1 mm tolerance): higher is better. HD95 and ASSD in mm: lower is better.
n varies below 1448 for HD95/NSD/ASSD where prediction or GT is entirely empty (undefined surface
distance).

| Config                        | DSC                    | NSD@1mm                | HD95 (mm)                | ASSD (mm)               |
|--------------------------------|-------------------------|-------------------------|---------------------------|---------------------------|
| Stage 2 (GT-crop, 5-fold CV)  | 0.730 ± 0.199 (n=1448) | 0.720 ± 0.210 (n=1442) | 9.277 ± 14.800 (n=1442)  | 2.250 ± 4.595 (n=1442)   |

No crop-coverage analysis here — that metric measures GT retention inside a *Stage-1-prediction*
derived crop, and this dataset's crops are already derived from ground-truth boxes (the whole
point of the GT-crop ablation), so it doesn't apply.

Source: `metrics.csv` in this folder, from `isles/scripts/evaluate.py` run on the pooled 5-fold
out-of-fold predictions (`all_folds_pred/`, 1448 files, one per case). Reproduce with:

```bash
conda activate cuda12.9
export nnUNet_results="/root/miccai26/isles/data/nnunet/results"
export nnUNet_preprocessed="/root/miccai26/isles/data/nnunet/preprocessed"
D64="$nnUNet_results/Dataset064_ISLESAtlas3GTCrop/nnUNetTrainerMetaBottleneck__nnUNetResEncUNetLPlans__3d_fullres"

mkdir -p isles/evaluation/paper_dataset064_5fold/all_folds_pred
for fold in 0 1 2 3 4; do
  cp "$D64/fold_${fold}/validation/"*.nii.gz isles/evaluation/paper_dataset064_5fold/all_folds_pred/
done

python isles/scripts/evaluate.py \
  --pred isles/evaluation/paper_dataset064_5fold/all_folds_pred \
  --gt "$nnUNet_preprocessed/Dataset064_ISLESAtlas3GTCrop/gt_segmentations" \
  --output isles/evaluation/paper_dataset064_5fold \
  --classes 1 --nsd_tolerance 1.0

python isles/scripts/_strip_lesion_cols.py isles/evaluation/paper_dataset064_5fold
```

Table computed directly from `metrics.csv` by:

```python
import csv
import numpy as np

rows = list(csv.DictReader(open("metrics.csv")))
metrics = [("DSC_1", "DSC"), ("NSD_1", "NSD@1mm"), ("HD95_1", "HD95 (mm)"), ("ASSD_1", "ASSD (mm)")]
cells = ["Stage 2 (GT-crop, 5-fold CV)"]
for col, _ in metrics:
    vals = [float(r[col]) for r in rows if r[col] not in ("", "None")]
    cells.append(f"{np.mean(vals):.3f} ± {np.std(vals):.3f} (n={len(vals)})")
print(" | ".join(cells))
```
