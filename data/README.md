# Data Folder (Important)

## What’s in here
- `clinical_audit_seed_data.csv`: the original 21-row seed dataset gotten from a Medical Laboratory Centre
- `large_audit_dataset.csv`: a 1,000-row augmented dataset created via sampling + jittering a local seed.
- `final_audit_results.csv`: the latest row-level audit output produced by the pipeline.

## Warning / Usage Notes
- This repository’s `large_audit_dataset.csv` is **augmented** (created for this audit) and intended for **testing / auditing** only.
- Do **not** treat these cases as real clinical records.
- The seed file is committed in this repo; you can override its path via `--seed-csv` if needed.

