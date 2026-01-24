# Data Folder (Important)

## What’s in here
- `large_audit_dataset.csv`: a 1,000-row synthetic dataset created via sampling + jittering a local seed.
- `final_audit_results.csv`: the latest row-level audit output produced by the pipeline.

## Warning / Usage Notes
- This repository’s `large_audit_dataset.csv` is **synthetic** (created for this audit) and intended for **testing / auditing** only.
- Do **not** treat these cases as real clinical records.
- The original seed file (`clinical_audit_seed_data.csv`) is **not committed** to GitHub. Keep it local at:
  - `data/clinical_audit_seed_data.csv`
  - (or pass a custom path via `--seed-csv`)

