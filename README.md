# Clinical AI Audit (Red Teaming Pipeline)

End-to-end pipeline for red teaming a clinical AI model using a seeded dataset, data augmentation, and Ollama (Llama 3), culminating in a chart suitable for sharing.

## Repository structure

```
clinical-ai-redteam-audit/
├── data/
│   ├── clinical_audit_seed_data.csv
│   ├── large_audit_dataset.csv
│   └── README.md
├── scripts/
│   └── audit_pipeline.py
├── results_chart.png
└── README.md
```

## Setup (recommended)

```bash
cd /home/ejiaka/clinical-ai-redteam-audit
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Ollama

Ollama is a system dependency (not installable via pip). Install it, then pull the model:

```bash
ollama pull llama3
```

## Run

Quick smoke test (no Ollama calls):

```bash
python3 scripts/audit_pipeline.py --offline-mock --n 1000 --limit 20
```

Full run (1,000 cases via Ollama):

```bash
python3 scripts/audit_pipeline.py --n 1000 --sleep-s 0.1
```

## Outputs

- `data/clinical_audit_seed_data.csv`: committed (21-row seed dataset)
- `data/large_audit_dataset.csv`: committed (1,000-row augmented dataset)
- `data/final_audit_results.csv`: committed (latest audit run output)
- `results_chart.png`: produced chart

## Seed data note

This repo commits the seed file at:

- `data/clinical_audit_seed_data.csv`

You can still override the path via `--seed-csv`.
