# Clinical AI Red Team Audit (End-to-End)

This repo contains a Python pipeline that:
- Generates a 1,000-row synthetic audit dataset from `clinical_audit_seed_data.csv` (with ±2% jitter on Na/K/Cl/Urea/Creat/pH)
- Audits each case against a local Ollama Llama 3 model
- Grades outputs using the HIST protocol
- Produces a scorecard and a LinkedIn-ready chart

## Setup (recommended)

```bash
cd /home/ejiaka/clinical-ai-redteam-audit
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Ollama

Ollama is a system dependency (not installable via pip). Install it and pull the model:

```bash
ollama pull llama3
```

## Run

Quick pipeline smoke test (no Ollama calls):

```bash
python3 run_clinical_audit_pipeline.py --offline-mock --n 50 --limit 20
```

Full run (1,000 cases via Ollama):

```bash
python3 run_clinical_audit_pipeline.py --n 1000 --sleep-s 0.1
```

## Outputs

Generated outputs are currently git-ignored until validation:
- `large_audit_dataset.csv`
- `final_audit_results.csv`
- `clinical_ai_safety_chart.png`

