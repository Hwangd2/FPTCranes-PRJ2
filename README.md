# AI Job Market Salary Prediction

This project implements the 12-stage workflow from the supplied **Document_QD Project KHDL&AI** and regenerates all requested evidence packs from `data/ai_jobs_market_2025_2026.csv`.

## Run

```bash
pip install -r requirements.txt
python pipeline.py
streamlit run streamlit.py
```

Pipeline progress uses Rich terminal logging. `INFO` is the default and shows each
stage, model family, tuning candidate, and final evaluation. Use `DEBUG` for fold
sizes, metrics, preprocessing details, and generated artifact paths:

```bash
python pipeline.py --log-level DEBUG
```

## Streamlit login

Local demo only: `admin / AIJob2026!`

For shared deployment, override the demo credential with either Streamlit secrets (`auth.username`, `auth.password_sha256`) or environment variables:

- `AIJOB_APP_USER`
- `AIJOB_APP_PASSWORD_SHA256`

## Output packs

1. `outputs/01_data_basic_clean/`
2. `outputs/02_data_ready_for_machine_learning/`
3. `outputs/03_model_comparison/`
4. `outputs/04_best_model_and_feature_importance/`
5. `outputs/05_salary_prediction/`

The final serialized inference bundle is in `artifacts/model_bundle.joblib`, and the checked-in
presentation report is in `docs/AI_Job_Market_Salary_Prediction_Report.docx`.

## Scientific boundary

The supplied snapshot contains contradictory and synthetic-looking structure. The model is suitable for academic supervised-regression and controlled scenario exploration; importance values must not be interpreted as causal real-world salary economics without external validation.

## Development governance

All specifications, plans, implementation tasks, and reviews are governed by
`.specify/memory/constitution.md`. In particular, changes must preserve temporal leakage
safety, reproducible artifacts, failing-first verification, offline training, read-only
Streamlit reporting, honest interpretation, and secure credential handling.

## Streamlit structure and theme

`streamlit.py` is the authenticated application router. Page scripts live in `src/pages/`,
shared UI elements in `src/components/`, and testable artifact, authentication, formatting,
and prediction helpers in `src/utils/`.

The default theme is light and is configured in `.streamlit/config.toml`. Light and dark
palettes are both defined, so users can switch mode from Streamlit's Settings menu. To make
dark mode the deployment default, change `theme.base` from `"light"` to `"dark"` and restart
the app. The UI uses native Streamlit elements so both palettes remain readable.

## Offline pipeline structure

`pipeline.py` remains the project-root command wrapper. `pineline.py` is a backward-compatible
alias, while the implementation lives in the `src.pipeline` package:

- `src/constants/pipeline.py`: immutable feature, split, stage, and visualization constants.
- `src/models/pipeline.py`: dataclasses for paths, model definitions, and training selection.
- `src/builder/`: generic project path construction.
- `src/training/`: one focused module per training operation, including model catalog,
  preprocessing and model-pipeline construction, temporal folds, comparison, tuning,
  selection, metrics, and out-of-fold errors.
- `src/pipeline/__init__.py`: the offline orchestrator and Stage 1 data-loading workflow; it
  defines but does not invoke `main()`.
- `src/pipeline/`: one descriptively named module per remaining stage, such as
  `data_quality_check.py` and `model_training_comparison.py`; each module's public function
  has the same name as its file.

Each run writes generated asset images beneath `assets/output-<UTC timestamp>/`. The dashboard
selects the newest timestamped pipeline diagram and retains legacy fallbacks for older checkouts.
