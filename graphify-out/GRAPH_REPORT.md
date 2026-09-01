# Graph Report - src  (2026-08-21)

## Corpus Check
- Corpus is ~7,807 words - fits in a single context window. You may not need a graph.

## Summary
- 221 nodes · 268 edges · 33 communities (25 shown, 8 thin omitted)
- Extraction: 81% EXTRACTED · 19% INFERRED · 0% AMBIGUOUS · INFERRED: 51 edges (avg confidence: 0.85)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Model Comparison Training
- Pipeline Orchestration
- Paths Quality Dashboard
- Leakage Feature Policy
- Tuning Model Selection
- Best Model Importance
- Encoding Temporal Split
- Pipeline Plotting
- Artifact Loading
- Streamlit Authentication
- App Config Paths
- Sidebar Navigation
- Display Formatting
- Prediction Helpers
- Page Header
- Page Styles
- Components Package
- Pages Package
- Training Package
- Utils Package

## God Nodes (most connected - your core abstractions)
1. `main()` - 19 edges
2. `PipelinePaths` - 13 edges
3. `best_model_selection_feature_importance_review()` - 12 edges
4. `contradictory_feature_investigation()` - 9 edges
5. `streamlit_salary_prediction_dashboard()` - 9 edges
6. `build_pipeline()` - 9 edges
7. `model_training_comparison()` - 8 edges
8. `evaluate_candidates()` - 8 edges
9. `select_best_model()` - 8 edges
10. `_save_figure()` - 8 edges

## Surprising Connections (you probably didn't know these)
- `main()` --calls--> `build_paths()`  [INFERRED]
  pipeline/__init__.py → builder/path_builder.py
- `best_model_selection_feature_importance_review()` --calls--> `regression_metrics()`  [INFERRED]
  pipeline/best_model_selection_feature_importance_review.py → training/regression_metrics.py
- `best_model_selection_feature_importance_review()` --calls--> `select_best_model()`  [INFERRED]
  pipeline/best_model_selection_feature_importance_review.py → training/select_best_model.py
- `best_model_selection_feature_importance_review()` --calls--> `plot_best_outputs()`  [INFERRED]
  pipeline/best_model_selection_feature_importance_review.py → utils/pipeline_plots.py
- `contradictory_feature_investigation()` --indirect_call--> `salary_tier_expected()`  [INFERRED]
  pipeline/contradictory_feature_investigation.py → utils/pipeline_features.py

## Import Cycles
- None detected.

## Communities (33 total, 8 thin omitted)

### Community 0 - "Model Comparison Training"
Cohesion: 0.07
Nodes (21): ModelDefinition, RegressorMixin, Construct an unfitted estimator for one training run., model_training_comparison(), Any, DataFrame, Stage 9: Model Training & Comparison., build_model_pipeline() (+13 more)

### Community 1 - "Pipeline Orchestration"
Cohesion: 0.12
Nodes (20): Logger, Namespace, build_asset_output_dir(), configure_logging(), main(), _parse_args(), Console, Path (+12 more)

### Community 2 - "Paths Quality Dashboard"
Cohesion: 0.10
Nodes (17): build_paths(), Path, Build the pipeline path object and ensure writable output directories exist., PipelinePaths, corrupted_row_removal(), DataFrame, Series, Stage 4: Corrupted Row Removal. (+9 more)

### Community 3 - "Leakage Feature Policy"
Cohesion: 0.12
Nodes (17): contradictory_feature_investigation(), DataFrame, Series, Stage 5: Contradictory-Feature Investigation., feature_selection_leakage_prevention(), Stage 6: Feature Selection & Leakage Prevention., project_scope_initial_inspection(), DataFrame (+9 more)

### Community 4 - "Tuning Model Selection"
Cohesion: 0.12
Nodes (14): evaluate_candidates(), Any, DataFrame, RegressorMixin, Evaluate one model family's parameter candidates with temporal folds., Any, DataFrame, select_best_model() (+6 more)

### Community 5 - "Best Model Importance"
Cohesion: 0.13
Nodes (13): TrainingSelection, best_model_selection_feature_importance_review(), Any, DataFrame, ndarray, Stage 10: Best Model Selection & Feature Importance Review., build_pipeline(), Pipeline (+5 more)

### Community 6 - "Encoding Temporal Split"
Cohesion: 0.12
Nodes (12): ColumnTransformer, OneHotEncoder, correlation_encoding_analysis(), DataFrame, Stage 7: Correlation Encoding & Analysis., DataFrame, Stage 8: Train-Test Split., train_test_split() (+4 more)

### Community 7 - "Pipeline Plotting"
Cohesion: 0.36
Nodes (11): Figure, chart_pipeline_12(), plot_basic_outputs(), plot_best_outputs(), plot_correlation(), plot_model_comparison(), DataFrame, ndarray (+3 more)

### Community 8 - "Artifact Loading"
Cohesion: 0.29
Nodes (11): load_csv(), load_json(), _load_model(), Any, DataFrame, Path, Load an optional CSV and invalidate the cache when the file changes., Load an optional JSON object and invalidate the cache when it changes. (+3 more)

### Community 9 - "Streamlit Authentication"
Cohesion: 0.31
Nodes (7): Stop the current rerun until the user authenticates., require_authentication(), AuthSettings, password_hash(), Any, Resolve credentials from environment, Streamlit secrets, or local demo defaults., resolve_auth_settings()

### Community 10 - "App Config Paths"
Cohesion: 0.40
Nodes (4): Config, Path, Resolve the newest timestamped pipeline diagram with legacy fallbacks., resolve_latest_pipeline_image()

### Community 12 - "Display Formatting"
Cohesion: 0.67
Nodes (3): format_money(), model_comparison_comment(), DataFrame

## Knowledge Gaps
- **1 isolated node(s):** `Config`
  These have ≤1 connection - possible missing edges or undocumented components.
- **8 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `main()` connect `Pipeline Orchestration` to `Model Comparison Training`, `Paths Quality Dashboard`, `Leakage Feature Policy`, `Best Model Importance`, `Encoding Temporal Split`?**
  _High betweenness centrality (0.197) - this node is a cross-community bridge._
- **Why does `best_model_selection_feature_importance_review()` connect `Best Model Importance` to `Model Comparison Training`, `Pipeline Orchestration`, `Paths Quality Dashboard`, `Tuning Model Selection`, `Pipeline Plotting`?**
  _High betweenness centrality (0.171) - this node is a cross-community bridge._
- **Why does `PipelinePaths` connect `Paths Quality Dashboard` to `Model Comparison Training`, `Pipeline Orchestration`, `Leakage Feature Policy`, `Best Model Importance`, `Encoding Temporal Split`?**
  _High betweenness centrality (0.123) - this node is a cross-community bridge._
- **Are the 14 inferred relationships involving `main()` (e.g. with `build_paths()` and `best_model_selection_feature_importance_review()`) actually correct?**
  _`main()` has 14 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `best_model_selection_feature_importance_review()` (e.g. with `build_pipeline()` and `out_of_fold_absolute_errors()`) actually correct?**
  _`best_model_selection_feature_importance_review()` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `contradictory_feature_investigation()` (e.g. with `experience_bucket()` and `salary_tier_expected()`) actually correct?**
  _`contradictory_feature_investigation()` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `streamlit_salary_prediction_dashboard()` (e.g. with `main()` and `save_json()`) actually correct?**
  _`streamlit_salary_prediction_dashboard()` has 3 INFERRED edges - model-reasoned connections that need verification._