from __future__ import annotations

from pathlib import Path

from sklearn.pipeline import Pipeline

from src.builder import build_paths
from src.constants import MODEL_FEATURES, TARGET
from src.models import ModelDefinition, PipelinePaths
from src.training import (
    build_model_pipeline,
    build_pipeline,
    build_preprocessor,
    model_catalog,
)
from src.training.tune_gradient_boosting import tune_gradient_boosting
from src.training.tune_random_forest import tune_random_forest
from src.utils.pipeline_features import normalize_skill_string, skill_count


def test_pipeline_constants_are_immutable_sequences() -> None:
    assert TARGET == "annual_salary_usd"
    assert isinstance(MODEL_FEATURES, tuple)
    assert "required_skills" in MODEL_FEATURES


def test_path_builder_constructs_and_creates_pipeline_directories(
    tmp_path: Path,
) -> None:
    paths = build_paths(tmp_path, tmp_path / "generated")

    assert isinstance(paths, PipelinePaths)
    assert paths.output == tmp_path / "generated"
    assert paths.artifacts.is_dir()
    assert paths.prediction.is_dir()


def test_model_catalog_uses_dataclass_definitions_and_fresh_constructors() -> None:
    definitions = model_catalog()

    random_forest = definitions["Random Forest"]
    assert isinstance(random_forest, ModelDefinition)
    assert random_forest.build_estimator() is not random_forest.build_estimator()


def test_training_functions_are_split_into_focused_modules() -> None:
    training_directory = Path(__file__).resolve().parents[1] / "src" / "training"
    expected_modules = {
        "compare_models.py",
        "build_model_pipeline.py",
        "build_pipeline.py",
        "build_preprocessor.py",
        "model_catalog.py",
        "monthly_temporal_folds.py",
        "out_of_fold_absolute_errors.py",
        "regression_metrics.py",
        "select_best_model.py",
        "tune_gradient_boosting.py",
        "tune_random_forest.py",
    }

    assert expected_modules <= {path.name for path in training_directory.glob("*.py")}
    assert not (training_directory / "model_training.py").exists()
    assert callable(tune_random_forest)
    assert callable(tune_gradient_boosting)


def test_training_owns_machine_learning_object_construction() -> None:
    assert build_model_pipeline.__module__ == "src.training.build_model_pipeline"
    assert build_pipeline.__module__ == "src.training.build_pipeline"
    assert build_preprocessor.__module__ == "src.training.build_preprocessor"

    builder_directory = Path(__file__).resolve().parents[1] / "src" / "builder"
    assert not (builder_directory / "model_builder.py").exists()
    assert not (builder_directory / "preprocessor_builder.py").exists()


def test_training_package_has_no_pipeline_or_ui_dependencies() -> None:
    training_directory = Path(__file__).resolve().parents[1] / "src" / "training"
    forbidden_imports = (
        "from src.builder",
        "from src.config",
        "from src.pages",
        "from src.utils",
        "from pipeline",
    )

    for module in training_directory.glob("*.py"):
        source = module.read_text(encoding="utf-8")
        assert not any(item in source for item in forbidden_imports), module.name


def test_model_builder_constructs_a_training_pipeline() -> None:
    ridge = model_catalog()["Ridge Regression"]

    pipeline = build_model_pipeline(ridge)

    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps) == ["preprocessor", "model"]


def test_pipeline_feature_helpers_normalize_and_deduplicate_skills() -> None:
    value = "Python | SQL|Python|  Machine Learning  "

    assert normalize_skill_string(value) == "Python|SQL|Machine Learning"
    assert skill_count(value) == 3


def test_pipeline_package_is_canonical_and_entrypoints_delegate() -> None:
    import pipeline
    import pineline
    from src import pipeline as pipeline_package

    project_root = Path(__file__).resolve().parents[1]
    assert (project_root / "pipeline.py").is_file()
    assert (project_root / "src" / "pipeline" / "__init__.py").is_file()
    assert not (project_root / "src" / "pipeline" / "pipeline.py").exists()
    assert pineline.main is pipeline.main is pipeline_package.main
    assert pipeline_package.PROJECT_ROOT == project_root


def test_pipeline_stages_are_exposed_from_the_package() -> None:
    from src import pipeline

    package_directory = Path(__file__).resolve().parents[1] / "src" / "pipeline"
    expected_modules = {
        2: "project_scope_initial_inspection",
        3: "data_quality_check",
        4: "corrupted_row_removal",
        5: "contradictory_feature_investigation",
        6: "feature_selection_leakage_prevention",
        7: "correlation_encoding_analysis",
        8: "train_test_split",
        9: "model_training_comparison",
        10: "best_model_selection_feature_importance_review",
        11: "save_deployable_pipeline_metadata",
        12: "streamlit_salary_prediction_dashboard",
    }

    assert {f"{module}.py" for module in expected_modules.values()} <= {
        path.name for path in package_directory.glob("*.py")
    }
    assert not any(package_directory.glob("stage*.py"))
    for module in expected_modules.values():
        stage = getattr(pipeline, module)
        assert stage.__name__ == module
        assert stage.__module__ == f"src.pipeline.{module}"
    assert not any(hasattr(pipeline, f"run_stage{number}") for number in range(2, 13))


def test_pipeline_builds_timestamped_asset_output_directory(tmp_path: Path) -> None:
    from src.pipeline import build_asset_output_dir

    output = build_asset_output_dir(tmp_path, "20260819T120000Z")

    assert output == tmp_path / "output-20260819T120000Z"
    assert output.is_dir()
