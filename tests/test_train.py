from pathlib import Path

import pytest


@pytest.mark.depends(on=["tests/test_ingest_data.py::test_split_housing_data"])
@pytest.mark.depends(name="test_train_models")
def test_train_models(
    processed_data_dir, artifacts_path, models_path, monkeypatch
):
    from tamlep_package.config_manager import config

    processed_data_dir = Path(processed_data_dir)
    artifacts_path = Path(artifacts_path)
    models_path = Path(models_path)
    monkeypatch.setattr(config, "artifacts_path", artifacts_path)
    monkeypatch.setattr(config, "processed_data_path", processed_data_dir)
    monkeypatch.setattr(config, "models_path", models_path)
    train_data_file = "train_set.csv"
    from tamlep_package.train import train_models

    train_models(train_data_file)
    model_outputs = [m.name for m in models_path.glob("*.joblib")]
    assert (
        "decision_tree.joblib" in model_outputs
    ), "Decision Tree model not trained"
    assert (
        "linear_regression.joblib" in model_outputs
    ), "Linear Regression model not trained"
    assert (
        "rf_with_grid_search.joblib" in model_outputs
    ), "Random Forest model with Grid Search not trained"
    assert (
        "rf_with_random_search.joblib" in model_outputs
    ), "Random Forest model with Random Search not trained"
    feature_importance_files = [f.name for f in artifacts_path.glob("*.csv")]
    assert (
        "decision_tree_feature_importance.csv" in feature_importance_files
    ), "Decision Tree feature importance not saved"
    assert (
        "linear_regression_feature_importance.csv" in feature_importance_files
    ), "Linear Regression feature importance not saved"
    assert (
        "rf_with_grid_search_feature_importance.csv"
        in feature_importance_files
    ), "Random Forest feature importance not saved"
    assert (
        "rf_with_random_search_feature_importance.csv"
        in feature_importance_files
    ), "Random Forest feature importance not saved"
