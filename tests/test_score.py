from pathlib import Path

import pytest


@pytest.mark.depends(on=["tests/test_train.py::test_train_models"])
@pytest.mark.depends(name="test_score_models")
def test_score_models(
    processed_data_dir, artifacts_path, models_path, monkeypatch
):
    from tamlep_package.config_manager import config

    processed_data_dir = Path(processed_data_dir)
    artifacts_path = Path(artifacts_path)
    models_path = Path(models_path)
    monkeypatch.setattr(config, "artifacts_path", artifacts_path)
    monkeypatch.setattr(config, "processed_data_path", processed_data_dir)
    monkeypatch.setattr(config, "models_path", models_path)
    test_data_file = "test_set.csv"
    from tamlep_package.score import score_models

    scoring_results_df = score_models(test_data_file)
    reqd_cols = ["model", "rmse", "mae"]
    error_msg = (
        "Scoring results does not have all required columns!\n"
        + "Expected columns: {}".format(", ".join(reqd_cols))
    )
    assert all(c in scoring_results_df.columns for c in reqd_cols), error_msg
    trained_model = [
        "rf_with_random_search",
        "rf_with_grid_search",
        "linear_regression",
        "decision_tree",
    ]
    error_msg = (
        "Scoring results does not have all required models!\n"
        + "Expected models: {}".format(", ".join(trained_model))
    )
    assert all(
        m in scoring_results_df["model"].values for m in trained_model
    ), error_msg
    scoring_results_path = artifacts_path / "model_scoring_results.csv"
    assert (
        scoring_results_path.exists()
    ), f"Scoring results file does not exist at {scoring_results_path}"
