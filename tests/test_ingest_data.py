import os
from pathlib import Path

import pytest


@pytest.mark.depends(
    on=["tests/test_installation.py::test_package_installation"]
)
@pytest.mark.depends(name="test_fetch_housing_data")
def test_fetch_housing_data(raw_data_dir):
    import pandas as pd

    from tamlep_package.ingest_data import fetch_housing_data

    downloaded_data_path = fetch_housing_data(download_folder=raw_data_dir)
    assert os.path.exists(downloaded_data_path), "Data not downloaded"
    raw_data_df = pd.read_csv(downloaded_data_path)
    # assert raw_data_df.shape == (20640, 10)
    # We will use a random sample of 200 records for testing purpose
    raw_data_df = raw_data_df.sample(
        n=200, replace=False, ignore_index=True, random_state=100, axis=0
    )
    raw_data_df.to_csv(downloaded_data_path, index=False)


@pytest.mark.depends(on=["test_fetch_housing_data"])
@pytest.mark.depends(name="test_split_housing_data")
def test_split_housing_data(raw_data_dir, processed_data_dir):
    from tamlep_package.ingest_data import split_housing_data

    housing_data_path = os.path.join(raw_data_dir, "housing.csv")
    housing_data_path = Path(housing_data_path)
    processed_data_dir = Path(processed_data_dir)
    split_housing_data(
        housing_data_path=housing_data_path,
        processed_data_dir_path=processed_data_dir,
    )
    assert os.path.exists(
        processed_data_dir / "train_set.csv"
    ), "Training set not created"
    assert os.path.exists(
        processed_data_dir / "test_set.csv"
    ), "Test set not created"


@pytest.mark.depends(on=["test_split_housing_data"])
def test_create_scatter_plot(processed_data_dir, artifacts_path, monkeypatch):
    import pandas as pd

    from tamlep_package import ingest_data
    from tamlep_package.ingest_data import create_scatter_plot

    processed_data_dir = Path(processed_data_dir)
    artifacts_path = Path(artifacts_path)
    train_df = pd.read_csv(processed_data_dir / "train_set.csv", index_col=0)
    # monkey patch config. to artifacts_path
    monkeypatch.setattr(ingest_data, "ARTIFACTS_PATH", artifacts_path)
    create_scatter_plot(train_df)

    assert os.path.exists(
        artifacts_path / "scatter_longitude_latitude.png"
    ), "Scatter plot not created"
