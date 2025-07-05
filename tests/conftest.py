import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def test_data_dir_path(tmp_path_factory):
    test_data_path: Path = tmp_path_factory.mktemp("test_data")
    return test_data_path.as_posix()


@pytest.fixture(scope="session")
def log_dir(test_data_dir_path):
    return os.path.join(test_data_dir_path, "logs")


@pytest.fixture(scope="session")
def raw_data_dir(test_data_dir_path):
    return os.path.join(test_data_dir_path, "datasets", "raw_data")


@pytest.fixture(scope="session")
def processed_data_dir(test_data_dir_path):
    return os.path.join(test_data_dir_path, "datasets", "processed_data")


@pytest.fixture(scope="session")
def log_file_path(test_data_dir_path):
    return os.path.join(test_data_dir_path, "logs", "app_log.log")


@pytest.fixture(scope="session")
def artifacts_path(test_data_dir_path):
    return os.path.join(test_data_dir_path, "artifacts")


@pytest.fixture(scope="session")
def models_path(test_data_dir_path):
    return os.path.join(test_data_dir_path, "models")
