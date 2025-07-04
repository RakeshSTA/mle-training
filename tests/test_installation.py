"""
Test tamlep_package installation
"""

import pytest


@pytest.mark.depends(name="test_package_installation")
def test_package_installation():
    try:
        from tamlep_package.config_manager import config
        from tamlep_package.ingest_data import fetch_housing_data
        from tamlep_package.score import score_models
        from tamlep_package.train import train_models

        config.refresh_configuration()
        print("Package installed successfully")
    except Exception as e:
        print(f"Unable to import all package modules.\nError: {str(e)}")
        raise
