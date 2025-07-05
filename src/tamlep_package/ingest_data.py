"""
Provides functions to download the datasets, perform EDA and split the
data into training and test sets.

Command-line Usage
------------------
To use this module from the command line, run:
    python ingest_data.py [OPTIONS]

Command-line Options
--------------------
--raw-data-dir
    Raw data directory name
--processed-data-dir
    Processed data directory name
--log-level : {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    Logging level
--log-file-path
    Path to log file
--no-console-log
    Flag to turn off logs to console

Command Line Example
--------------------
```
python ingest_data.py --log-level INFO \
--log-file-path ./logs/app.log \
--no-console-log
```
"""

import logging
import os
import tarfile
import urllib
import urllib.request
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split

from tamlep_package.config_manager import config

logger = logging.getLogger(__name__)

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

HOUSING_PATH = os.path.join("datasets", "housing")
RAW_DATA_PATH = config.raw_data_path
PROCESSED_DATA_PATH = config.processed_data_path
ARTIFACTS_PATH = config.artifacts_path


def fetch_housing_data(housing_url=HOUSING_URL, download_folder=RAW_DATA_PATH):
    """
    Download housing dataset from the provided URL

    Parameters
    ----------
    housing_url : str, optional
        The URL to download data. Defaults to:
        https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.tgz
    download_folder: str, optional
        The folder to download the data. Defaults to raw data folder
        in the configuration.
    """
    download_folder = Path(download_folder)
    housing_data_path = download_folder / "housing.csv"
    if housing_data_path.exists():
        logger.info(f"Housing data already exists at {housing_data_path}")
    else:
        download_folder.mkdir(exist_ok=True, parents=True)
        tgz_path = os.path.join(download_folder, "housing.tgz")
        urllib.request.urlretrieve(housing_url, tgz_path)
        housing_tgz = tarfile.open(tgz_path)
        housing_tgz.extractall(path=download_folder)
        housing_tgz.close()
        logger.info(f"Data downloaded to {download_folder}")
    return housing_data_path


def income_cat_proportions(data: pd.DataFrame) -> pd.Series:
    """
    Get the proportion of records for each income category

    Parameters
    ----------
    data : pd.DataFrame
        The pandas dataframe containing the data.
        Income category column should be named `income_cat`

    Returns
    -------
    pd.Series
        Pandas series with income category wise data proportions.
    """
    return data["income_cat"].value_counts() / len(data)


def split_housing_data(
    housing_data_path: Path,
    processed_data_dir_path: Path = PROCESSED_DATA_PATH,
):
    """
    Split the housing data into training and test sets

    Parameters
    ----------
    housing_data_path : pathlib.Path
        Path to raw dataset.
    processed_data_dir_path : pathlib.Path, optional
        Path to save the processed and split data.
        Defaults to processed data folder in configuration.
    """
    if (processed_data_dir_path / "train_set.csv").exists():
        logger.info("Stratified sampling data set already exists.")
        return
    else:
        housing_data_path = Path(housing_data_path)
        housing_df = pd.read_csv(housing_data_path)
        housing_df["income_cat"] = pd.cut(
            housing_df["median_income"],
            bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
            labels=[1, 2, 3, 4, 5],
        )
        # Stratified splitting
        split = StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        )
        for train_index, test_index in split.split(
            housing_df, housing_df["income_cat"]
        ):
            strat_train_set = housing_df.loc[train_index]
            strat_test_set = housing_df.loc[test_index]
        # Random splitting
        rand_train_set, rand_test_set = train_test_split(
            housing_df, test_size=0.2, random_state=42
        )
        # Compare stratified splitting to random splitting
        compare_props = pd.DataFrame(
            {
                "Overall": income_cat_proportions(housing_df),
                "Stratified": income_cat_proportions(strat_test_set),
                "Random": income_cat_proportions(rand_test_set),
            }
        ).sort_index()
        compare_props["Rand. %error"] = (
            100 * compare_props["Random"] / compare_props["Overall"] - 100
        )
        compare_props["Strat. %error"] = (
            100 * compare_props["Stratified"] / compare_props["Overall"] - 100
        )
        logger.info(
            "Comparison of stratified splitting strategy to random "
            + "splitting strategy:\n{}\n\n".format(compare_props)
        )
        logger.info(
            "Saving stratified sampling data set to processed data folder..."
        )
        processed_data_dir_path.mkdir(exist_ok=True, parents=True)
        strat_train_set.to_csv(
            processed_data_dir_path / "train_set.csv", index=True
        )
        strat_test_set.to_csv(
            processed_data_dir_path / "test_set.csv", index=True
        )


def get_correlation_matrix(
    data_df: pd.DataFrame,
    order_by: List[str] | str = "median_house_value",
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Get the correlation matrix for the data.

    Returns the correlation matrix and also saves it to artifacts folder.

    Parameters
    ----------
    data_df: pd.DataFrame
        The dataframe to compute correlations
    order_by: List[str] | str, optional
        The column or list of columns to order the results.
        Defaults to `median_house_value`
    ascending: bool, optional
        Sort order. Sort in descending order bu default

    Returns
    -------
    pd.DataFrame
        The dataframe containing the correlation matrix.
        The dataframe will also be saved as .csv files inside
        the artifacts folder.
    """
    if "ocean_proximity" in data_df.columns:
        data_df = data_df.drop(columns=["ocean_proximity"])
    corr_matrix = data_df.corr()
    corr_matrix.sort_values(by=order_by, ascending=ascending, inplace=True)
    ARTIFACTS_PATH.mkdir(exist_ok=True, parents=True)
    corr_matrix_path = ARTIFACTS_PATH / "correlation_matrix.csv"
    corr_matrix.to_csv(corr_matrix_path, index=False)
    logger.info(f"Correlation matrix saved to {corr_matrix_path}.")
    return corr_matrix


def create_scatter_plot(
    data_df: pd.DataFrame,
    x: str = "longitude",
    y: str = "latitude",
    alpha: float = 0.1,
    figsize: tuple = (10, 10),
    show_plot: bool = False,
):
    """
    Create a scatter plot of housing data by longitude and latitude.

    Parameters
    ----------
    data_df : pd.DataFrame
        The dataframe containing the data.
    x : str, optional
        Defaults to "longitude"
    y : str, optional
        Defaults to "latitude"
    alpha : float, optional
        Defaults to 0.1
    figsize : tuple, optional
        Defaults to (10, 10)
    show_plot : bool, optional
        Whether to show the plot. Defaults to False
    """
    title = f"Scatter plot of {x} and {y}"
    _, ax = plt.subplots(figsize=figsize)
    data_df.plot(x=x, y=y, kind="scatter", alpha=alpha, title=title, ax=ax)
    ARTIFACTS_PATH.mkdir(exist_ok=True, parents=True)
    scatter_plot_path = ARTIFACTS_PATH / f"scatter_{x}_{y}.png"
    plt.savefig(scatter_plot_path, dpi=300, bbox_inches="tight")
    logger.info(f"Scatter plot saved to {scatter_plot_path}")
    if show_plot:
        plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-data-dir",
        help="raw data directory name",
        # default=config.raw_data_path,
    )
    parser.add_argument(
        "--processed-data-dir",
        help="processed data directory name",
    )
    parser.add_argument(
        "--log-level",
        help="Logging level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    parser.add_argument(
        "--log-file-path",
        help="path to log file",
    )
    parser.add_argument(
        "--no-console-log",
        help="turn off logs to console",
        action="store_true",
    )
    args = parser.parse_args()
    if args.log_file_path:
        log_dir = Path(args.log_file_path).parent
        log_dir.mkdir(exist_ok=True, parents=True)
        log_file = Path(args.log_file_path).name
        add_file_logger = True
    else:
        log_dir = None
        log_file = None
    if args.raw_data_dir or args.processed_data_dir or args.log_file_path:
        config.set_up_paths(
            raw_data_dir=args.raw_data_dir,
            processed_data_dir=args.processed_data_dir,
            log_dir=log_dir,
        )
    if args.log_level or args.log_file_path or args.no_console_log:
        config.set_up_logging(
            log_level=args.log_level,
            log_file=log_file,
            add_console_logger=not args.no_console_log,
            add_file_logger=add_file_logger,
        )
    # Refresh all modules with updated configuration if any command line
    # arguments were provided.
    if any(vars(args).values()):
        config.refresh_configuration()
    logger.info(f"Raw data path: {config.raw_data_path}")
    logger.info(f"Processed data path: {config.processed_data_path}")
    logger.info(f"Logging level: {config.log_level}")
    if args.log_file_path:
        logger.info(f"Log file path: {config.log_path / config.log_file}")
    logger.info(f"Log handlers configured: {config.root_handlers}")

    # Download raw data
    try:
        housing_data_path = fetch_housing_data()
    except Exception as e:
        logger.error(f"Error downloading data: {e}")
        raise e
    # Split data into training and test sets
    split_housing_data(housing_data_path)
    # Read train and test set
    train_df = pd.read_csv(PROCESSED_DATA_PATH / "train_set.csv", index_col=0)
    test_df = pd.read_csv(PROCESSED_DATA_PATH / "test_set.csv", index_col=0)
    # Get correlation matrix
    get_correlation_matrix(train_df)
    # Create scatter plot of longitude and latitude
    create_scatter_plot(train_df, x="longitude", y="latitude")
