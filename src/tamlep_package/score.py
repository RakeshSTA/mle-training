"""
Score trained models on the test dataset.


Command-line Usage
------------------
To use this module from the command line, run:
    python score.py [OPTIONS]


Command-line Options
--------------------
--test-data-path: str, optional
    Path to test dataset
--models-dir-path: str,
    Path to saved model objects as joblib files
--artifacts-dir-path
    Path to artifacts folder to save scoring results
--log-level : {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    Logging level
--log-file-path
    Path to log file
--no-console-log
    Flag to turn off logs to console


Examples
--------
```
python score.py --log-level INFO \
--log-file-path ./logs/app.log \
--no-console-log
```

"""

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from tamlep_package.config_manager import config
from tamlep_package.utils import timer

logger = logging.getLogger(__name__)


@timer
def score_models(test_data_file: Optional[str] = None):
    """
    Score the trained models on the test dataset.

    Parameters
    ----------
    test_data_file : str, optional
        The name of the test data file.
        Defaults to "test_set.csv"
        Reads the test data from processed data folder set
        in the configuration.
        The scoring results are saved to artifacts folder set
        in the configuration. Models are sorted by ascending
        order of mean squared error & mean absolute error.

    Returns
    -------
    pd.DataFrame
        The pandas dataframe containing mean squared error &
        mean absolute error for all trained models. Models are
        sorted by ascending order of mean squared error &
        mean absolute error.

    """
    logger.info("Scoring models on test set")
    logger.info("Loading test dataset")
    if test_data_file is None:
        test_data_file = "test_set.csv"
    test_data_path = config.processed_data_path / test_data_file
    logger.info(f"Read test data from {test_data_path}")
    test_df = pd.read_csv(
        test_data_path,
        index_col=0,
    )
    logger.info("Read test data")
    X_test = test_df.drop(columns=["median_house_value"])
    y_test = test_df["median_house_value"].copy()
    scoring_results = {}
    for model_path in config.models_path.glob("*.joblib"):
        model_name = model_path.stem
        logger.info(f"Scoring {model_name}!")
        model_pipeline = joblib.load(model_path)
        y_pred = model_pipeline.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        scoring_results[model_name] = {
            "rmse": np.round(rmse, 2),
            "mae": np.round(mae, 2),
        }
    scoring_results_df = pd.DataFrame.from_dict(
        scoring_results, orient="index"
    )
    scoring_results_df.index.name = "model"
    scoring_results_df.reset_index(inplace=True, drop=False)
    scoring_results_df.sort_values(
        by=["rmse", "mae"], ascending=True, inplace=True, ignore_index=True
    )
    results_path = Path(config.artifacts_path / "model_scoring_results.csv")
    results_path.parent.mkdir(exist_ok=True, parents=True)
    scoring_results_df.to_csv(results_path, index=False)
    logger.info(f"Results saved to: {results_path.as_posix()}")
    return scoring_results_df


if __name__ == "__main__":
    # from tamlep_package.utils import attach_debugger
    # attach_debugger()
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-data-path",
        help="Path to training dataset",
    )
    parser.add_argument(
        "--models-dir-path",
        help="Path to save output models",
    )
    parser.add_argument(
        "--artifacts-dir-path",
        help="Path to artifacts folder",
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
    if args.test_data_path:
        test_data_file = Path(args.test_data_path).name
        test_data_dir = Path(args.test_data_path).parent
    else:
        test_data_file = None
        test_data_dir = None
    if (
        args.models_dir_path
        or args.artifacts_dir_path
        or args.log_file_path
        or args.test_data_path
    ):
        config.set_up_paths(
            models_dir=args.models_dir_path,
            artifacts_dir=args.artifacts_dir_path,
            log_dir=log_dir,
            processed_data_dir=test_data_dir,
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
    logger.info(f"Models path: {config.models_path}")
    logger.info(f"Artifacts path: {config.artifacts_path}")
    logger.info(f"Logging level: {config.log_level}")
    if args.log_file_path:
        logger.info(f"Log file path: {config.log_path / config.log_file}")
    logger.info(f"Log handlers configured: {config.root_handlers}")
    # score models
    scoring_results_df = score_models(test_data_file)
    logger.info("Scoring complete!")
    logger.info(f"\n# Models sorted by rank:\n{scoring_results_df}\n")
