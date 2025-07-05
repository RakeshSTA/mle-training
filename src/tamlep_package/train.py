"""
Train Machine learning models on housing dataset.

Command-line Usage
------------------
To use this module from the command line, run:
    python train.py [OPTIONS]

Command-line Options
--------------------
--train-data-path: str, optional
    Path to training dataset
--models-dir-path: str,
    Path to save output model objects as joblib file
--artifacts-dir-path
    Path to save training artifacts (feature importance data)
--log-level : {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    Logging level
--log-file-path
    Path to log file
--no-console-log
    Flag to turn off logs to console

Examples
--------
```
python train.py --log-level INFO \
--log-file-path ./logs/app.log \
--no-console-log
```
"""

import logging
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from scipy.stats import randint
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

from tamlep_package.config_manager import config
from tamlep_package.utils import set_seed
from tamlep_package.utils import timer

logger = logging.getLogger(__name__)


pre_proc_steps = [
    (
        "one_hot_enc",
        OneHotEncoder(drop="first", sparse_output=False),
        ["ocean_proximity"],
    ),
    ("imputer", SimpleImputer(strategy="median"), ["total_bedrooms"]),
]

pre_proc_pipeline = ColumnTransformer(
    transformers=pre_proc_steps, remainder="passthrough", n_jobs=-1
)

models_dict = {
    "linear_regression": LinearRegression(),
    "decision_tree": DecisionTreeRegressor(random_state=42),
}

# Random Forest Model with Random Search
rf_model = RandomForestRegressor(random_state=42)
param_distribs = {
    "n_estimators": randint(low=1, high=200),
    "max_features": randint(low=1, high=8),
}
rf_with_random_search = rnd_search = RandomizedSearchCV(
    rf_model,
    param_distributions=param_distribs,
    n_iter=10,
    cv=5,
    scoring="neg_mean_squared_error",
    random_state=42,
    n_jobs=-1,
)
models_dict["rf_with_random_search"] = rf_with_random_search

# Random Forest Model with Grid Search
param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
]
rf_model2 = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training
rf_with_grid_search = GridSearchCV(
    rf_model2,
    param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
    return_train_score=True,
    n_jobs=-1,
)
models_dict["rf_with_grid_search"] = rf_with_random_search


@timer
def train_models(train_data_file: Optional[str] = None):
    """
    Train models and save them to models folder.

    Read processed data from the processed data folder set in configuration.
    Trains the following models:
    1. Linear Regressionm
    2. Decision Tree Regressor
    3. Random Forest Regressor with random search hyperparameter tuning
    4. Random Forest Regressor with grid search hyperparameter tuning

    Trained model objects are saved to a joblib file each in
    the `models_path` folder set in configuration.
    Feature importance data for each model is saved to a .csv file in
    the `artifacts` folder set in configuration.

    Parameters
    ----------
    train_data_file : str, optional
         File name of training data file.
         Defaults to 'train_set.csv'
         Reads the file from the processed data folder set in configuration.
    """
    if train_data_file is None:
        train_data_file = "train_set.csv"
    train_data_path = config.processed_data_path / train_data_file
    logger.info(f"Read processed data from {train_data_path}")
    train_df = pd.read_csv(
        train_data_path,
        index_col=0,
    )
    logger.info("Read training data")
    X_train = train_df.drop(columns=["median_house_value"])
    y_train = train_df["median_house_value"].copy()
    for model_name, model in models_dict.items():
        logger.info(f"Training {model_name}..")
        model_pipeline = Pipeline(
            steps=[
                ("pre_proc", pre_proc_pipeline),
                ("model", model),
            ]
        )
        set_seed(100)
        model_pipeline.fit(X_train, y_train)
        model_path = Path(config.models_path / f"{model_name}.joblib")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        feature_importance_path = Path(
            config.artifacts_path / f"{model_name}_feature_importance.csv"
        )
        feature_importance_path.parent.mkdir(parents=True, exist_ok=True)
        if model_name == "linear_regression":
            feature_importance_df = pd.DataFrame(
                {
                    "feature": pre_proc_pipeline.get_feature_names_out(),
                    "importance": model_pipeline.named_steps["model"].coef_,
                }
            )
        elif model_name == "decision_tree":
            feature_importance_df = pd.DataFrame(
                {
                    "feature": pre_proc_pipeline.get_feature_names_out(),
                    "importance": model_pipeline.named_steps[
                        "model"
                    ].feature_importances_,
                }
            )
        elif model_name in ["rf_with_random_search", "rf_with_grid_search"]:
            best_params = model_pipeline.named_steps["model"].best_params_
            logger.info(
                f"Best parameters for {model_name}:\n{best_params}\n\n"
            )
            best_model = model_pipeline.named_steps["model"].best_estimator_
            feature_importance_df = pd.DataFrame(
                {
                    "feature": pre_proc_pipeline.get_feature_names_out(),
                    "importance": best_model.feature_importances_,
                }
            )

        joblib.dump(model_pipeline, open(model_path, "wb"))
        logger.info(
            f"Model {model_name} saved to:\n{model_path.as_posix()}\n\n"
        )
        feature_importance_df.sort_values(
            by="importance", ascending=False, inplace=True
        )
        feature_importance_df.to_csv(feature_importance_path, index=False)
        logger.info(
            "Feature importance for {} saved to:\n{}\n\n".format(
                model_name, feature_importance_path.as_posix()
            )
        )


if __name__ == "__main__":
    # from tamlep_package.utils import attach_debugger
    # attach_debugger()
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-data-path",
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
    if args.train_data_path:
        train_data_file = Path(args.train_data_path).name
        train_data_dir = Path(args.train_data_path).parent
    else:
        train_data_file = None
        train_data_dir = None
    if (
        args.models_dir_path
        or args.artifacts_dir_path
        or args.log_file_path
        or args.train_data_path
    ):
        config.set_up_paths(
            models_dir=args.models_dir_path,
            artifacts_dir=args.artifacts_dir_path,
            log_dir=log_dir,
            processed_data_dir=train_data_dir,
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
    # train models
    train_models(train_data_file)
    logger.info("Training complete!")
