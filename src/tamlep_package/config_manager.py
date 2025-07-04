"""
"ConfigManager" class for managing project configuration.

This module provides a ConfigManager class for managing project configuration.
It loads default configurations from a `config.toml` file and
allows overriding.
"""

import importlib
import logging
import logging.config
import os
import sys
import tomllib
from pathlib import Path

logger = logging.getLogger(__name__)


def reload_project_modules(project_package: str = "tamlep_package"):
    """
    Reload all tamlep_package modules except the config_manager.

    Parameters
    ----------
    project_package: str, optional
        The package to reload. Defaults to tamlep_package.
    """
    modules_to_reload = [
        m
        for m in sys.modules.keys()
        if project_package in m and m != f"{project_package}.config_manager"
    ]
    for m in modules_to_reload:
        module = importlib.import_module(m)
        try:
            importlib.reload(module)
        except (ImportError, AttributeError) as e:
            logger.error(f"Error reloading module {m}: {e}")
    logger.info(f"{project_package} modules reloaded successfully.")


class ConfigManager:
    """
    A singleton class for managing project configuration.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """
        Initialize the configuration manager.
        """
        self._load_config()
        self.set_up_paths()
        self.set_up_logging()

    def _load_config(self):
        """
        Load default configurations from `config.toml` file.
        """
        self.config_file_path = os.path.join(
            Path(__file__).parent, "config.toml"
        )
        with open(self.config_file_path, "rb") as f:
            self._default_config = tomllib.load(f)

    def set_up_paths(
        self,
        models_dir: str | None = None,
        data_dir: str | None = None,
        raw_data_dir: str | None = None,
        processed_data_dir: str | None = None,
        log_dir: str | None = None,
        artifacts_dir: str | None = None,
        refresh_config: bool = False,
    ):
        """
        Set up project paths based on the configuration.

        Parameters
        ----------
        models_dir, data_dir, raw_data_dir, processed_data_dir : str, optional
            Folder names to override the corresponding default configuration.
        log_dir, artifacts_dir : str, optional
            Folder names to override the corresponding default configuration.
        """
        models_dir = models_dir or self._default_config["paths"]["models_dir"]
        log_dir = log_dir or self._default_config["paths"]["log_dir"]
        data_dir = data_dir or self._default_config["paths"]["data_dir"]
        raw_data_dir = (
            raw_data_dir or self._default_config["paths"]["raw_data_dir"]
        )
        processed_data_dir = (
            processed_data_dir
            or self._default_config["paths"]["processed_data_dir"]
        )
        artifacts_dir = (
            artifacts_dir or self._default_config["paths"]["artifacts_dir"]
        )
        self.project_root = Path(__file__).parent.parent.parent
        self.models_path = self.project_root / models_dir
        self.log_path = self.project_root / log_dir
        self.data_path = self.project_root / data_dir
        self.raw_data_path = self.data_path / raw_data_dir
        self.processed_data_path = self.data_path / processed_data_dir
        self.artifacts_path = self.project_root / artifacts_dir
        if refresh_config:
            self.refresh_configuration()

    def set_up_logging(
        self,
        log_level: str | None = None,
        backup_count: int | None = None,
        max_size_mb: int | None = None,
        log_file: str | None = None,
        error_file: str | None = None,
        add_console_logger: bool = True,
        add_file_logger: bool = False,
        add_error_file_logger: bool = False,
        refresh_config: bool = False,
    ):
        """
        Set up logging configuration.

        Parameters
        ----------
        log_level: str, optional
            Python logging module log level.
        backup_count: int, optional
            Number of backup logfiles to retain
        max_size_mb: int, optional
            Maximum size of logfile in megabytes before rotating.
        log_file: str, optional
            Name of the logfile.
        error_file: str, optional
            Name of the error logfile.
        add_console_logger: bool, optional
            Whether to add a console logger.
        add_file_logger: bool, optional
            Whether to add a file logger.
        add_error_file_logger: bool, optional
            Whether to add an error file logger.
        refresh_config: bool, optional
            Whether to refresh the configuration.
        """
        self.log_level = (
            log_level or self._default_config["logging"]["log_level"]
        )
        self.backup_count = (
            backup_count or self._default_config["logging"]["backup_count"]
        )
        self.max_size_mb = (
            max_size_mb or self._default_config["logging"]["max_size_mb"]
        )
        self.log_file = log_file or self._default_config["logging"]["log_file"]
        self.error_file = (
            error_file or self._default_config["logging"]["error_file"]
        )
        self.root_handlers = []
        handler_config = {}
        if add_console_logger:
            self.root_handlers.append("console")
            handler_config["console"] = {
                "class": "logging.StreamHandler",
                "formatter": "simple",
                "level": self.log_level,
                "stream": "ext://sys.stdout",
            }
        if add_file_logger:
            self.log_path.mkdir(exist_ok=True, parents=True)
            self.root_handlers.append("file")
            handler_config["file"] = {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "detailed",
                "level": self.log_level,
                # specify path
                "filename": (self.log_path / self.log_file).as_posix(),
                # MB to bytes
                "maxBytes": self.max_size_mb * 1024 * 1024,
                "backupCount": self.backup_count,
                "encoding": "utf-8",
            }
        if add_error_file_logger:
            self.log_path.mkdir(exist_ok=True, parents=True)
            self.root_handlers.append("error_file")
            handler_config["error_file"] = {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "detailed",
                "level": "ERROR",
                "filename": (self.log_path / self.error_file).as_posix(),
                "maxBytes": self.max_size_mb * 1024 * 1024,
                "backupCount": self.backup_count,
                "encoding": "utf-8",
            }
        self._detailed_log_format = (
            "%(asctime)s | %(name)-12s | %(levelname)-8s | %(message)s"
        )
        self._simple_log_format = "%(name)s | %(levelname)s | %(message)s"
        self._logging_config_dict = {
            "version": 1,  # always 1
            # prevents disabling existing loggers
            "disable_existing_loggers": False,
            "formatters": {
                # full format with timestamp for file logs
                "detailed": {
                    "format": self._detailed_log_format,
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
                # shorter format for console output
                "simple": {"format": self._simple_log_format},
            },
            # console handler, file handler and error handler defined
            "handlers": handler_config,
            "loggers": {
                # Root logger
                "root": {
                    "handlers": self.root_handlers,
                    "level": self.log_level,
                    "propagate": True,
                },
            },
        }
        # Apply configuration
        logging.config.dictConfig(self._logging_config_dict)
        if refresh_config:
            self.refresh_configuration()
        # Log startup message
        logger.info(f"Logging configured with level: {self.log_level}")

    def refresh_configuration(self):
        """
        Reload the project modules except the config_manager.
        """
        reload_project_modules()


config = ConfigManager()
