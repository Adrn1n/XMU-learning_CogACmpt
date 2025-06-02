"""
Centralized logging configuration for Conv-TasNet audio source separation project.

This module provides a standardized logging setup that writes to both console and files
in the log/ directory with proper formatting and log rotation.
"""

import logging
import logging.handlers
from typing import Optional
import os
import sys


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_dir: Optional[str] = None,
    console_output: bool = True,
    file_output: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
) -> logging.Logger:
    """
    Set up a logger with both console and file handlers.

    Args:
        name: Logger name (typically __name__ of the calling module)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory to store log files. If None, uses PROJECT_ROOT/log
        console_output: Whether to output logs to console
        file_output: Whether to output logs to files
        max_bytes: Maximum size of each log file before rotation
        backup_count: Number of backup files to keep

    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    # Determine log directory
    if log_dir is None:
        # Get project root (assuming this file is in src/ directory)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        log_dir = os.path.join(project_root, "log")

    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # File handlers
    if file_output:
        # Main log file with rotation
        main_log_file = os.path.join(log_dir, f"{name.replace('.', '_')}.log")
        file_handler = logging.handlers.RotatingFileHandler(
            main_log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)

        # Error log file (only errors and critical)
        error_log_file = os.path.join(log_dir, f"{name.replace('.', '_')}_errors.log")
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        logger.addHandler(error_handler)

    return logger


def setup_main_logger(log_dir: Optional[str] = None) -> logging.Logger:
    """
    Set up a specialized logger for main execution and general processes.

    Args:
        log_dir: Directory to store log files

    Returns:
        Configured main logger
    """
    return setup_logger(
        name="conv_tasnet.main",
        level=logging.INFO,
        log_dir=log_dir,
        console_output=True,
        file_output=True,
    )


def setup_training_logger(log_dir: Optional[str] = None) -> logging.Logger:
    """
    Set up a specialized logger for training processes.

    Args:
        log_dir: Directory to store log files

    Returns:
        Configured training logger
    """
    return setup_logger(
        name="conv_tasnet.training",
        level=logging.INFO,
        log_dir=log_dir,
        console_output=True,
        file_output=True,
    )


def setup_evaluation_logger(log_dir: Optional[str] = None) -> logging.Logger:
    """
    Set up a specialized logger for evaluation processes.

    Args:
        log_dir: Directory to store log files

    Returns:
        Configured evaluation logger
    """
    return setup_logger(
        name="conv_tasnet.evaluation",
        level=logging.INFO,
        log_dir=log_dir,
        console_output=True,
        file_output=True,
    )


def setup_config_logger(log_dir: Optional[str] = None) -> logging.Logger:
    """
    Set up a specialized logger for configuration and setup processes.

    Args:
        log_dir: Directory to store log files

    Returns:
        Configured configuration logger
    """
    return setup_logger(
        name="conv_tasnet.config",
        level=logging.DEBUG,  # More verbose for config debugging
        log_dir=log_dir,
        console_output=True,
        file_output=True,
    )


def setup_data_logger(log_dir: Optional[str] = None) -> logging.Logger:
    """
    Set up a specialized logger for data loading and processing.

    Args:
        log_dir: Directory to store log files

    Returns:
        Configured data processing logger
    """
    return setup_logger(
        name="conv_tasnet.data",
        level=logging.INFO,
        log_dir=log_dir,
        console_output=True,
        file_output=True,
    )


def setup_visualization_logger(log_dir: Optional[str] = None) -> logging.Logger:
    """
    Set up a specialized logger for visualization processes.

    Args:
        log_dir: Directory to store log files

    Returns:
        Configured visualization logger
    """
    return setup_logger(
        name="conv_tasnet.visualization",
        level=logging.INFO,
        log_dir=log_dir,
        console_output=True,
        file_output=True,
    )


def log_system_info(logger: logging.Logger) -> None:
    """
    Log basic system information for debugging purposes.

    Args:
        logger: Logger instance to use for logging
    """
    import platform
    import torch

    logger.info("=== System Information ===")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python version: {platform.python_version()}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU device: {torch.cuda.get_device_name(0)}")
    logger.info("=== End System Information ===")


def log_config_params(logger: logging.Logger, config_module) -> None:
    """
    Log important configuration parameters.

    Args:
        logger: Logger instance to use for logging
        config_module: Configuration module containing parameters
    """
    logger.info("=== Configuration Parameters ===")

    # Model parameters
    logger.info("Model Configuration:")
    logger.info(f"  N_SOURCES: {getattr(config_module, 'N_SOURCES', 'Not set')}")
    logger.info(
        f"  N_ENCODER_FILTERS: {getattr(config_module, 'N_ENCODER_FILTERS', 'Not set')}"
    )
    logger.info(
        f"  L_CONV_KERNEL_SIZE: {getattr(config_module, 'L_CONV_KERNEL_SIZE', 'Not set')}"
    )
    logger.info(
        f"  B_TCN_CHANNELS: {getattr(config_module, 'B_TCN_CHANNELS', 'Not set')}"
    )
    logger.info(
        f"  H_TCN_CHANNELS: {getattr(config_module, 'H_TCN_CHANNELS', 'Not set')}"
    )
    logger.info(f"  X_TCN_BLOCKS: {getattr(config_module, 'X_TCN_BLOCKS', 'Not set')}")
    logger.info(
        f"  R_TCN_REPEATS: {getattr(config_module, 'R_TCN_REPEATS', 'Not set')}"
    )

    # Training parameters
    logger.info("Training Configuration:")
    logger.info(
        f"  EPOCHS_TO_TRAIN: {getattr(config_module, 'EPOCHS_TO_TRAIN', 'Not set')}"
    )
    logger.info(
        f"  BATCH_SIZE_TRAIN: {getattr(config_module, 'BATCH_SIZE_TRAIN', 'Not set')}"
    )
    logger.info(
        f"  LEARNING_RATE: {getattr(config_module, 'LEARNING_RATE', 'Not set')}"
    )
    logger.info(f"  DEVICE: {getattr(config_module, 'DEVICE', 'Not set')}")

    # Data parameters
    logger.info("Data Configuration:")
    logger.info(f"  SAMPLE_RATE: {getattr(config_module, 'SAMPLE_RATE', 'Not set')}")
    logger.info(
        f"  DURATION_SECONDS: {getattr(config_module, 'DURATION_SECONDS', 'Not set')}"
    )
    logger.info(
        f"  MAX_TRAIN_FILES: {getattr(config_module, 'MAX_TRAIN_FILES', 'Not set')}"
    )
    logger.info(
        f"  MAX_TEST_FILES: {getattr(config_module, 'MAX_TEST_FILES', 'Not set')}"
    )

    logger.info("=== End Configuration Parameters ===")


# Convenience function to get a logger with standard settings
def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Get a logger with standard project settings.

    Args:
        name: Logger name (typically __name__)
        level: Logging level

    Returns:
        Configured logger instance
    """
    return setup_logger(name, level)
