"""
Configuration module for Conv-TasNet audio source separation project.

This module contains all configuration parameters for the Conv-TasNet model,
training settings, dataset paths, and other project-wide settings.
"""

import os

from logger_config import setup_config_logger

# Initialize logger for configuration module
logger = setup_config_logger()

# --- Project Root Discovery ---
logger.debug(f"Current working directory: {os.getcwd()}")
logger.debug(f"Config file location: {__file__}")

_config_file_dir = os.path.dirname(__file__)
logger.debug(f"Config file directory: {_config_file_dir}")

PROJECT_ROOT = os.path.abspath(os.path.join(_config_file_dir, ".."))
logger.info(f"Project root directory: {PROJECT_ROOT}")

# Validate project root
if not os.path.exists(PROJECT_ROOT):
    logger.error(f"Project root does not exist: {PROJECT_ROOT}")
elif not os.path.isdir(PROJECT_ROOT):
    logger.error(f"Project root is not a directory: {PROJECT_ROOT}")
else:
    logger.debug(f"Project root validated successfully")
    logger.debug(f"Project root contents: {os.listdir(PROJECT_ROOT)}")

# --- Global Parameters ---
MAX_TRAIN_FILES = None  # Use all available training files
MAX_TEST_FILES = 5  # Keep a small test set for quick evaluation

# Source Separation Parameters (for variable sources)
MIN_SOURCES_TRAIN = 2  # Minimum number of sources in a training mixture
MAX_SOURCES_TRAIN = 5  # Maximum number of sources in a training mixture (This sets N_SOURCES for the model)
MIN_SOURCES_EVAL = 1  # Minimum number of sources in an evaluation mixture
MAX_SOURCES_EVAL = 3  # Maximum number of sources in an evaluation mixture

# N_SOURCES is the maximum number of sources the model is designed to output
# This must be >= MAX_SOURCES_TRAIN and >= MAX_SOURCES_EVAL
N_SOURCES = MAX_SOURCES_TRAIN

SAMPLE_RATE = 16000  # Standard sample rate
# **Reduced duration to save memory**
DURATION_SECONDS = 3  # Duration of audio segments in seconds (e.g., 2 or 3 seconds)
DURATION_SAMPLES = SAMPLE_RATE * DURATION_SECONDS  # Duration in samples

# --- Paths ---
# Assuming the script is run from the 'src' directory or paths are relative to project root
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) # Definition moved up for debugging

MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "conv_tasnet_model.pth")
METRICS_FILE_PATH = os.path.join(MODEL_DIR, "training_metrics.json")
OUTPUT_AUDIO_DIR_BASE = os.path.join(PROJECT_ROOT, "assets/audio/eval")

# Example LibriSpeech paths (adjust as per your actual dataset location)
# These might need to be absolute paths or correctly relative to your execution context

# Updated to reflect actual directory structure under assets/audio/
TRAIN_LIBRISPEECH_DIR_SEGMENT = "assets/audio/train/LibriSpeech/train-clean-100"
TEST_LIBRISPEECH_DIR_SEGMENT = "assets/audio/test/LibriSpeech_1/test-clean"

TRAIN_LIBRISPEECH_DIR = os.path.join(PROJECT_ROOT, TRAIN_LIBRISPEECH_DIR_SEGMENT)
TEST_LIBRISPEECH_DIR = os.path.join(PROJECT_ROOT, TEST_LIBRISPEECH_DIR_SEGMENT)

# --- Dataset Path Validation ---
logger.info("Validating dataset paths...")

logger.debug(f"Training dataset path: {TRAIN_LIBRISPEECH_DIR}")
if os.path.exists(TRAIN_LIBRISPEECH_DIR):
    if os.path.isdir(TRAIN_LIBRISPEECH_DIR):
        logger.info("Training dataset directory found and accessible")
    else:
        logger.error(
            f"Training dataset path exists but is not a directory: {TRAIN_LIBRISPEECH_DIR}"
        )
else:
    logger.warning(f"Training dataset directory not found: {TRAIN_LIBRISPEECH_DIR}")

logger.debug(f"Test dataset path: {TEST_LIBRISPEECH_DIR}")
if os.path.exists(TEST_LIBRISPEECH_DIR):
    if os.path.isdir(TEST_LIBRISPEECH_DIR):
        logger.info("Test dataset directory found and accessible")
    else:
        logger.error(
            f"Test dataset path exists but is not a directory: {TEST_LIBRISPEECH_DIR}"
        )
else:
    logger.warning(f"Test dataset directory not found: {TEST_LIBRISPEECH_DIR}")

# Validate intermediate directory structure
assets_dir = os.path.join(PROJECT_ROOT, "assets")
logger.debug(f"Checking assets directory: {assets_dir}")

if os.path.isdir(assets_dir):
    logger.debug(f"Assets directory contents: {os.listdir(assets_dir)}")

    # Check training audio directory structure
    assets_train_dir = os.path.join(assets_dir, "audio", "train")
    if os.path.isdir(assets_train_dir):
        logger.debug(f"Training audio directory found: {assets_train_dir}")

        assets_train_librispeech_dir = os.path.join(assets_train_dir, "LibriSpeech")
        if os.path.isdir(assets_train_librispeech_dir):
            logger.debug(
                f"LibriSpeech training directory found with contents: {os.listdir(assets_train_librispeech_dir)}"
            )
        else:
            logger.warning(
                f"LibriSpeech training directory not found: {assets_train_librispeech_dir}"
            )
    else:
        logger.warning(f"Training audio directory not found: {assets_train_dir}")

    # Check test audio directory structure
    assets_test_dir = os.path.join(assets_dir, "audio", "test")
    if os.path.isdir(assets_test_dir):
        logger.debug(f"Test audio directory found: {assets_test_dir}")

        assets_test_librispeech1_dir = os.path.join(assets_test_dir, "LibriSpeech_1")
        if os.path.isdir(assets_test_librispeech1_dir):
            logger.debug(
                f"LibriSpeech_1 test directory found with contents: {os.listdir(assets_test_librispeech1_dir)}"
            )
        else:
            logger.warning(
                f"LibriSpeech_1 test directory not found: {assets_test_librispeech1_dir}"
            )
    else:
        logger.warning(f"Test audio directory not found: {assets_test_dir}")
else:
    logger.error(f"Assets directory not found: {assets_dir}")

logger.info("Dataset path validation completed")

NOISE_DIR = None  # Optional: Path to a directory of noise files for training/evaluation

# --- Conv-TasNet Model Parameters ---
# Adjusted model size slightly down from the "working" 2-source config
# while keeping it reasonable for up to 10 sources.
N_ENCODER_FILTERS = 256  # n: Number of filters in autoencoder (encoder output channels)
L_CONV_KERNEL_SIZE = 16  # l: Length of the filters (kernel size) in autoencoder
# Separator (TemporalConvNet - TCN) Parameters
B_TCN_CHANNELS = (
    128  # b: Number of channels in bottleneck and residual paths' 1x1-conv blocks
)
H_TCN_CHANNELS = 256  # h: Number of channels in convolutional blocks (depthwise conv)
P_TCN_KERNEL_SIZE = 4  # p: Kernel size in convolutional blocks (depthwise conv)
X_TCN_BLOCKS = 8  # x: Number of convolutional blocks in each repeat (slightly reduced)
R_TCN_REPEATS = 2  # r: Number of repeats of x blocks (slightly reduced)
Sc_TCN_CHANNELS = 128  # sc: Number of channels in skip-connection paths' 1x1-conv blocks. If <=0, skip-connections are not used. (reduced)
NORM_TYPE = "gLN"  # Type of normalization (gLN, cLN, BN)
CAUSAL_CONV = True  # Use causal convolutions in TCN blocks (False for potentially better performance)

# --- Training Parameters ---
EPOCHS_TO_TRAIN = 5  # Keep for a short test run
# **Crucially reduced batch size to fit in memory**
BATCH_SIZE_TRAIN = 4  # Start with 1. If it works, try increasing to 2.
BATCH_SIZE_EVAL = 2  # Keep eval batch size, duration is now 3s

LEARNING_RATE = (
    1e-6  # Slightly reduced learning rate, often better with smaller batches
)
NOISE_LEVEL_TRAIN = 0.01  # Standard deviation of Gaussian noise added during training if target_snr_db is not used

# --- Evaluation Parameters ---
SNR_CONDITIONS_DB = [
    None,
    20,
    10,
    5,
    0,
    -5,
]  # SNR conditions for evaluation (None for clean)
NUM_SAMPLES_TO_SAVE_EVAL = (
    3  # Number of audio samples to save during evaluation for each condition
)
EPS = 1e-8  # Epsilon for numerical stability in SI-SNR calculation

# --- Device Configuration ---
# Set to 'cuda' if GPU is available, otherwise 'cpu'
# import torch # Moved to main script to avoid circular dependency if config is imported early
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cuda"  # Set to cuda if you want to use GPU, ensure torch is imported in main before this is used
