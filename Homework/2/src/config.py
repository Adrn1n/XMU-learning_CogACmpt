import os

# --- Debugging prints at the start of config.py ---
print(f"[Config Debug] Current working directory: {os.getcwd()}")
print(f"[Config Debug] __file__ = {__file__}")
_config_file_dir = os.path.dirname(__file__)
print(f"[Config Debug] os.path.dirname(__file__) = {_config_file_dir}")
PROJECT_ROOT = os.path.abspath(os.path.join(_config_file_dir, ".."))
print(f"[Config Debug] Calculated PROJECT_ROOT = {PROJECT_ROOT}")
print(f"[Config Debug] os.path.exists(PROJECT_ROOT) = {os.path.exists(PROJECT_ROOT)}")
print(f"[Config Debug] os.path.isdir(PROJECT_ROOT) = {os.path.isdir(PROJECT_ROOT)}")

if os.path.isdir(PROJECT_ROOT):
    print(f"[Config Debug] Contents of PROJECT_ROOT ('{PROJECT_ROOT}'): {os.listdir(PROJECT_ROOT)}")
# --- End of initial debugging prints ---

# --- Global Parameters ---
MAX_TRAIN_FILES = 100  # Max files to load for training
MAX_TEST_FILES = 20    # Max files to load for testing
N_SOURCES = 2          # Number of sources to separate
SAMPLE_RATE = 16000    # Target sample rate for all audio
DURATION_SECONDS = 2   # Duration of audio segments in seconds
DURATION_SAMPLES = SAMPLE_RATE * DURATION_SECONDS # Duration in samples

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

# --- More debugging for dataset paths ---
print(f"[Config Debug] TRAIN_LIBRISPEECH_DIR = {TRAIN_LIBRISPEECH_DIR}")
print(f"[Config Debug] os.path.exists(TRAIN_LIBRISPEECH_DIR) = {os.path.exists(TRAIN_LIBRISPEECH_DIR)}")
print(f"[Config Debug] os.path.isdir(TRAIN_LIBRISPEECH_DIR) = {os.path.isdir(TRAIN_LIBRISPEECH_DIR)}")

print(f"[Config Debug] TEST_LIBRISPEECH_DIR = {TEST_LIBRISPEECH_DIR}")
print(f"[Config Debug] os.path.exists(TEST_LIBRISPEECH_DIR) = {os.path.exists(TEST_LIBRISPEECH_DIR)}")
print(f"[Config Debug] os.path.isdir(TEST_LIBRISPEECH_DIR) = {os.path.isdir(TEST_LIBRISPEECH_DIR)}")

# Check intermediate paths
assets_dir = os.path.join(PROJECT_ROOT, "assets")
print(f"[Config Debug] Checking assets_dir: {assets_dir}")
print(f"[Config Debug] os.path.exists(assets_dir) = {os.path.exists(assets_dir)}")
print(f"[Config Debug] os.path.isdir(assets_dir) = {os.path.isdir(assets_dir)}")
if os.path.isdir(assets_dir):
    print(f"[Config Debug] Contents of assets_dir: {os.listdir(assets_dir)}")
    
    assets_train_dir = os.path.join(assets_dir, "train")
    print(f"[Config Debug] Checking assets_train_dir: {assets_train_dir}")
    print(f"[Config Debug] os.path.exists(assets_train_dir) = {os.path.exists(assets_train_dir)}")
    print(f"[Config Debug] os.path.isdir(assets_train_dir) = {os.path.isdir(assets_train_dir)}")
    if os.path.isdir(assets_train_dir):
        print(f"[Config Debug] Contents of assets_train_dir: {os.listdir(assets_train_dir)}")

        assets_train_librispeech_dir = os.path.join(assets_train_dir, "LibriSpeech")
        print(f"[Config Debug] Checking assets_train_librispeech_dir: {assets_train_librispeech_dir}")
        print(f"[Config Debug] os.path.exists(assets_train_librispeech_dir) = {os.path.exists(assets_train_librispeech_dir)}")
        print(f"[Config Debug] os.path.isdir(assets_train_librispeech_dir) = {os.path.isdir(assets_train_librispeech_dir)}")
        if os.path.isdir(assets_train_librispeech_dir):
            print(f"[Config Debug] Contents of assets_train_librispeech_dir: {os.listdir(assets_train_librispeech_dir)}")

    assets_test_dir = os.path.join(assets_dir, "test")
    print(f"[Config Debug] Checking assets_test_dir: {assets_test_dir}")
    print(f"[Config Debug] os.path.exists(assets_test_dir) = {os.path.exists(assets_test_dir)}")
    print(f"[Config Debug] os.path.isdir(assets_test_dir) = {os.path.isdir(assets_test_dir)}")
    if os.path.isdir(assets_test_dir):
        print(f"[Config Debug] Contents of assets_test_dir: {os.listdir(assets_test_dir)}")
        
        assets_test_librispeech1_dir = os.path.join(assets_test_dir, "LibriSpeech_1")
        print(f"[Config Debug] Checking assets_test_librispeech1_dir: {assets_test_librispeech1_dir}")
        print(f"[Config Debug] os.path.exists(assets_test_librispeech1_dir) = {os.path.exists(assets_test_librispeech1_dir)}")
        print(f"[Config Debug] os.path.isdir(assets_test_librispeech1_dir) = {os.path.isdir(assets_test_librispeech1_dir)}")
        if os.path.isdir(assets_test_librispeech1_dir):
            print(f"[Config Debug] Contents of assets_test_librispeech1_dir: {os.listdir(assets_test_librispeech1_dir)}")
# --- End of dataset path debugging ---

NOISE_DIR = None # Optional: Path to a directory of noise files for training/evaluation

# --- Conv-TasNet Model Parameters ---
N_ENCODER_FILTERS = 256  # N: Number of filters in autoencoder (encoder output channels)
L_CONV_KERNEL_SIZE = 16  # L: Length of the filters (kernel size) in autoencoder
# Separator (TemporalConvNet - TCN) Parameters
B_TCN_CHANNELS = 256     # B: Number of channels in bottleneck and residual paths' 1x1-conv blocks
H_TCN_CHANNELS = 512     # H: Number of channels in convolutional blocks (depthwise conv)
P_TCN_KERNEL_SIZE = 3    # P: Kernel size in convolutional blocks (depthwise conv)
X_TCN_BLOCKS = 8         # X: Number of convolutional blocks in each repeat
R_TCN_REPEATS = 3        # R: Number of repeats of X blocks
Sc_TCN_CHANNELS = 256    # Sc: Number of channels in skip-connection paths' 1x1-conv blocks. If <=0, skip-connections are not used.
NORM_TYPE = 'gLN'        # Type of normalization (gLN, cLN, BN)
CAUSAL_CONV = False      # Use causal convolutions in TCN blocks

# --- Training Parameters ---
EPOCHS_TO_TRAIN = 1 # Number of epochs for training
BATCH_SIZE_TRAIN = 2 # Batch size for training
BATCH_SIZE_EVAL = 2  # Batch size for evaluation
LEARNING_RATE = 1e-3
NOISE_LEVEL_TRAIN = 0.01 # Standard deviation of Gaussian noise added during training if target_snr_db is not used

# --- Evaluation Parameters ---
SNR_CONDITIONS_DB = [None, 20, 10, 5, 0, -5]  # SNR conditions for evaluation (None for clean)
NUM_SAMPLES_TO_SAVE_EVAL = 2 # Number of audio samples to save during evaluation for each condition
EPS = 1e-8 # Epsilon for numerical stability in SI-SNR calculation

# --- Device Configuration ---
# Set to 'cuda' if GPU is available, otherwise 'cpu'
# import torch # Moved to main script to avoid circular dependency if config is imported early
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu" # Placeholder, will be set in main.py after torch import
