"""
Utility functions for Conv-TasNet audio source separation project.

This module contains functions for audio loading, processing, data generation,
and evaluation utilities.
"""

import os
import torch
import torchaudio
import torchaudio.transforms as trans
import torch.nn.functional as func
import numpy as np
from scipy.io import wavfile as scipy_wavfile
import json

from logger_config import setup_data_logger
import config

# Initialize logger for data utilities
logger = setup_data_logger()


def load_audio_filepaths(audio_dir, max_files=None):
    """
    Load all .flac and .wav file paths from a directory and its subdirectories.

    Args:
        audio_dir (str): Directory to search for audio files
        max_files (int, optional): Maximum number of files to return

    Returns:
        list: List of file paths found
    """
    filepaths = []
    logger.info(f"Loading audio files from directory: {audio_dir}")

    if not os.path.isdir(audio_dir):
        logger.error(f"Path is not a directory or does not exist: {audio_dir}")
        return filepaths

    for root, dirnames, filenames in os.walk(audio_dir):
        for file in filenames:
            if file.endswith(".flac") or file.endswith(".wav"):
                filepaths.append(os.path.join(root, file))

    logger.info(f"Found {len(filepaths)} audio files in {audio_dir}")

    if max_files is not None:
        if max_files == 0:
            filepaths = []
            logger.info("max_files set to 0, returning empty list")
        elif len(filepaths) > max_files:
            if isinstance(max_files, int) and max_files > 0:
                filepaths = sorted(filepaths)[:max_files]
                logger.info(f"Limited to {max_files} files for processing")

    return filepaths


def load_audio_segment(
    file_path, target_sr=config.SAMPLE_RATE, duration_samples=config.DURATION_SAMPLES
):
    """
    Load, resample, and pad/truncate a single audio file to a specific duration.

    Args:
        file_path (str): Path to the audio file
        target_sr (int): Target sample rate
        duration_samples (int): Target duration in samples

    Returns:
        torch.Tensor or None: Processed audio waveform, or None if loading failed
    """
    try:
        waveform, sr = torchaudio.load(file_path)
    except Exception as e:
        logger.error(f"Error loading audio file {file_path}: {e}")
        return None

    # Convert to mono by averaging channels if multi-channel
    if waveform.ndim > 1 and waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample if necessary
    if sr != target_sr:
        resampler = trans.Resample(sr, target_sr, dtype=waveform.dtype)
        waveform = resampler(waveform)

    # Pad or truncate to ensure fixed length
    current_samples = waveform.shape[-1]
    if current_samples < duration_samples:
        padding = duration_samples - current_samples
        waveform = func.pad(waveform, (0, padding))  # Pad at the end
    elif current_samples > duration_samples:
        waveform = waveform[..., :duration_samples]  # Truncate from the end

    return waveform.squeeze().numpy()  # Return as numpy array (1D)


def create_mixture_from_sources(
    sources_list_np,  # Accept a list of source numpy arrays
    noise_profile=None,
    target_snr_db=None,
    training_noise_level=config.NOISE_LEVEL_TRAIN,
):
    """
    Creates a mixture from a list of source signals and optionally adds noise.
    sources_list_np: List of numpy arrays for the source signals.
    noise_profile: Optional numpy array for a specific noise signal to add.
    target_snr_db: If not None, adds noise to achieve this specific SNR.
    training_noise_level: If target_snr_db is None and noise_profile is None, adds Gaussian noise with this std dev.
    Returns: final_mixture (np.array), original_sources_list_np (list of np.array), added_noise (np.array)
    """
    if not sources_list_np:
        return np.array([]), [], np.array([])  # Handle empty list

    # Ensure all sources are of the same length (should be handled by load_audio_segment, but double check)
    min_len = min(len(s) for s in sources_list_np)
    sources_list_np = [s[:min_len] for s in sources_list_np]

    # Create clean mixture by summing all sources
    clean_mixture = np.sum(sources_list_np, axis=0)
    mixture_for_snr_calc = clean_mixture.copy()
    added_noise = np.zeros_like(clean_mixture, dtype=np.float32)

    if target_snr_db is not None:
        # ... (SNR calculation and noise scaling based on mixture_for_snr_calc) ...
        # This part needs to be implemented based on how SNR is calculated and noise is scaled.
        # For now, let\'s assume a placeholder for noise addition.
        # Example: Calculate power of clean_mixture, then calculate required noise power for target_snr_db.
        # Generate noise with that power and add it.
        # This is a complex part and depends on the existing SNR logic.
        # For demonstration, we\'ll just add random noise if target_snr_db is set.
        signal_power = np.mean(mixture_for_snr_calc**2)
        if signal_power > 1e-6:  # Avoid division by zero or very small signal power
            noise_power_target = signal_power / (10 ** (target_snr_db / 10))
            noise_std = np.sqrt(noise_power_target)
            added_noise = np.random.normal(
                0, noise_std, mixture_for_snr_calc.shape
            ).astype(np.float32)
        else:  # If signal is silent, add no noise or very low level noise
            added_noise = np.zeros_like(mixture_for_snr_calc, dtype=np.float32)

    elif noise_profile is not None:
        # Ensure noise_profile is tiled or truncated to match mixture_for_snr_calc length
        if len(noise_profile) < len(mixture_for_snr_calc):
            num_repeats = int(np.ceil(len(mixture_for_snr_calc) / len(noise_profile)))
            added_noise = np.tile(noise_profile, num_repeats)[
                : len(mixture_for_snr_calc)
            ].astype(np.float32)
        else:
            added_noise = noise_profile[: len(mixture_for_snr_calc)].astype(np.float32)
        # Potentially scale noise_profile here if needed
        # For now, assume it\'s used as is or pre-scaled.

    elif training_noise_level > 0:
        added_noise = np.random.normal(
            0, training_noise_level, mixture_for_snr_calc.shape
        ).astype(np.float32)

    final_mixture = mixture_for_snr_calc + added_noise

    return (
        final_mixture.astype(np.float32),
        [
            s.astype(np.float32) for s in sources_list_np
        ],  # Return the list of original sources
        added_noise.astype(np.float32),
    )


def save_audio_sample(audio_data, file_path, sample_rate=config.SAMPLE_RATE):
    """Saves a single audio track to a .wav file."""
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Normalize to [-1, 1] and convert to int16 for WAV saving if not already float32
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        # Scipy wavfile expects int16 or float32. If float32, it should be in [-1, 1]
        # Let's ensure it is for broader compatibility, though direct float32 usually works.
        # max_val = np.max(np.abs(audio_data))
        # if max_val > 1.0:
        #     audio_data = audio_data / max_val

        # scipy_wavfile.write expects data in shape (samples,) or (samples, channels)
        if (
            audio_data.ndim > 1 and audio_data.shape[0] < audio_data.shape[1]
        ):  # (channels, samples)
            audio_data = audio_data.T  # Transpose to (samples, channels)
        elif audio_data.ndim == 1:  # (samples,)
            pass  # Already in correct shape

        scipy_wavfile.write(file_path, sample_rate, audio_data)
    except Exception as e:
        logger.error(f"Error saving audio to {file_path}: {e}")


def save_evaluation_audio_samples(
    mixture_np,
    sources_np_list,
    estimated_sources_np_list,
    added_noise_np,
    sample_id,
    condition_name,
    base_output_dir=config.OUTPUT_AUDIO_DIR_BASE,
    sr=config.SAMPLE_RATE,
):
    """
    Saves a set of audio files (mixture, original sources, estimated sources, noise) for an evaluation sample.
    sources_np_list: list of numpy arrays for original sources.
    estimated_sources_np_list: list of numpy arrays for estimated sources.
    """
    sample_output_dir = os.path.join(base_output_dir, f"sample{sample_id}")
    os.makedirs(sample_output_dir, exist_ok=True)

    save_audio_sample(
        mixture_np, os.path.join(sample_output_dir, f"{condition_name}_mixture.wav"), sr
    )
    if added_noise_np is not None and np.any(
        added_noise_np
    ):  # Save noise if it exists and is not all zeros
        save_audio_sample(
            added_noise_np,
            os.path.join(sample_output_dir, f"{condition_name}_added_noise.wav"),
            sr,
        )

    for i, source_np in enumerate(sources_np_list):
        save_audio_sample(
            source_np,
            os.path.join(sample_output_dir, f"{condition_name}_s{i+1}_original.wav"),
            sr,
        )

    for i, est_source_np in enumerate(estimated_sources_np_list):
        save_audio_sample(
            est_source_np,
            os.path.join(sample_output_dir, f"{condition_name}_s{i+1}_estimated.wav"),
            sr,
        )

    logger.info(
        f"Saved audio samples for sample_id {sample_id} ({condition_name}) to {sample_output_dir}"
    )


# --- Functions moved from visualize.py ---


def load_metrics(metrics_file_path):
    """
    Load training metrics from JSON file.

    Args:
        metrics_file_path (str): Path to the metrics JSON file

    Returns:
        dict or None: Loaded metrics data, or None if loading failed
    """
    try:
        with open(metrics_file_path, "r") as f:
            metrics = json.load(f)
            logger.info(f"Successfully loaded metrics from {metrics_file_path}")
            return metrics
    except FileNotFoundError:
        logger.error(f"Metrics file not found: {metrics_file_path}")
        return None
    except json.JSONDecodeError:
        logger.error(f"Error reading JSON from: {metrics_file_path}")
        return None


def normalize_audio_to_float(data):
    """Converts audio data to float32, normalizing if it's an integer type."""
    if data.dtype != np.float32 and data.dtype != np.float64:
        if np.issubdtype(data.dtype, np.integer):
            max_val = np.iinfo(data.dtype).max
            data = data.astype(np.float32) / max_val
        else:
            data = data.astype(np.float32)
    return data
