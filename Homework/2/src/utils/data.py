import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
import numpy as np
import os
from scipy.io import wavfile as scipy_wavfile  # For saving audio

# Add the parent directory to path to import config
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


def load_librispeech_filepaths(librispeech_dir, max_files=None):
    """Loads all .flac file paths from a LibriSpeech directory and its subdirectories."""
    filepaths = []
    print(
        f"[Debug] Attempting to load audio files from: {librispeech_dir}"
    )  # DEBUG PRINT
    if not os.path.isdir(librispeech_dir):
        print(
            f"[Debug] Error: Path is not a directory or does not exist: {librispeech_dir}"
        )
        return filepaths

    for root, dirnames, filenames in os.walk(librispeech_dir):
        # print(f"[Debug] Walking: root={root}, dirs={dirnames}, files={filenames}") # Optional: very verbose
        for file in filenames:
            if file.endswith(".flac") or file.endswith(".wav"):
                filepaths.append(os.path.join(root, file))

    print(
        f"[Debug] Found {len(filepaths)} raw audio files in {librispeech_dir} before applying max_files limit."
    )  # DEBUG PRINT

    if max_files is not None:
        if max_files == 0:
            filepaths = []
        elif len(filepaths) > max_files:
            filepaths = sorted(filepaths)[
                :max_files
            ]  # Sort for deterministic selection
    return filepaths


def load_audio_segment(
    file_path, target_sr=config.SAMPLE_RATE, duration_samples=config.DURATION_SAMPLES
):
    """Loads, resamples, and pads/truncates a single audio file to a specific duration."""
    try:
        waveform, sr = torchaudio.load(file_path)
    except Exception as e:
        print(f"Error loading audio file {file_path}: {e}")
        return None

    # Convert to mono by averaging channels if multi-channel
    if waveform.ndim > 1 and waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample if necessary
    if sr != target_sr:
        resampler = T.Resample(sr, target_sr, dtype=waveform.dtype)
        waveform = resampler(waveform)

    # Pad or truncate to ensure fixed length
    current_samples = waveform.shape[-1]
    if current_samples < duration_samples:
        padding = duration_samples - current_samples
        waveform = F.pad(waveform, (0, padding))  # Pad at the end
    elif current_samples > duration_samples:
        waveform = waveform[..., :duration_samples]  # Truncate from the end

    return waveform.squeeze().numpy()  # Return as numpy array (1D)


def create_mixture_from_sources(
    s1_np,
    s2_np,
    noise_profile=None,
    target_snr_db=None,
    training_noise_level=config.NOISE_LEVEL_TRAIN,
):
    """
    Creates a mixture from two source signals and optionally adds noise.
    s1_np, s2_np: Numpy arrays for source 1 and source 2.
    noise_profile: Optional numpy array for a specific noise signal to add.
    target_snr_db: If not None, adds noise to achieve this specific SNR.
    training_noise_level: If target_snr_db is None and noise_profile is None, adds Gaussian noise with this std dev.
    Returns: final_mixture (np.array), s1_np (np.array), s2_np (np.array), added_noise (np.array)
    """
    # Ensure sources are of the same length (should be handled by load_audio_segment)
    min_len = min(len(s1_np), len(s2_np))
    s1_np = s1_np[:min_len]
    s2_np = s2_np[:min_len]

    clean_mixture = s1_np + s2_np
    mixture_for_snr_calc = clean_mixture.copy()  # Base for SNR calculation
    added_noise = np.zeros_like(clean_mixture, dtype=np.float32)

    if target_snr_db is not None:
        signal_power = np.mean(mixture_for_snr_calc**2)
        if signal_power > 1e-8:  # Avoid division by zero or issues with silence
            noise_power_target = signal_power / (10 ** (target_snr_db / 10))
            noise_std_dev = np.sqrt(noise_power_target)
            # Generate Gaussian noise if no specific noise profile is given
            if noise_profile is None:
                added_noise = np.random.normal(
                    0, noise_std_dev, mixture_for_snr_calc.shape
                ).astype(np.float32)
            else:  # Scale provided noise_profile
                if len(noise_profile) < len(mixture_for_snr_calc):
                    padding = len(mixture_for_snr_calc) - len(noise_profile)
                    noise_profile = np.pad(
                        noise_profile, (0, padding), "wrap"
                    )  # Pad by wrapping
                noise_profile = noise_profile[
                    : len(mixture_for_snr_calc)
                ]  # Ensure same length
                current_noise_power = np.mean(noise_profile**2)
                if current_noise_power > 1e-8:
                    scaling_factor = np.sqrt(noise_power_target / current_noise_power)
                    added_noise = noise_profile * scaling_factor
                # else: noise_profile is silent, added_noise remains zeros
        # else: signal is silent, added_noise remains zeros
        final_mixture = mixture_for_snr_calc + added_noise
    elif (
        noise_profile is not None
    ):  # Add provided noise directly (scaled by some factor if needed, or as is)
        if len(noise_profile) < len(mixture_for_snr_calc):
            padding = len(mixture_for_snr_calc) - len(noise_profile)
            noise_profile = np.pad(noise_profile, (0, padding), "wrap")
        added_noise = noise_profile[: len(mixture_for_snr_calc)].astype(np.float32)
        # Potentially scale `added_noise` here based on a fixed factor or another SNR logic if desired
        final_mixture = mixture_for_snr_calc + added_noise
    elif (
        training_noise_level > 0
    ):  # Fallback to Gaussian noise for training if no other spec
        added_noise = np.random.normal(
            0, training_noise_level, mixture_for_snr_calc.shape
        ).astype(np.float32)
        final_mixture = mixture_for_snr_calc + added_noise
    else:  # Clean mixture, no noise added
        final_mixture = mixture_for_snr_calc

    return (
        final_mixture.astype(np.float32),
        s1_np.astype(np.float32),
        s2_np.astype(np.float32),
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
        print(f"Error saving audio to {file_path}: {e}")


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

    # print(f"Saved audio samples for sample_id {sample_id} ({condition_name}) to {sample_output_dir}")
