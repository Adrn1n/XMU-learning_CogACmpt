"""
Visualization utilities for Conv-TasNet audio source separation project.

This module contains functions for creating plots and visualizations of
training progress, evaluation results, and audio analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.io import wavfile
import librosa
import librosa.display

from logger_config import setup_visualization_logger
import config
import utils

# Initialize logger for visualization module
logger = setup_visualization_logger()

# --- Configuration for individual sample visualization ---
OUTPUT_AUDIO_DIR_BASE = os.path.join(config.PROJECT_ROOT, "assets", "audio", "eval")


def create_snr_performance_plot(metrics, save_path):
    """
    Create a plot showing SI-SNR performance vs. input SNR conditions.

    Args:
        metrics (dict): Dictionary containing evaluation metrics
        save_path (str): Path where the plot will be saved
    """
    logger.info("Creating SNR performance plot")

    # Extract SNR conditions and corresponding SI-SNR values
    snr_conditions = []
    si_snr_values = []

    for snr_str, si_snr in metrics["evaluation_final_si_snrs"].items():
        if snr_str == "clean":
            snr_conditions.append(np.inf)  # Represent clean as infinite SNR
        else:
            # Extract numeric SNR value (e.g., "20dB" -> 20)
            snr_value = float(snr_str.replace("dB", ""))
            snr_conditions.append(snr_value)
        si_snr_values.append(si_snr)

    # Sort by SNR condition for proper plotting
    sorted_data = sorted(zip(snr_conditions, si_snr_values), reverse=True)
    snr_conditions, si_snr_values = zip(*sorted_data)

    # Create labels for x-axis
    x_labels = []
    x_positions = []
    for i, snr in enumerate(snr_conditions):
        if np.isinf(snr):
            x_labels.append("Clean")
        else:
            x_labels.append(f"{int(snr)}dB")
        x_positions.append(i)

    # Create the plot
    plt.figure(figsize=(12, 8))

    # Plot SI-SNR values
    bars = plt.bar(
        x_positions, si_snr_values, color="steelblue", alpha=0.7, edgecolor="navy"
    )

    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, si_snr_values)):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Customize the plot
    plt.xlabel("Input SNR Condition", fontsize=12, fontweight="bold")
    plt.ylabel("SI-SNR (dB)", fontsize=12, fontweight="bold")
    plt.title(
        "Conv-TasNet Performance: SI-SNR vs. Input SNR Conditions\n"
        + f'(Trained for {metrics["epochs_trained"]} epoch(s))',
        fontsize=14,
        fontweight="bold",
    )
    plt.xticks(x_positions, x_labels)
    plt.grid(True, alpha=0.3, axis="y")

    # Add horizontal line at 0 dB for reference
    plt.axhline(y=0, color="red", linestyle="--", alpha=0.5, label="0 dB Reference")
    plt.legend()

    # Add annotations
    plt.figtext(
        0.02,
        0.02,
        "Note: Negative SI-SNR values indicate the separated signals are worse than the mixture.\n"
        + "This is expected for a model trained for only 1 epoch.",
        fontsize=9,
        style="italic",
        wrap=True,
    )

    plt.tight_layout()
    plt.savefig(save_path, format="svg", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"SNR performance plot saved to: {save_path}")


def create_training_progress_plot(metrics, save_path):
    """
    Create a plot showing training progress over epochs.

    Args:
        metrics (dict): Dictionary containing training metrics
        save_path (str): Path where the plot will be saved
    """

    epochs = metrics["epoch_numbers"]
    losses = metrics["training_avg_losses"]
    si_snrs = metrics["training_avg_si_snrs"]

    plt.figure(figsize=(12, 5))

    # Plot 1: Training Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, "b-o", linewidth=2, markersize=6)
    plt.xlabel("Epoch", fontweight="bold")
    plt.ylabel("Average Training Loss", fontweight="bold")
    plt.title("Training Loss Progress", fontweight="bold")
    plt.grid(True, alpha=0.3)

    # Plot 2: Training SI-SNR
    plt.subplot(1, 2, 2)
    plt.plot(epochs, si_snrs, "r-o", linewidth=2, markersize=6)
    plt.xlabel("Epoch", fontweight="bold")
    plt.ylabel("Average Training SI-SNR (dB)", fontweight="bold")
    plt.title("Training SI-SNR Progress", fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, format="svg", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Training progress plot saved to: {save_path}")


# --- Functions for Individual Sample Visualization ---


def plot_individual_waveforms(sample_dir_name, condition, plots_dir):
    """
    Plot and save waveforms for a specific sample and condition.

    Args:
        sample_dir_name (str): Name of the sample directory
        condition (str): SNR condition (e.g., 'clean', '20dB')
        plots_dir (str): Directory to save plots
    """
    sample_path = os.path.join(OUTPUT_AUDIO_DIR_BASE, sample_dir_name)
    if not os.path.isdir(sample_path):
        logger.warning(f"Sample directory not found: {sample_path}")
        return

    files_to_visualize = {
        "Original Source 1": os.path.join(sample_path, f"{condition}_s1_original.wav"),
        "Original Source 2": os.path.join(sample_path, f"{condition}_s2_original.wav"),
        "Mixture": os.path.join(sample_path, f"{condition}_mixture.wav"),
        "Estimated Source 1": os.path.join(
            sample_path, f"{condition}_s1_estimated.wav"
        ),
        "Estimated Source 2": os.path.join(
            sample_path, f"{condition}_s2_estimated.wav"
        ),
    }
    if condition != "clean":
        files_to_visualize["Added Noise"] = os.path.join(
            sample_path, f"{condition}_added_noise.wav"
        )

    num_files = sum(
        1 for f_path in files_to_visualize.values() if os.path.exists(f_path)
    )
    if num_files == 0:
        logger.warning(
            f"No audio files found for {sample_dir_name}, condition {condition}. Skipping waveform plots."
        )
        return

    plt.figure(figsize=(15, 2 * num_files if num_files > 0 else 2))
    plot_idx = 1

    for label, file_path in files_to_visualize.items():
        if os.path.exists(file_path):
            try:
                sample_rate_wav, data_wav = wavfile.read(file_path)
                data_wav = utils.normalize_audio_to_float(
                    data_wav
                )  # Ensure float for consistency
                time = np.arange(len(data_wav)) / sample_rate_wav

                plt.subplot(num_files, 1, plot_idx)
                plt.plot(time, data_wav)
                plt.title(f"Waveform: {label}", fontsize=10)
                plt.xlabel("Time (s)", fontsize=8)
                plt.ylabel("Amplitude", fontsize=8)
                plt.ylim([-1.1, 1.1])
                plt.grid(True, linestyle="--", alpha=0.7)
                plot_idx += 1
            except Exception as e:
                logger.error(f"Error plotting waveform for {file_path}: {e}")
        else:
            logger.debug(f"Skipping waveform (not found): {file_path}")

    if plot_idx > 1:  # Only save if at least one plot was made
        plt.suptitle(
            f"Waveforms for {sample_dir_name} - {condition}",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout(
            rect=(0, 0, 1, 0.96)
        )  # Adjust layout to make space for suptitle
        save_name = f"waveforms_{sample_dir_name}_{condition}.svg"
        plt.savefig(os.path.join(plots_dir, save_name), format="svg", dpi=300)
        plt.close()
        logger.info(
            f"Individual waveform plot saved to: {os.path.join(plots_dir, save_name)}"
        )
    else:
        plt.close()  # Close the figure if no plots were made


def plot_individual_spectrograms(
    sample_dir_name,
    condition,
    plots_dir,
    sr=config.SAMPLE_RATE,
    n_fft=(
        config.VIS_SPECTROGRAM_N_FFT
        if hasattr(config, "VIS_SPECTROGRAM_N_FFT")
        else 1024
    ),  # Use config or default
    hop_length=(
        config.VIS_SPECTROGRAM_HOP_LENGTH
        if hasattr(config, "VIS_SPECTROGRAM_HOP_LENGTH")
        else 256
    ),  # Use config or default
    n_mels=(
        config.VIS_SPECTROGRAM_N_MELS
        if hasattr(config, "VIS_SPECTROGRAM_N_MELS")
        else 80
    ),  # Use config or default
):
    """
    Plot and save spectrograms for a specific sample and condition.

    Args:
        sample_dir_name (str): Name of the sample directory
        condition (str): SNR condition
        plots_dir (str): Directory to save plots
        sr (int): Sample rate
        n_fft (int): Number of FFT components
        hop_length (int): Hop length for STFT
        n_mels (int): Number of mel frequency bins
    """
    sample_path = os.path.join(OUTPUT_AUDIO_DIR_BASE, sample_dir_name)
    if not os.path.isdir(sample_path):
        logger.warning(f"Sample directory not found: {sample_path}")
        return

    files_to_visualize = {
        "Original Source 1": os.path.join(sample_path, f"{condition}_s1_original.wav"),
        "Original Source 2": os.path.join(sample_path, f"{condition}_s2_original.wav"),
        "Mixture": os.path.join(sample_path, f"{condition}_mixture.wav"),
        "Estimated Source 1": os.path.join(
            sample_path, f"{condition}_s1_estimated.wav"
        ),
        "Estimated Source 2": os.path.join(
            sample_path, f"{condition}_s2_estimated.wav"
        ),
    }
    if condition != "clean":
        files_to_visualize["Added Noise"] = os.path.join(
            sample_path, f"{condition}_added_noise.wav"
        )

    num_files = sum(
        1 for f_path in files_to_visualize.values() if os.path.exists(f_path)
    )
    if num_files == 0:
        logger.warning(
            f"No audio files found for {sample_dir_name}, condition {condition}. Skipping spectrogram plots."
        )
        return

    plt.figure(figsize=(15, 2.5 * num_files if num_files > 0 else 2.5))
    plot_idx = 1

    for label, file_path in files_to_visualize.items():
        if os.path.exists(file_path):
            try:
                data_lib, sr_lib = librosa.load(file_path, sr=sr)
                s = librosa.feature.melspectrogram(
                    y=data_lib,
                    sr=sr_lib,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    n_mels=n_mels,
                )
                s_db = librosa.power_to_db(s, ref=np.max)

                plt.subplot(num_files, 1, plot_idx)
                librosa.display.specshow(
                    s_db, sr=sr_lib, hop_length=hop_length, x_axis="time", y_axis="mel"
                )
                plt.colorbar(format="%+2.0f dB")
                plt.title(f"Spectrogram: {label}", fontsize=10)
                plot_idx += 1
            except Exception as e:
                logger.error(f"Error plotting spectrogram for {file_path}: {e}")
        else:
            logger.debug(f"Skipping spectrogram (not found): {file_path}")

    if plot_idx > 1:  # Only save if at least one plot was made
        plt.suptitle(
            f"Spectrograms for {sample_dir_name} - {condition}",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout(rect=(0, 0, 1, 0.96))  # Adjust layout
        save_name = f"spectrograms_{sample_dir_name}_{condition}.svg"
        plt.savefig(os.path.join(plots_dir, save_name), format="svg", dpi=300)
        plt.close()
        logger.info(
            f"Individual spectrogram plot saved to: {os.path.join(plots_dir, save_name)}"
        )
    else:
        plt.close()


def visualize_specific_sample(sample_id, condition, plots_dir, sr=config.SAMPLE_RATE):
    """
    Generate and save waveform and spectrogram plots for a specific sample.

    Args:
        sample_id (str): Sample identifier
        condition (str): SNR condition
        plots_dir (str): Directory to save plots
        sr (int): Sample rate
    """
    logger.info(f"Visualizing individual sample: {sample_id}, Condition: {condition}")
    plot_individual_waveforms(sample_id, condition, plots_dir)
    plot_individual_spectrograms(sample_id, condition, plots_dir, sr=sr)
    logger.info(f"Finished visualizing individual sample: {sample_id}")


def main():
    """Main visualization function."""
    logger.info("Starting visualization process")

    # Paths relative to the project root
    metrics_file = os.path.join(config.PROJECT_ROOT, "model", "training_metrics.json")
    plots_dir = os.path.join(config.PROJECT_ROOT, "plots")

    # Create plots directory
    os.makedirs(plots_dir, exist_ok=True)

    # Load metrics
    logger.info("Loading training metrics...")
    metrics = utils.load_metrics(metrics_file)

    if metrics is None:
        logger.error("Failed to load metrics. Exiting.")
        return

    logger.info(f"Successfully loaded metrics for {metrics['epochs_trained']} epoch(s)")

    # Create visualizations
    logger.info("Creating visualizations...")

    # 1. SNR Performance Plot
    snr_plot_path = os.path.join(plots_dir, "snr_performance.svg")
    create_snr_performance_plot(metrics, snr_plot_path)

    # 2. Training Progress Plot
    training_plot_path = os.path.join(plots_dir, "training_progress.svg")
    create_training_progress_plot(metrics, training_plot_path)

    logger.info(f"All visualizations saved to: {plots_dir}")

    # --- Add visualization for a specific sample ---
    # Example: Visualize 'sample1' under 'clean' and 'snr0' conditions

    sample1_clean_path = os.path.join(
        OUTPUT_AUDIO_DIR_BASE, "sample1", "clean_mixture.wav"
    )
    sample1_snr0_path = os.path.join(
        OUTPUT_AUDIO_DIR_BASE, "sample1", "snr0_mixture.wav"
    )

    if os.path.exists(sample1_clean_path):
        visualize_specific_sample("sample1", "clean", plots_dir, sr=config.SAMPLE_RATE)
    else:
        logger.warning(
            f"Skipping visualization for sample1 (clean) - mixture file not found at {sample1_clean_path}"
        )

    if os.path.exists(sample1_snr0_path):
        visualize_specific_sample("sample1", "snr0", plots_dir, sr=config.SAMPLE_RATE)
    else:
        logger.warning(
            f"Skipping visualization for sample1 (snr0) - mixture file not found at {sample1_snr0_path}"
        )

    # You can add more calls to visualize_specific_sample for other samples/conditions


if __name__ == "__main__":
    main()
