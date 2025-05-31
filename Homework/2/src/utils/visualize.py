#!/usr/bin/env python3
"""
Visualization script for Conv-TasNet multi-speaker speech separation results.
Analyzes training metrics and SNR-based evaluation performance.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import librosa  # Added for spectrograms and audio loading
import librosa.display  # Added for spectrogram display
from scipy.io import wavfile  # Added for wav file reading

# Determine the project root directory dynamically
# Assuming this script is in src/utils/visualize.py, project_root is two levels up.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# --- Configuration for individual sample visualization ---
OUTPUT_AUDIO_DIR_BASE = os.path.join(
    PROJECT_ROOT, "assets", "audio", "eval"
)  # Use PROJECT_ROOT
DEFAULT_SR = 16000  # Default sample rate


def load_metrics(metrics_file_path):
    """Load training metrics from JSON file."""
    try:
        with open(metrics_file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Metrics file not found: {metrics_file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error reading JSON from: {metrics_file_path}")
        return None


def create_snr_performance_plot(metrics, save_path):
    """Create a plot showing SI-SNR performance vs. input SNR conditions."""

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
    print(f"SNR performance plot saved to: {save_path}")


def create_training_progress_plot(metrics, save_path):
    """Create a plot showing training progress over epochs."""

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
    print(f"Training progress plot saved to: {save_path}")


def create_summary_report(metrics, save_path):
    """Create a text summary report of the results."""

    report = []
    report.append("=" * 80)
    report.append("Conv-TasNet Multi-Speaker Speech Separation - Training Summary")
    report.append("=" * 80)
    report.append("")

    # Training Summary
    report.append("TRAINING SUMMARY:")
    report.append(f"  • Epochs Trained: {metrics['epochs_trained']}")
    report.append(f"  • Final Training Loss: {metrics['training_avg_losses'][-1]:.4f}")
    report.append(
        f"  • Final Training SI-SNR: {metrics['training_avg_si_snrs'][-1]:.2f} dB"
    )
    report.append("")

    # Evaluation Summary
    report.append("EVALUATION SUMMARY (SI-SNR Performance):")
    eval_results = metrics["evaluation_final_si_snrs"]

    # Sort by performance (best to worst)
    sorted_results = sorted(eval_results.items(), key=lambda x: x[1], reverse=True)

    for condition, si_snr in sorted_results:
        report.append(f"  • {condition:>6}: {si_snr:>6.2f} dB")

    report.append("")

    # Analysis
    report.append("ANALYSIS:")
    best_condition = max(eval_results.items(), key=lambda x: x[1])
    worst_condition = min(eval_results.items(), key=lambda x: x[1])

    report.append(
        f"  • Best Performance: {best_condition[0]} ({best_condition[1]:.2f} dB)"
    )
    report.append(
        f"  • Worst Performance: {worst_condition[0]} ({worst_condition[1]:.2f} dB)"
    )

    # Performance degradation analysis
    clean_performance = eval_results.get("clean", 0)
    degradations = []
    for condition, si_snr in eval_results.items():
        if condition != "clean":
            degradation = clean_performance - si_snr
            degradations.append((condition, degradation))

    if degradations:
        report.append("")
        report.append("  • Performance Degradation from Clean Condition:")
        for condition, deg in sorted(degradations, key=lambda x: x[1]):
            report.append(f"    - {condition}: {deg:.2f} dB worse")

    report.append("")
    report.append("OBSERVATIONS:")
    report.append(
        "  • All SI-SNR values are negative, indicating the separated signals"
    )
    report.append("    are currently worse than the input mixture.")
    report.append("  • This is expected for a model trained for only 1 epoch.")
    report.append("  • Performance generally degrades with lower input SNR conditions.")
    report.append("  • The model shows some robustness across different noise levels.")
    report.append("")
    report.append("RECOMMENDATIONS:")
    report.append("  • Train for more epochs (typically 50-100+ epochs)")
    report.append("  • Monitor convergence using validation data")
    report.append("  • Consider learning rate scheduling")
    report.append("  • Evaluate with more test samples for statistical significance")
    report.append("")
    report.append("=" * 80)

    # Save the report
    with open(save_path, "w") as f:
        f.write("\n".join(report))

    print(f"Summary report saved to: {save_path}")

    # Also print to console
    print("\n" + "\n".join(report))


# --- Functions for Individual Sample Visualization ---


def to_float(data):
    """Converts audio data to float32, normalizing if it's an integer type."""
    if data.dtype != np.float32 and data.dtype != np.float64:
        if np.issubdtype(data.dtype, np.integer):
            max_val = np.iinfo(data.dtype).max
            data = data.astype(np.float32) / max_val
        else:
            data = data.astype(np.float32)
    return data


def plot_individual_waveforms(sample_dir_name, condition, plots_dir, sr=DEFAULT_SR):
    """Plots and saves waveforms for a specific sample and condition."""
    sample_path = os.path.join(OUTPUT_AUDIO_DIR_BASE, sample_dir_name)
    if not os.path.isdir(sample_path):
        print(f"Sample directory not found: {sample_path}")
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
        print(
            f"No audio files found for {sample_dir_name}, condition {condition}. Skipping waveform plots."
        )
        return

    plt.figure(figsize=(15, 2 * num_files if num_files > 0 else 2))
    plot_idx = 1

    for label, file_path in files_to_visualize.items():
        if os.path.exists(file_path):
            try:
                sample_rate_wav, data_wav = wavfile.read(file_path)
                data_wav = to_float(data_wav)  # Ensure float for consistency
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
                print(f"Error plotting waveform for {file_path}: {e}")
        else:
            print(f"Skipping waveform (not found): {file_path}")

    if plot_idx > 1:  # Only save if at least one plot was made
        plt.suptitle(
            f"Waveforms for {sample_dir_name} - {condition}",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout(
            rect=[0, 0, 1, 0.96]
        )  # Adjust layout to make space for suptitle
        save_name = f"waveforms_{sample_dir_name}_{condition}.svg"
        plt.savefig(os.path.join(plots_dir, save_name), format="svg", dpi=300)
        plt.close()
        print(
            f"Individual waveform plot saved to: {os.path.join(plots_dir, save_name)}"
        )
    else:
        plt.close()  # Close the figure if no plots were made


def plot_individual_spectrograms(
    sample_dir_name,
    condition,
    plots_dir,
    sr=DEFAULT_SR,
    n_fft=1024,
    hop_length=256,
    n_mels=80,
):
    """Plots and saves spectrograms for a specific sample and condition."""
    sample_path = os.path.join(OUTPUT_AUDIO_DIR_BASE, sample_dir_name)
    if not os.path.isdir(sample_path):
        print(f"Sample directory not found: {sample_path}")
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
        print(
            f"No audio files found for {sample_dir_name}, condition {condition}. Skipping spectrogram plots."
        )
        return

    plt.figure(figsize=(15, 2.5 * num_files if num_files > 0 else 2.5))
    plot_idx = 1

    for label, file_path in files_to_visualize.items():
        if os.path.exists(file_path):
            try:
                data_lib, sr_lib = librosa.load(file_path, sr=sr)
                S = librosa.feature.melspectrogram(
                    y=data_lib,
                    sr=sr_lib,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    n_mels=n_mels,
                )
                S_db = librosa.power_to_db(S, ref=np.max)

                plt.subplot(num_files, 1, plot_idx)
                librosa.display.specshow(
                    S_db, sr=sr_lib, hop_length=hop_length, x_axis="time", y_axis="mel"
                )
                plt.colorbar(format="%+2.0f dB")
                plt.title(f"Spectrogram: {label}", fontsize=10)
                plot_idx += 1
            except Exception as e:
                print(f"Error plotting spectrogram for {file_path}: {e}")
        else:
            print(f"Skipping spectrogram (not found): {file_path}")

    if plot_idx > 1:  # Only save if at least one plot was made
        plt.suptitle(
            f"Spectrograms for {sample_dir_name} - {condition}",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout
        save_name = f"spectrograms_{sample_dir_name}_{condition}.svg"
        plt.savefig(os.path.join(plots_dir, save_name), format="svg", dpi=300)
        plt.close()
        print(
            f"Individual spectrogram plot saved to: {os.path.join(plots_dir, save_name)}"
        )
    else:
        plt.close()


def visualize_specific_sample(sample_id, condition, plots_dir, sr=DEFAULT_SR):
    """Generates and saves waveform and spectrogram plots for a specific sample."""
    print(
        f"\n--- Visualizing individual sample: {sample_id}, Condition: {condition} ---"
    )
    plot_individual_waveforms(sample_id, condition, plots_dir, sr=sr)
    plot_individual_spectrograms(sample_id, condition, plots_dir, sr=sr)
    print(f"--- Finished visualizing individual sample: {sample_id} ---")


def main():
    """Main visualization function."""

    # Paths relative to the project root
    metrics_file = os.path.join(
        PROJECT_ROOT, "model", "training_metrics.json"
    )  # Use PROJECT_ROOT
    plots_dir = os.path.join(PROJECT_ROOT, "plots")  # Use PROJECT_ROOT

    # Create plots directory
    os.makedirs(plots_dir, exist_ok=True)

    # Load metrics
    print("Loading training metrics...")
    metrics = load_metrics(metrics_file)

    if metrics is None:
        print("Failed to load metrics. Exiting.")
        return

    print(f"Successfully loaded metrics for {metrics['epochs_trained']} epoch(s)")

    # Create visualizations
    print("\nCreating visualizations...")

    # 1. SNR Performance Plot
    snr_plot_path = os.path.join(plots_dir, "snr_performance.svg")
    create_snr_performance_plot(metrics, snr_plot_path)

    # 2. Training Progress Plot
    training_plot_path = os.path.join(plots_dir, "training_progress.svg")
    create_training_progress_plot(metrics, training_plot_path)

    # 3. Summary Report
    report_path = os.path.join(plots_dir, "training_summary.txt")
    create_summary_report(metrics, report_path)

    print(f"\nAll visualizations saved to: {plots_dir}")

    # --- Add visualization for a specific sample ---
    # Example: Visualize 'sample1' under 'clean' and 'snr0' conditions

    sample1_clean_path = os.path.join(
        OUTPUT_AUDIO_DIR_BASE, "sample1", "clean_mixture.wav"
    )
    sample1_snr0_path = os.path.join(
        OUTPUT_AUDIO_DIR_BASE, "sample1", "snr0_mixture.wav"
    )

    if os.path.exists(sample1_clean_path):
        visualize_specific_sample("sample1", "clean", plots_dir, sr=DEFAULT_SR)
    else:
        print(
            f"\nSkipping visualization for sample1 (clean) - mixture file not found at {sample1_clean_path}"
        )

    if os.path.exists(sample1_snr0_path):
        visualize_specific_sample("sample1", "snr0", plots_dir, sr=DEFAULT_SR)
    else:
        print(
            f"\nSkipping visualization for sample1 (snr0) - mixture file not found at {sample1_snr0_path}"
        )

    # You can add more calls to visualize_specific_sample for other samples/conditions


if __name__ == "__main__":
    main()
