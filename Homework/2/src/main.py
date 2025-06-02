"""
Main training and evaluation script for Conv-TasNet audio source separation.

This script handles the complete pipeline including data loading, model training,
evaluation, and result visualization for the Conv-TasNet architecture.
"""

import random
from tqdm import tqdm
import torch
import torch.nn.functional as func
import torch.optim as optim
import numpy as np
import os
import json
import datetime
from collections import defaultdict
import platform

from logger_config import (
    setup_main_logger,
    setup_training_logger,
    setup_evaluation_logger,
    log_system_info,
)

# Import project modules
import config
import utils
from losses import si_snr_loss, match_length
from model import ConvTasNet

# Declare logger variables globally but initialize them in main_run
logger = None
train_logger = None
eval_logger = None

# --- Global Parameters ---
# These are now primarily sourced from config.py
MAX_TRAIN_FILES = config.MAX_TRAIN_FILES
MAX_TEST_FILES = config.MAX_TEST_FILES
N_SOURCES = config.N_SOURCES
SAMPLE_RATE = config.SAMPLE_RATE
DURATION_SECONDS = config.DURATION_SECONDS
DURATION_SAMPLES = config.DURATION_SAMPLES

MODEL_DIR = config.MODEL_DIR
MODEL_PATH = config.MODEL_PATH
METRICS_FILE_PATH = config.METRICS_FILE_PATH
OUTPUT_AUDIO_DIR_BASE = config.OUTPUT_AUDIO_DIR_BASE

# Conv-TasNet Parameters (can also be moved to a model-specific config or stay here if closely tied to main script logic)
N_ENCODER_FILTERS = config.N_ENCODER_FILTERS
L_CONV_KERNEL_SIZE = config.L_CONV_KERNEL_SIZE
B_TCN_CHANNELS = config.B_TCN_CHANNELS
H_TCN_CHANNELS = config.H_TCN_CHANNELS
P_TCN_KERNEL_SIZE = config.P_TCN_KERNEL_SIZE
X_TCN_BLOCKS = config.X_TCN_BLOCKS
R_TCN_REPEATS = config.R_TCN_REPEATS
Sc_TCN_CHANNELS = config.Sc_TCN_CHANNELS
NORM_TYPE = config.NORM_TYPE
CAUSAL_CONV = config.CAUSAL_CONV

# Training parameters from config
EPOCHS_TO_TRAIN = config.EPOCHS_TO_TRAIN
BATCH_SIZE_TRAIN = config.BATCH_SIZE_TRAIN
LEARNING_RATE = config.LEARNING_RATE
NOISE_LEVEL_TRAIN = config.NOISE_LEVEL_TRAIN

# Evaluation parameters from config
SNR_CONDITIONS_DB = config.SNR_CONDITIONS_DB
NUM_SAMPLES_TO_SAVE_EVAL = config.NUM_SAMPLES_TO_SAVE_EVAL
BATCH_SIZE_EVAL = config.BATCH_SIZE_EVAL


# --- Training ---
def train(
    model,
    audio_files_with_speakers_list,
    optimizer,
    device,
    epochs=config.EPOCHS_TO_TRAIN,
    batch_size=config.BATCH_SIZE_TRAIN,
    min_sources=config.MIN_SOURCES_TRAIN,
    max_sources=config.MAX_SOURCES_TRAIN,
    noise_level=config.NOISE_LEVEL_TRAIN,
    sample_rate=config.SAMPLE_RATE,
    duration_samples=config.DURATION_SAMPLES,
    model_n_sources=config.N_SOURCES,
):
    model.train()
    all_epoch_losses = []
    all_epoch_train_si_snrs = []

    if not audio_files_with_speakers_list:
        train_logger.warning("No audio files provided for training. Skipping training.")
        return [], []

    # Group files by speaker
    speaker_to_files = defaultdict(list)
    for filepath, speaker_id in audio_files_with_speakers_list:
        speaker_to_files[speaker_id].append(filepath)

    # Filter out speakers with no files (should not happen if input is correct)
    speaker_to_files = {k: v for k, v in speaker_to_files.items() if v}

    if not speaker_to_files:
        train_logger.warning(
            "No speakers found after grouping files. Skipping training."
        )
        return [], []

    for epoch in range(epochs):
        # Determine how many mixtures we can potentially create.
        # This is a rough estimate for tqdm. The actual number might be less
        # if we can't form enough unique speaker combinations.
        # A simple estimate: total files / average sources per mix.
        total_files = len(audio_files_with_speakers_list)
        if min_sources == 0 and max_sources == 0:
            avg_sources_per_mix = 1
        elif min_sources == max_sources:
            avg_sources_per_mix = max(1, min_sources)
        else:
            avg_sources_per_mix = (min_sources + max_sources) / 2
            if avg_sources_per_mix == 0:
                avg_sources_per_mix = 1

        num_potential_mixtures = int(total_files / avg_sources_per_mix)

        if num_potential_mixtures == 0:
            train_logger.warning(
                f"Epoch {epoch+1}: Not enough files/speakers to create any mixtures. Skipping epoch."
            )
            all_epoch_losses.append(0)
            all_epoch_train_si_snrs.append(0)
            continue

        num_batches_in_epoch = num_potential_mixtures // batch_size
        if num_batches_in_epoch == 0:
            train_logger.warning(
                f"Epoch {epoch+1}: Not enough potential mixtures for a batch. Skipping epoch."
            )
            all_epoch_losses.append(0)
            all_epoch_train_si_snrs.append(0)
            continue

        total_epoch_loss = 0
        total_epoch_si_snr = 0
        actual_batches_processed_in_epoch = 0

        progress_bar = tqdm(range(num_batches_in_epoch), desc=f"Epoch {epoch+1}")

        for batch_idx in progress_bar:
            batch_mixtures_np_list = []
            batch_original_sources_list = []
            samples_generated_for_batch = 0

            for _ in range(batch_size):  # Try to create `batch_size` mixtures
                if min_sources > max_sources:
                    num_sources_for_this_mix = max_sources
                elif min_sources == max_sources:
                    num_sources_for_this_mix = min_sources
                else:
                    num_sources_for_this_mix = random.randint(min_sources, max_sources)

                if num_sources_for_this_mix == 0:
                    continue

                # Select unique speakers for this mixture
                available_speaker_ids = list(speaker_to_files.keys())
                if len(available_speaker_ids) < num_sources_for_this_mix:
                    # Not enough unique speakers available to form this mixture
                    # train_logger.debug(f"Not enough unique speakers ({len(available_speaker_ids)}) for {num_sources_for_this_mix} sources.")
                    break  # Stop trying to add more samples to this batch

                selected_speaker_ids = random.sample(
                    available_speaker_ids, num_sources_for_this_mix
                )

                files_for_this_mix = []
                valid_selection = True
                for spk_id in selected_speaker_ids:
                    if not speaker_to_files[
                        spk_id
                    ]:  # Should not happen if speaker_to_files is maintained
                        valid_selection = False
                        break
                    files_for_this_mix.append(random.choice(speaker_to_files[spk_id]))

                if not valid_selection:
                    # train_logger.debug("Failed to select files for chosen speakers.")
                    continue  # Try to form the next sample in batch if this one failed

                current_sources_np_list = []
                loading_error = False
                for file_path in files_for_this_mix:
                    source_np = utils.load_audio_segment(
                        file_path, sample_rate, duration_samples
                    )
                    if source_np is None:
                        train_logger.debug(
                            f"Skipping mixture due to loading error for one of: {files_for_this_mix}"
                        )
                        loading_error = True
                        break
                    current_sources_np_list.append(source_np)

                if (
                    loading_error
                    or len(current_sources_np_list) != num_sources_for_this_mix
                ):
                    continue

                (
                    mixture_np,
                    original_sources_this_mix,
                    _,
                ) = utils.create_mixture_from_sources(
                    current_sources_np_list,
                    noise_profile=None,
                    target_snr_db=None,
                    training_noise_level=noise_level,
                )
                if mixture_np.size == 0:
                    continue

                batch_mixtures_np_list.append(mixture_np)
                batch_original_sources_list.append(original_sources_this_mix)
                samples_generated_for_batch += 1

            if (
                samples_generated_for_batch == 0 and batch_idx > 0
            ):  # If a batch (not the first) is empty, maybe we ran out of speaker combos
                progress_bar.set_description(
                    f"Epoch {epoch+1} (No more unique speaker sets for full batch)"
                )
                break  # End epoch early if we can't form new batches

            if not batch_mixtures_np_list:
                if (
                    samples_generated_for_batch == 0 and batch_idx == 0
                ):  # First batch is empty
                    train_logger.warning(
                        f"Epoch {epoch+1}: Could not form any mixtures in the first batch attempt. Check speaker/file availability."
                    )
                # If batch is empty, skip processing for this iteration of progress_bar
                # This might happen if `num_batches_in_epoch` was an overestimate
                # and we ran out of unique speaker combinations.
                continue

            # --- Process the batch ---
            if not batch_mixtures_np_list:
                continue  # Skip if the batch ended up empty

            # Convert mixture list to tensor
            mixtures = (
                torch.tensor(np.array(batch_mixtures_np_list), dtype=torch.float32)
                .unsqueeze(1)
                .to(device)
            )  # (batch, 1, time)

            # Prepare target sources with zero-padding
            # Determine max length in the current batch after loading/padding
            max_batch_len = mixtures.shape[-1]
            batch_sources_tensor_list = (
                []
            )  # List of tensors for each source position (up to model_n_sources)

            for s_idx in range(
                model_n_sources
            ):  # Iterate up to the maximum number of sources
                source_tensors_for_this_pos = []
                for sample_in_batch_idx in range(
                    len(batch_original_sources_list)
                ):  # Iterate through samples in the batch
                    original_sources_this_sample = batch_original_sources_list[
                        sample_in_batch_idx
                    ]
                    if s_idx < len(original_sources_this_sample):
                        # This source position exists for this sample, use the actual source
                        source_np = original_sources_this_sample[s_idx]
                        source_tensor = torch.tensor(source_np, dtype=torch.float32)
                    else:
                        # This source position does NOT exist for this sample, use zeros
                        source_tensor = torch.zeros(
                            max_batch_len, dtype=torch.float32
                        )  # Pad with zeros

                    # Ensure all tensors in this position have the same length (max_batch_len)
                    current_len = source_tensor.shape[-1]
                    if current_len < max_batch_len:
                        padding = max_batch_len - current_len
                        source_tensor = func.pad(source_tensor, (0, padding))
                    elif current_len > max_batch_len:
                        source_tensor = source_tensor[
                            ..., :max_batch_len
                        ]  # Use ellipsis for potential multi-dim tensors

                    source_tensors_for_this_pos.append(source_tensor)

                if not source_tensors_for_this_pos:  # If list is empty, cannot stack
                    # This might happen if batch_original_sources_list was empty, though guarded earlier
                    # Or if model_n_sources is 0
                    if (
                        model_n_sources > 0
                    ):  # Only create dummy if model expects sources
                        dummy_tensor_for_stacking = torch.zeros(
                            (len(batch_original_sources_list), max_batch_len),
                            dtype=torch.float32,
                        )
                        batch_sources_tensor_list.append(dummy_tensor_for_stacking)
                    continue  # Skip appending if model_n_sources is 0 or list is empty

                batch_sources_tensor_list.append(
                    torch.stack(source_tensors_for_this_pos, dim=0)
                )  # Stack samples for this source position

            # Stack source positions to get the final target tensor (batch, N_SOURCES, time)
            if (
                not batch_sources_tensor_list
            ):  # If no source tensors were created (e.g. model_n_sources = 0)
                # Create a dummy tensor if model expects no sources, or handle error
                if model_n_sources == 0:
                    sources = torch.empty(
                        (len(batch_mixtures_np_list), 0, max_batch_len),
                        dtype=torch.float32,
                    ).to(device)
                else:
                    # This case should ideally not be reached if batch_mixtures_np_list is not empty
                    # and model_n_sources > 0. If it is, it indicates an issue in logic.
                    train_logger.warning(
                        "batch_sources_tensor_list is empty despite model_n_sources > 0. Skipping batch."
                    )
                    continue
            else:
                sources = torch.stack(batch_sources_tensor_list, dim=1).to(
                    device
                )  # (batch, model_n_sources, time)

            optimizer.zero_grad()
            estimated_sources = model(mixtures)  # (batch, model_n_sources, time)

            # Match length before loss calculation
            estimated_sources, sources = match_length(estimated_sources, sources)

            # si_snr_loss with PIT should handle the zero-padded targets correctly
            loss_val, si_snr_val = si_snr_loss(
                estimated_sources,
                sources,
                n_sources=model_n_sources,  # Use the model's N_SOURCES
                pit=True,
                reduction="mean",
            )
            loss_val.backward()
            optimizer.step()

            total_epoch_loss += loss_val.item()
            total_epoch_si_snr += si_snr_val.item()
            actual_batches_processed_in_epoch += 1
            progress_bar.set_postfix(
                {"Loss": f"{loss_val.item():.4f}", "SI-SNR": f"{si_snr_val.item():.2f}"}
            )

        # Calculate average loss and SI-SNR for the epoch
        avg_epoch_loss = (
            total_epoch_loss / actual_batches_processed_in_epoch
            if actual_batches_processed_in_epoch > 0
            else 0
        )
        avg_epoch_si_snr = (
            total_epoch_si_snr / actual_batches_processed_in_epoch
            if actual_batches_processed_in_epoch > 0
            else 0
        )
        all_epoch_losses.append(avg_epoch_loss)
        all_epoch_train_si_snrs.append(avg_epoch_si_snr)

        train_logger.info(
            f"Epoch {epoch+1} Average Loss: {avg_epoch_loss:.4f}, Average Train SI-SNR: {avg_epoch_si_snr:.2f} dB"
        )
    return all_epoch_losses, all_epoch_train_si_snrs


# --- Evaluation ---
def evaluate(
    model,
    current_test_files_with_speakers,
    device,
    min_sources=config.MIN_SOURCES_EVAL,
    max_sources=config.MAX_SOURCES_EVAL,
    batch_size=config.BATCH_SIZE_EVAL,
    target_snr_db=None,
    sample_rate=config.SAMPLE_RATE,
    duration_samples=config.DURATION_SAMPLES,
    save_audio_flag=False,
    num_samples_to_save=0,
    output_dir_base=config.OUTPUT_AUDIO_DIR_BASE,
    condition_name="eval",
    model_n_sources=config.N_SOURCES,
):
    model.eval()
    total_si_snr_val = 0
    num_mixtures_processed = 0
    saved_sample_count = 0

    if not current_test_files_with_speakers:
        eval_logger.error("No test files provided for evaluation.")
        return 0

    speaker_to_files_test = defaultdict(list)
    for filepath, speaker_id in current_test_files_with_speakers:
        speaker_to_files_test[speaker_id].append(filepath)

    speaker_to_files_test = {k: v for k, v in speaker_to_files_test.items() if v}

    if not speaker_to_files_test:
        eval_logger.error("No speakers found after grouping test files.")
        return 0

    # Estimate number of mixtures for tqdm
    total_test_files = len(current_test_files_with_speakers)
    if min_sources == 0 and max_sources == 0:
        avg_sources_per_mix = 1
    elif min_sources == max_sources:
        avg_sources_per_mix = max(1, min_sources)
    else:
        avg_sources_per_mix = (min_sources + max_sources) / 2
        if avg_sources_per_mix == 0:
            avg_sources_per_mix = 1

    num_potential_mixtures = int(total_test_files / avg_sources_per_mix)
    if num_potential_mixtures == 0:
        eval_logger.error(
            "Not enough test files/speakers to create any mixtures for evaluation."
        )
        return 0

    # We will iterate until we can't form more batches or meet num_potential_mixtures
    # The progress bar will be based on num_potential_mixtures, but actual processing might be less.

    with torch.no_grad():
        progress_bar = tqdm(
            total=num_potential_mixtures,
            desc=f"Evaluating (SNR: {target_snr_db if target_snr_db is not None else 'Clean'})",
        )

        mixtures_created_this_run = 0
        # Loop to create batches until we run out of options or hit a target
        while mixtures_created_this_run < num_potential_mixtures:
            batch_mixtures_np_list = []
            batch_original_sources_list = []
            batch_added_noise_np_list = []
            samples_generated_for_batch = 0

            for _ in range(batch_size):  # Try to fill a batch
                if (
                    mixtures_created_this_run + samples_generated_for_batch
                    >= num_potential_mixtures
                ):
                    break  # Reached estimated total

                if min_sources > max_sources:
                    num_sources_for_this_mix = max_sources
                elif min_sources == max_sources:
                    num_sources_for_this_mix = min_sources
                else:
                    num_sources_for_this_mix = random.randint(min_sources, max_sources)

                if num_sources_for_this_mix == 0:
                    continue

                available_speaker_ids_test = list(speaker_to_files_test.keys())
                if len(available_speaker_ids_test) < num_sources_for_this_mix:
                    # Not enough unique speakers for this mixture
                    break  # Stop trying to add to this batch

                selected_speaker_ids_test = random.sample(
                    available_speaker_ids_test, num_sources_for_this_mix
                )

                files_for_this_mix = []
                valid_selection = True
                for spk_id in selected_speaker_ids_test:
                    if not speaker_to_files_test[spk_id]:
                        valid_selection = False
                        break
                    files_for_this_mix.append(
                        random.choice(speaker_to_files_test[spk_id])
                    )

                if not valid_selection:
                    continue

                current_sources_np_list = []
                loading_error = False
                for file_path in files_for_this_mix:
                    source_np = utils.load_audio_segment(
                        file_path, sample_rate, duration_samples
                    )
                    if source_np is None:
                        loading_error = True
                        break
                    current_sources_np_list.append(source_np)

                if (
                    loading_error
                    or len(current_sources_np_list) != num_sources_for_this_mix
                ):
                    continue

                (
                    mixture_np,
                    original_sources_this_mix,
                    noise_added_np,
                ) = utils.create_mixture_from_sources(
                    current_sources_np_list,
                    noise_profile=None,
                    target_snr_db=target_snr_db,
                    training_noise_level=0,
                )
                if mixture_np.size == 0:
                    continue

                batch_mixtures_np_list.append(mixture_np)
                batch_original_sources_list.append(original_sources_this_mix)
                batch_added_noise_np_list.append(noise_added_np)
                samples_generated_for_batch += 1

            if (
                not batch_mixtures_np_list
            ):  # If batch is empty, we probably ran out of unique speaker combos
                break  # End evaluation loop

            # --- Process the batch ---
            # Convert mixture list to tensor
            mixtures_tensor = (
                torch.tensor(np.array(batch_mixtures_np_list), dtype=torch.float32)
                .unsqueeze(1)
                .to(device)
            )

            # Prepare target sources with zero-padding, similar to training
            max_batch_len = mixtures_tensor.shape[-1]
            batch_sources_tensor_list = []

            for s_idx in range(model_n_sources):
                source_tensors_for_this_pos = []
                for sample_in_batch_idx in range(len(batch_original_sources_list)):
                    original_sources_this_sample = batch_original_sources_list[
                        sample_in_batch_idx
                    ]
                    if s_idx < len(original_sources_this_sample):
                        source_np = original_sources_this_sample[s_idx]
                        source_tensor = torch.tensor(source_np, dtype=torch.float32)
                    else:
                        source_tensor = torch.zeros(max_batch_len, dtype=torch.float32)

                    current_len = source_tensor.shape[-1]
                    if current_len < max_batch_len:
                        padding = max_batch_len - current_len
                        source_tensor = func.pad(source_tensor, (0, padding))
                    elif current_len > max_batch_len:
                        source_tensor = source_tensor[..., :max_batch_len]

                    source_tensors_for_this_pos.append(source_tensor)

                if not source_tensors_for_this_pos:
                    if model_n_sources > 0:
                        dummy_tensor_for_stacking = torch.zeros(
                            (len(batch_original_sources_list), max_batch_len),
                            dtype=torch.float32,
                        )
                        batch_sources_tensor_list.append(dummy_tensor_for_stacking)
                    continue
                batch_sources_tensor_list.append(
                    torch.stack(source_tensors_for_this_pos, dim=0)
                )

            if not batch_sources_tensor_list:
                if model_n_sources == 0:
                    sources_tensor = torch.empty(
                        (len(batch_mixtures_np_list), 0, max_batch_len),
                        dtype=torch.float32,
                    ).to(device)
                else:
                    eval_logger.warning(
                        "batch_sources_tensor_list is empty in eval. Skipping batch."
                    )
                    progress_bar.update(
                        len(batch_mixtures_np_list) if batch_mixtures_np_list else 0
                    )
                    mixtures_created_this_run += (
                        len(batch_mixtures_np_list) if batch_mixtures_np_list else 0
                    )
                    num_mixtures_processed += (
                        len(batch_mixtures_np_list) if batch_mixtures_np_list else 0
                    )  # Ensure this is updated
                    continue
            else:
                sources_tensor = torch.stack(batch_sources_tensor_list, dim=1).to(
                    device
                )  # (batch, model_n_sources, time)

            estimated_sources_tensor = model(mixtures_tensor)
            estimated_sources_tensor, sources_tensor = match_length(
                estimated_sources_tensor, sources_tensor
            )

            _, si_snr_val_batch = si_snr_loss(
                estimated_sources_tensor,
                sources_tensor,
                n_sources=model_n_sources,
                pit=True,
                reduction="none",  # Get SI-SNR for each item in batch
            )
            # total_si_snr_val += si_snr_val_batch.item() * len(batch_mixtures_np_list) # Accumulate sum of SI-SNR for each sample
            total_si_snr_val += torch.sum(si_snr_val_batch).item()
            num_mixtures_processed += len(batch_mixtures_np_list)
            progress_bar.update(len(batch_mixtures_np_list))
            mixtures_created_this_run += len(batch_mixtures_np_list)

            # --- Saving audio samples during evaluation ---
            if save_audio_flag and saved_sample_count < num_samples_to_save:
                # Save samples from the current batch until num_samples_to_save is reached
                for sample_in_batch_idx in range(len(batch_mixtures_np_list)):
                    if saved_sample_count < num_samples_to_save:
                        mix_to_save = batch_mixtures_np_list[sample_in_batch_idx]
                        # Pass the original sources list for this sample (only the actual ones)
                        srcs_to_save = batch_original_sources_list[sample_in_batch_idx]
                        # Pass ALL estimated sources from the model (up to model_n_sources)
                        est_srcs_to_save = [
                            estimated_sources_tensor[sample_in_batch_idx, s_idx, :]
                            .cpu()
                            .numpy()
                            for s_idx in range(model_n_sources)
                        ]
                        noise_to_save = batch_added_noise_np_list[sample_in_batch_idx]

                        utils.save_evaluation_audio_samples(
                            mix_to_save,
                            srcs_to_save,  # Pass the actual original sources
                            est_srcs_to_save,  # Pass all estimated sources
                            noise_to_save,
                            sample_id=saved_sample_count + 1,
                            condition_name=condition_name,
                            base_output_dir=output_dir_base,
                            sr=sample_rate,
                        )
                        saved_sample_count += 1
                    else:
                        break  # Reached num_samples_to_save for saving

        progress_bar.close()

    avg_si_snr = (
        total_si_snr_val / num_mixtures_processed if num_mixtures_processed > 0 else 0
    )
    return avg_si_snr


# --- Main Execution ---
def main_run():
    global logger, train_logger, eval_logger

    # Initialize loggers
    logger = setup_main_logger()
    train_logger = setup_training_logger()
    eval_logger = setup_evaluation_logger()

    # Log system information and start message
    if callable(
        log_system_info
    ):  # Check if log_system_info is indeed imported and callable
        log_system_info(logger)
    else:
        logger.warning("log_system_info function not available from logger_config.")
    logger.info("Starting Conv-TasNet main execution...")

    # --- Device Configuration ---
    system_platform = platform.system()

    if system_platform == "Linux" or system_platform == "Windows":
        if torch.cuda.is_available():  # Covers NVIDIA GPUs and AMD GPUs via ROCm
            device = torch.device(
                "cuda"
            )  # PyTorch uses 'cuda' for both NVIDIA and ROCm

            if torch.version.cuda:  # Check for NVIDIA CUDA
                logger.info(
                    f"{system_platform} detected and NVIDIA CUDA is available. Using CUDA (device 'cuda')."
                )
            elif torch.version.hip:  # Check for AMD ROCm (HIP)
                logger.info(
                    f"{system_platform} detected and AMD ROCm (HIP) is available. Using ROCm (via 'cuda' device)."
                )
            else:
                # Fallback if torch.cuda.is_available() is true but neither specific version is identified
                logger.info(
                    f"{system_platform} detected and a CUDA-compatible GPU is available. Using it (device 'cuda')."
                )
        elif hasattr(torch, "xpu") and torch.xpu.is_available():  # Check for Intel XPU
            device = torch.device("xpu")
            logger.info(
                f"{system_platform} detected and Intel XPU is available (for Intel GPU/iGPU). Using XPU."
            )
        else:
            device = torch.device("cpu")
            logger.info(
                f"{system_platform} detected. No CUDA, ROCm, or XPU compatible GPU found. Using CPU."
            )
    elif system_platform == "Darwin":  # Darwin is macOS
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info(
                f"{system_platform} (macOS) detected and MPS is available. Using MPS for Metal GPU."
            )
        else:
            device = torch.device("cpu")
            logger.info(
                f"{system_platform} (macOS) detected, but MPS is not available. Using CPU."
            )
    else:  # Fallback to CPU for other systems
        device = torch.device("cpu")
        logger.info(
            f"Unsupported platform {system_platform} or specific GPU check failed. Using CPU."
        )

    logger.info(f"Selected device: {device}")

    # --- Initialize Metrics ---
    metrics = {
        "epochs_trained": 0,
        "total_training_time_seconds": 0,
        "training_runs": [],
        "evaluation_runs": [],
        "final_evaluation_metrics_per_snr": {},
        "system_info": {},  # Placeholder for system info if logged elsewhere
        "config_summary": {},  # Placeholder for config summary
    }
    # Attempt to load existing metrics if available
    if os.path.exists(METRICS_FILE_PATH):
        try:
            with open(METRICS_FILE_PATH, "r") as f:
                metrics = json.load(f)
            logger.info(f"Loaded existing metrics from {METRICS_FILE_PATH}")
        except json.JSONDecodeError:
            logger.error(
                f"Error decoding JSON from {METRICS_FILE_PATH}. Starting with fresh metrics."
            )
        except Exception as e:
            logger.error(
                f"Could not load metrics from {METRICS_FILE_PATH}: {e}. Starting with fresh metrics."
            )

    # --- Model Initialization ---
    logger.info("Initializing Conv-TasNet model...")
    model = ConvTasNet(
        n=N_ENCODER_FILTERS,
        l=L_CONV_KERNEL_SIZE,
        b=B_TCN_CHANNELS,
        h=H_TCN_CHANNELS,
        p=P_TCN_KERNEL_SIZE,
        x=X_TCN_BLOCKS,
        r=R_TCN_REPEATS,
        c=N_SOURCES,
        sc=Sc_TCN_CHANNELS,
        norm_type=NORM_TYPE,
        causal=CAUSAL_CONV,
    ).to(device)
    logger.info(f"Model initialized and moved to device: {device}")
    logger.debug(f"Model architecture: {model}")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    logger.info(f"Optimizer initialized: Adam with learning rate {LEARNING_RATE}")

    # --- Data Loading ---
    logger.info("Loading training and test audio filepaths...")
    try:
        train_files_with_speakers = utils.load_audio_filepaths(
            config.TRAIN_LIBRISPEECH_DIR, max_files=MAX_TRAIN_FILES
        )
        test_files_with_speakers = utils.load_audio_filepaths(
            config.TEST_LIBRISPEECH_DIR, max_files=MAX_TEST_FILES
        )
        logger.info(
            f"Loaded {len(train_files_with_speakers)} training files and {len(test_files_with_speakers)} test files."
        )
    except Exception as e:
        logger.error(f"Failed to load audio filepaths: {e}", exc_info=True)
        return  # Exit if data loading fails

    if not train_files_with_speakers:
        logger.warning(
            "No training files loaded. Training will be skipped. Check dataset paths and content."
        )
    if not test_files_with_speakers:
        logger.warning(
            "No test files loaded. Evaluation might be limited or skipped. Check dataset paths and content."
        )

    # --- Training Loop ---
    start_time_total_training = datetime.datetime.now()
    logger.info(f"Starting training for {EPOCHS_TO_TRAIN} epochs...")

    if train_files_with_speakers and EPOCHS_TO_TRAIN > 0:
        current_training_run_metrics = train(
            model,
            train_files_with_speakers,
            optimizer,
            device,
            epochs=EPOCHS_TO_TRAIN,
            batch_size=BATCH_SIZE_TRAIN,
            min_sources=config.MIN_SOURCES_TRAIN,
            max_sources=config.MAX_SOURCES_TRAIN,
            noise_level=NOISE_LEVEL_TRAIN,
            sample_rate=SAMPLE_RATE,
            duration_samples=DURATION_SAMPLES,
            model_n_sources=N_SOURCES,
        )
        metrics["training_runs"].append(current_training_run_metrics)
        metrics["epochs_trained"] += EPOCHS_TO_TRAIN
        logger.info("Training completed.")
    else:
        logger.info(
            "Skipping training as no training files are available or EPOCHS_TO_TRAIN is 0."
        )

    end_time_total_training = datetime.datetime.now()
    total_training_time = end_time_total_training - start_time_total_training
    metrics["total_training_time_seconds"] = total_training_time.total_seconds()
    logger.info(f"Total training time: {total_training_time}")

    # --- Evaluation ---
    logger.info("Starting evaluation...")
    final_si_snrs_per_condition = {}

    if test_files_with_speakers:
        # Evaluate on clean test set (no added noise, or specific SNR if desired)
        logger.info("Evaluating on the clean test set (original SNR)...")
        eval_metrics_clean = evaluate(
            model,
            test_files_with_speakers,
            device,
            min_sources=config.MIN_SOURCES_EVAL,
            max_sources=config.MAX_SOURCES_EVAL,
            batch_size=BATCH_SIZE_EVAL,
            target_snr_db=None,  # Evaluate with original SNR
            sample_rate=SAMPLE_RATE,
            duration_samples=DURATION_SAMPLES,
            save_audio_flag=(NUM_SAMPLES_TO_SAVE_EVAL > 0),
            num_samples_to_save=NUM_SAMPLES_TO_SAVE_EVAL,
            output_dir_base=OUTPUT_AUDIO_DIR_BASE,
            condition_name="clean_test_original_snr",
            model_n_sources=N_SOURCES,
        )
        if eval_metrics_clean and "average_si_snr" in eval_metrics_clean:
            final_si_snrs_per_condition["clean_test_original_snr"] = eval_metrics_clean[
                "average_si_snr"
            ]
            metrics.setdefault("evaluation_runs", []).append(
                {"condition": "clean_test_original_snr", **eval_metrics_clean}
            )

        # Evaluate across specified SNR conditions
        if SNR_CONDITIONS_DB:
            logger.info(f"Evaluating across SNR conditions: {SNR_CONDITIONS_DB} dB")
            for snr_db in SNR_CONDITIONS_DB:
                condition_name = f"test_snr_{snr_db}dB"
                logger.info(
                    f"Evaluating with target SNR: {snr_db} dB ({condition_name})"
                )
                current_evaluation_run_metrics = evaluate(
                    model,
                    test_files_with_speakers,
                    device,
                    min_sources=config.MIN_SOURCES_EVAL,
                    max_sources=config.MAX_SOURCES_EVAL,
                    batch_size=BATCH_SIZE_EVAL,
                    target_snr_db=snr_db,
                    sample_rate=SAMPLE_RATE,
                    duration_samples=DURATION_SAMPLES,
                    save_audio_flag=(NUM_SAMPLES_TO_SAVE_EVAL > 0),
                    num_samples_to_save=NUM_SAMPLES_TO_SAVE_EVAL,
                    output_dir_base=OUTPUT_AUDIO_DIR_BASE,
                    condition_name=condition_name,
                    model_n_sources=N_SOURCES,
                )
                # Store metrics for this SNR condition
                if (
                    isinstance(current_evaluation_run_metrics, dict)
                    and "average_si_snr" in current_evaluation_run_metrics
                ):
                    final_si_snrs_per_condition[condition_name] = (
                        current_evaluation_run_metrics["average_si_snr"]
                    )
                    metrics.setdefault("evaluation_runs", []).append(
                        {"condition": condition_name, **current_evaluation_run_metrics}
                    )
        else:
            logger.info(
                "No specific SNR conditions provided for evaluation. Skipping SNR-based evaluation."
            )
    else:
        logger.info("Skipping evaluation as no test files are available.")

    metrics["final_evaluation_metrics_per_snr"] = final_si_snrs_per_condition
    logger.info(
        f"Final evaluation SI-SNRs per condition: {final_si_snrs_per_condition}"
    )

    # --- Save Model and Metrics ---
    try:
        os.makedirs(MODEL_DIR, exist_ok=True)
        torch.save(model.state_dict(), MODEL_PATH)
        logger.info(f"Model saved to {MODEL_PATH}")

        with open(METRICS_FILE_PATH, "w") as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Metrics saved to {METRICS_FILE_PATH}")
    except Exception as e:
        logger.error(f"Error saving model or metrics: {e}", exc_info=True)

    logger.info("Conv-TasNet main execution finished.")


# Ensure the main execution guard is at the end of the script
if __name__ == "__main__":
    main_run()
