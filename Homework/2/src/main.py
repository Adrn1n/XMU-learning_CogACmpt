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

from logger_config import (
    setup_main_logger,
    setup_training_logger,
    setup_evaluation_logger,
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
    audio_files_list,
    optimizer,
    device,
    epochs=config.EPOCHS_TO_TRAIN,
    batch_size=config.BATCH_SIZE_TRAIN,
    min_sources=config.MIN_SOURCES_TRAIN,  # Use min/max from config
    max_sources=config.MAX_SOURCES_TRAIN,  # Use min/max from config
    noise_level=config.NOISE_LEVEL_TRAIN,
    sample_rate=config.SAMPLE_RATE,
    duration_samples=config.DURATION_SAMPLES,
    model_n_sources=config.N_SOURCES,  # Pass the model's expected output size
):
    model.train()
    all_epoch_losses = []
    all_epoch_train_si_snrs = []

    for epoch in range(epochs):
        # Need a way to sample files for variable source counts
        # A simple approach: shuffle files and sample groups for each mixture
        shuffled_files = audio_files_list[:]  # Copy to avoid modifying original list
        random.shuffle(shuffled_files)

        # Determine how many mixtures we can create in this epoch
        # A rough estimate: total files // average number of sources
        if (
            min_sources == 0 and max_sources == 0
        ):  # Avoid division by zero if no sources
            avg_sources_per_mix = (
                1  # Default to 1 to avoid error, though this case should be handled
            )
        elif (
            min_sources == max_sources
        ):  # Avoid division by zero if min_sources can be 0
            avg_sources_per_mix = max(1, min_sources)  # Ensure it is at least 1
        else:
            avg_sources_per_mix = (min_sources + max_sources) / 2
            if avg_sources_per_mix == 0:  # Handle case where average is still zero
                avg_sources_per_mix = 1

        num_potential_mixtures = int(len(shuffled_files) / avg_sources_per_mix)
        if num_potential_mixtures == 0:
            train_logger.warning(
                f"Epoch {epoch+1}: Not enough files to create any mixtures. Skipping epoch."
            )
            all_epoch_losses.append(0)  # Append 0 loss if epoch is skipped
            all_epoch_train_si_snrs.append(0)  # Append 0 SI-SNR if epoch is skipped
            continue

        # We will generate batches of mixtures dynamically
        num_batches_in_epoch = num_potential_mixtures // batch_size
        if num_batches_in_epoch == 0:
            train_logger.warning(
                f"Epoch {epoch+1}: Not enough potential mixtures for a batch. Skipping epoch."
            )
            all_epoch_losses.append(0)
            all_epoch_train_si_snrs.append(0)
            continue

        # Use an index to track files consumed
        file_idx = 0
        total_epoch_loss = 0
        total_epoch_si_snr = 0
        actual_batches_processed_in_epoch = 0  # Correctly count batches processed

        progress_bar = tqdm(range(num_batches_in_epoch), desc=f"Epoch {epoch+1}")

        for batch_idx in progress_bar:
            batch_mixtures_np_list = []
            batch_original_sources_list = (
                []
            )  # Store lists of original sources for padding
            samples_generated_for_batch = (
                0  # Track samples successfully generated for this batch
            )

            for sample_in_batch_idx in range(batch_size):
                # Randomly determine the number of sources for this mixture
                if min_sources > max_sources:  # Safety check
                    train_logger.warning(
                        f"min_sources ({min_sources}) > max_sources ({max_sources}). Using max_sources."
                    )
                    num_sources_for_this_mix = max_sources
                elif min_sources == max_sources:
                    num_sources_for_this_mix = min_sources
                else:
                    num_sources_for_this_mix = random.randint(min_sources, max_sources)

                if num_sources_for_this_mix == 0:  # Skip if no sources are to be mixed
                    continue

                # Check if enough files are left
                if file_idx + num_sources_for_this_mix > len(shuffled_files):
                    # Not enough files for a full mixture, break or handle partial batch
                    # For simplicity here, let's break the inner sample loop
                    if (
                        samples_generated_for_batch == 0
                    ):  # If even the first sample can't be created
                        # This condition might lead to premature end of epoch if not enough files for *any* batch
                        pass  # Let the outer loop handle epoch termination if num_batches_in_epoch was 0
                    break  # Break the inner sample loop and process what we have

                # Select files for this mixture
                files_for_this_mix = shuffled_files[
                    file_idx : file_idx + num_sources_for_this_mix
                ]
                file_idx += num_sources_for_this_mix  # Advance the file index

                # Load sources
                current_sources_np_list = []
                loading_error = False
                for file_path in files_for_this_mix:
                    source_np = utils.load_audio_segment(
                        file_path, sample_rate, duration_samples
                    )
                    if source_np is None:
                        train_logger.debug(
                            f"Skipping mixture due to loading error: {files_for_this_mix}"
                        )
                        loading_error = True
                        break
                    current_sources_np_list.append(source_np)

                if (
                    loading_error
                    or len(current_sources_np_list) != num_sources_for_this_mix
                ):
                    continue  # Skip this mixture if loading failed

                # Create mixture
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
                if (
                    mixture_np.size == 0
                ):  # Skip if mixture is empty (e.g. all sources were empty)
                    continue

                batch_mixtures_np_list.append(mixture_np)
                batch_original_sources_list.append(
                    original_sources_this_mix
                )  # Store the list
                samples_generated_for_batch += 1

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
            actual_batches_processed_in_epoch += 1  # Increment for each batch processed
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
    current_test_files,
    device,
    min_sources=config.MIN_SOURCES_EVAL,  # Use min/max from config
    max_sources=config.MAX_SOURCES_EVAL,  # Use min/max from config
    batch_size=config.BATCH_SIZE_EVAL,
    target_snr_db=None,
    sample_rate=config.SAMPLE_RATE,
    duration_samples=config.DURATION_SAMPLES,
    save_audio_flag=False,
    num_samples_to_save=0,
    output_dir_base=config.OUTPUT_AUDIO_DIR_BASE,
    condition_name="eval",
    model_n_sources=config.N_SOURCES,  # Pass the model's expected output size
):
    model.eval()
    total_si_snr_val = 0
    num_mixtures_processed = 0  # Count individual mixtures, not batches
    saved_sample_count = 0

    # Similar dynamic batching approach as training
    shuffled_test_files = current_test_files[:]
    random.shuffle(shuffled_test_files)

    # Estimate number of mixtures
    if min_sources == 0 and max_sources == 0:
        avg_sources_per_mix = 1
    elif min_sources == max_sources:
        avg_sources_per_mix = max(1, min_sources)
    else:
        avg_sources_per_mix = (min_sources + max_sources) / 2
        if avg_sources_per_mix == 0:
            avg_sources_per_mix = 1

    num_potential_mixtures = int(len(shuffled_test_files) / avg_sources_per_mix)
    if num_potential_mixtures == 0:
        eval_logger.error(
            "Not enough test files to create any mixtures for evaluation."
        )
        return 0

    file_idx = 0
    # batch_count = 0 # Not strictly needed if progress is by sample

    with torch.no_grad():
        # Progress bar based on estimated number of mixtures
        progress_bar = tqdm(
            total=num_potential_mixtures,
            desc=f"Evaluating (SNR: {target_snr_db if target_snr_db is not None else 'Clean'})",
        )

        while (
            file_idx < len(shuffled_test_files)
            and num_mixtures_processed < num_potential_mixtures
        ):
            batch_mixtures_np_list = []
            batch_original_sources_list = []  # Store lists of original sources
            batch_added_noise_np_list = []  # Store noise for saving
            samples_generated_for_batch = 0

            # Generate samples for the current batch
            # Try to fill the batch, but stop if not enough files or potential mixtures reached
            while (
                samples_generated_for_batch < batch_size
                and file_idx < len(shuffled_test_files)
                and num_mixtures_processed + samples_generated_for_batch
                < num_potential_mixtures
            ):

                if min_sources > max_sources:
                    num_sources_for_this_mix = max_sources
                elif min_sources == max_sources:
                    num_sources_for_this_mix = min_sources
                else:
                    num_sources_for_this_mix = random.randint(min_sources, max_sources)

                if num_sources_for_this_mix == 0:
                    continue

                # Check if enough files are left for this *single* mixture
                if file_idx + num_sources_for_this_mix > len(shuffled_test_files):
                    break  # Not enough files for even one more full mixture

                # Select files for this mixture
                files_for_this_mix = shuffled_test_files[
                    file_idx : file_idx + num_sources_for_this_mix
                ]
                # file_idx is advanced *after* successful mixture creation to avoid skipping files on error

                # Load sources
                current_sources_np_list = []
                loading_error = False
                for file_path in files_for_this_mix:
                    source_np = utils.load_audio_segment(
                        file_path, sample_rate, duration_samples
                    )
                    if source_np is None:
                        loading_error = True
                        file_idx += num_sources_for_this_mix  # Consume files even on error to avoid infinite loop on bad files
                        break  # Break from loading sources for this mixture
                    current_sources_np_list.append(source_np)

                if (
                    loading_error
                    or len(current_sources_np_list) != num_sources_for_this_mix
                ):
                    if not loading_error:  # If not a loading error, still consume files
                        file_idx += num_sources_for_this_mix
                    continue  # Skip this mixture

                # Create mixture
                (
                    mixture_np,
                    original_sources_this_mix,
                    noise_added_np,
                ) = utils.create_mixture_from_sources(
                    current_sources_np_list,
                    noise_profile=None,
                    target_snr_db=target_snr_db,
                    training_noise_level=0,  # No extra training noise in eval
                )
                if mixture_np.size == 0:
                    file_idx += num_sources_for_this_mix  # Consume files
                    continue

                batch_mixtures_np_list.append(mixture_np)
                batch_original_sources_list.append(original_sources_this_mix)
                batch_added_noise_np_list.append(noise_added_np)
                samples_generated_for_batch += 1
                file_idx += num_sources_for_this_mix  # Advance file index *after* successful processing

            # --- Process the batch ---
            if not batch_mixtures_np_list:
                # This might happen if the remaining files are not enough for any mixture
                # or if all remaining potential mixtures were skipped due to errors.
                if (
                    file_idx >= len(shuffled_test_files)
                    or num_mixtures_processed >= num_potential_mixtures
                ):
                    break  # End evaluation if no more files or potential mixtures
                else:
                    continue  # Try to form the next batch

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
                        len(batch_mixtures_np_list)
                    )  # Still update progress for skipped items
                    num_mixtures_processed += len(batch_mixtures_np_list)
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
                n_sources=model_n_sources,  # Use the model's N_SOURCES
                pit=True,
                reduction="none",  # Get SI-SNR for each item in batch
            )
            # total_si_snr_val += si_snr_val_batch.item() * len(batch_mixtures_np_list) # Accumulate sum of SI-SNR for each sample
            total_si_snr_val += torch.sum(
                si_snr_val_batch
            ).item()  # Sum SI-SNR values for the batch
            num_mixtures_processed += len(
                batch_mixtures_np_list
            )  # Count total samples processed
            # batch_count += 1 # Count batches processed
            progress_bar.update(
                len(batch_mixtures_np_list)
            )  # Update progress bar by number of samples processed

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

        progress_bar.close()  # Close progress bar after the loop

    avg_si_snr = (
        total_si_snr_val / num_mixtures_processed if num_mixtures_processed > 0 else 0
    )
    return avg_si_snr


# --- Main Execution ---
def main_run():
    global logger, train_logger, eval_logger  # Ensure global loggers are used

    # Initialize loggers
    logger = setup_main_logger()
    train_logger = setup_training_logger()
    eval_logger = setup_evaluation_logger()

    logger.info("Starting Conv-TasNet main script.")

    # --- User choice for training or evaluation ---
    if os.path.exists(MODEL_PATH):
        choice = input(
            "Model already exists. Do you want to (t)rain again, (e)valuate, or (q)uit? [t/e/q]: "
        ).lower()
        if choice == "q":
            logger.info("Quitting.")
            return
        elif choice == "e":
            skip_training = True
            logger.info("Skipping training and proceeding to evaluation.")
        else:  # Default to training if 't' or anything else
            skip_training = False
            logger.info("Proceeding with training.")
    else:
        skip_training = False
        logger.info("No existing model found. Proceeding with training.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Data directories from config
    train_librispeech_dir = config.TRAIN_LIBRISPEECH_DIR
    test_librispeech_dir = config.TEST_LIBRISPEECH_DIR
    # noise_dir = config.NOISE_DIR # For future use with specific noise files

    # Collect all .flac files using data_utils
    train_filepaths = utils.load_audio_filepaths(
        train_librispeech_dir, max_files=config.MAX_TRAIN_FILES  # Use config value
    )
    test_filepaths = utils.load_audio_filepaths(
        test_librispeech_dir, max_files=config.MAX_TEST_FILES  # Use config value
    )
    logger.info(
        f"Found {len(train_filepaths)} training files and {len(test_filepaths)} test files."
    )

    # Limit number of files for faster execution (optional) - This is now handled by MAX_TRAIN_FILES/MAX_TEST_FILES in load_audio_filepaths
    # if config.MAX_TRAIN_FILES is not None and config.MAX_TRAIN_FILES > 0:
    #     train_filepaths = train_filepaths[: config.MAX_TRAIN_FILES]
    # if config.MAX_TEST_FILES is not None and config.MAX_TEST_FILES > 0:
    #     test_filepaths = test_filepaths[: config.MAX_TEST_FILES]

    logger.info(
        f"Using {len(train_filepaths)} training files and {len(test_filepaths)} test files for this run."
    )

    if (
        not train_filepaths
        or not test_filepaths
        or len(test_filepaths) < config.MIN_SOURCES_EVAL
    ):
        logger.error(
            f"Not enough audio files found. Need at least {config.MIN_SOURCES_EVAL} for evaluation. Exiting."
        )
        return
    if len(train_filepaths) < config.MIN_SOURCES_TRAIN:
        logger.error(
            f"Not enough audio files found for training. Need at least {config.MIN_SOURCES_TRAIN}. Exiting."
        )
        return

    # Model, Optimizer
    # Model is initialized with config.N_SOURCES (which is MAX_SOURCES_TRAIN)
    model = ConvTasNet(
        n=config.N_ENCODER_FILTERS,
        l=config.L_CONV_KERNEL_SIZE,
        b=config.B_TCN_CHANNELS,
        h=config.H_TCN_CHANNELS,
        p=config.P_TCN_KERNEL_SIZE,
        x=config.X_TCN_BLOCKS,
        r=config.R_TCN_REPEATS,
        c=config.N_SOURCES,  # Use the max number of sources
        sc=config.Sc_TCN_CHANNELS,
        norm_type=config.NORM_TYPE,
        causal=config.CAUSAL_CONV,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # ... (model loading remains similar) ...
    if os.path.exists(config.MODEL_PATH):
        logger.info(f"Loading pre-trained model from {config.MODEL_PATH}")
        try:
            model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model: {e}. Training from scratch.")
    else:
        logger.info(
            f"No pre-trained model found at {config.MODEL_PATH}. Training from scratch."
        )

    epochs_to_train = config.EPOCHS_TO_TRAIN
    # training_losses = # These are returned by train function
    # training_si_snrs = []

    if not skip_training:
        train_logger.info("Starting training process...")
        if epochs_to_train > 0:
            logger.info("Starting training...")
            training_losses, training_si_snrs = train(
                model,
                train_filepaths,
                optimizer,
                device=device,
                epochs=epochs_to_train,
                batch_size=config.BATCH_SIZE_TRAIN,
                min_sources=config.MIN_SOURCES_TRAIN,  # Pass min/max
                max_sources=config.MAX_SOURCES_TRAIN,  # Pass min/max
                noise_level=config.NOISE_LEVEL_TRAIN,
                sample_rate=config.SAMPLE_RATE,
                duration_samples=config.DURATION_SAMPLES,
                model_n_sources=config.N_SOURCES,  # Pass model's N_SOURCES
            )
            # Save model after training
            if not os.path.exists(config.MODEL_DIR):
                os.makedirs(config.MODEL_DIR)
            torch.save(model.state_dict(), config.MODEL_PATH)
            logger.info(f"Model saved to {config.MODEL_PATH}")
        else:
            logger.info("Skipping training as epochs_to_train is 0.")
            # If not training, initialize with empty lists or load previous if available
            training_losses = []
            training_si_snrs = []
    else:
        logger.info("Skipping training as per user choice or existing model.")
        # Load existing metrics if available to continue evaluation or visualization
        if os.path.exists(METRICS_FILE_PATH):
            with open(METRICS_FILE_PATH, "r") as f:
                metrics = json.load(f)
            logger.info(f"Loaded existing metrics from {METRICS_FILE_PATH}")
        else:
            logger.warning(
                f"No metrics file found at {METRICS_FILE_PATH}. Evaluation might be limited."
            )
            metrics = {}  # Initialize empty metrics if none exist

    # --- Evaluation Phase ---
    eval_logger.info("Starting evaluation process...")
    logger.info("\n--- Starting SNR-based Evaluation ---")
    evaluation_final_si_snrs = {}

    for snr_db_eval in config.SNR_CONDITIONS_DB:
        snr_label = f"{snr_db_eval}dB" if snr_db_eval is not None else "clean"
        logger.info(f"--- Evaluating for SNR: {snr_label} ---")

        avg_si_snr = evaluate(
            model,
            test_filepaths,
            device=device,
            min_sources=config.MIN_SOURCES_EVAL,  # Pass min/max for eval
            max_sources=config.MAX_SOURCES_EVAL,  # Pass min/max for eval
            batch_size=config.BATCH_SIZE_EVAL,
            target_snr_db=snr_db_eval,
            sample_rate=config.SAMPLE_RATE,
            duration_samples=config.DURATION_SAMPLES,
            save_audio_flag=True,
            num_samples_to_save=config.NUM_SAMPLES_TO_SAVE_EVAL,
            output_dir_base=config.OUTPUT_AUDIO_DIR_BASE,
            condition_name=snr_label,
            model_n_sources=config.N_SOURCES,  # Pass model's N_SOURCES
        )

        logger.info(f"Average SI-SNR ({snr_label}): {avg_si_snr:.2f} dB")
        evaluation_final_si_snrs[snr_label] = avg_si_snr

    # Save metrics
    metrics_to_save = {
        "epochs_trained": epochs_to_train,
        "epoch_numbers": (
            list(range(1, epochs_to_train + 1)) if epochs_to_train > 0 else []
        ),
        "training_avg_losses": training_losses,
        "training_avg_si_snrs": training_si_snrs,
        "evaluation_final_si_snrs": evaluation_final_si_snrs,
        # Add other relevant config parameters if needed for reproducibility
        "config_N_SOURCES": config.N_SOURCES,
        "config_MIN_SOURCES_TRAIN": config.MIN_SOURCES_TRAIN,
        "config_MAX_SOURCES_TRAIN": config.MAX_SOURCES_TRAIN,
        "config_MIN_SOURCES_EVAL": config.MIN_SOURCES_EVAL,
        "config_MAX_SOURCES_EVAL": config.MAX_SOURCES_EVAL,
    }
    if not os.path.exists(config.MODEL_DIR):
        os.makedirs(config.MODEL_DIR)
    with open(config.METRICS_FILE_PATH, "w") as f:
        json.dump(metrics_to_save, f, indent=4)
    logger.info(f"Training and evaluation metrics saved to {config.METRICS_FILE_PATH}")


if __name__ == "__main__":
    main_run()
