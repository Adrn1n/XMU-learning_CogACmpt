import os
import torch
import numpy as np
import torch.optim as optim
from tqdm import tqdm
import random
import json

# Import config from the same directory (src/)
import config

# Import data from the utils subdirectory (src/utils/)
from utils import data as data_utils

# Import model and losses from their new files
from model import ConvTasNet
from losses import si_snr_loss, match_length

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


# --- Training and Evaluation ---
def train(
    model,
    audio_files_list,
    optimizer,
    device,
    epochs=EPOCHS_TO_TRAIN,
    batch_size=BATCH_SIZE_TRAIN,
    n_sources=N_SOURCES,
    noise_level=NOISE_LEVEL_TRAIN,
    sample_rate=SAMPLE_RATE,
    duration_samples=DURATION_SAMPLES,
):
    model.train()
    all_epoch_losses = []
    all_epoch_train_si_snrs = []

    for epoch in range(epochs):
        random.shuffle(audio_files_list)
        total_epoch_loss = 0
        total_epoch_si_snr = 0
        num_batches_in_epoch = 0

        # Create batches of file pairs
        file_pairs_for_epoch = []
        for i in range(
            0, len(audio_files_list) - 1, n_sources
        ):  # Ensure we have enough files for a pair
            if i + n_sources <= len(audio_files_list):
                file_pairs_for_epoch.append(tuple(audio_files_list[i : i + n_sources]))

        if not file_pairs_for_epoch:
            print(f"Epoch {epoch+1}: Not enough files to create pairs. Skipping epoch.")
            continue

        progress_bar = tqdm(
            range(0, len(file_pairs_for_epoch), batch_size), desc=f"Epoch {epoch+1}"
        )
        for i in progress_bar:
            batch_file_pairs = file_pairs_for_epoch[i : i + batch_size]
            if not batch_file_pairs:
                continue

            batch_mixtures_np_list = []
            batch_sources_np_list = [[] for _ in range(n_sources)]
            # batch_noises_np_list = [] # Not strictly needed for training loss, but create_mixture_from_sources returns it

            for file_pair in batch_file_pairs:
                s1_np = data_utils.load_audio_segment(
                    file_pair[0], sample_rate, duration_samples
                )
                s2_np = data_utils.load_audio_segment(
                    file_pair[1], sample_rate, duration_samples
                )

                if s1_np is None or s2_np is None:
                    print(f"Skipping pair due to loading error: {file_pair}")
                    continue

                mixture_np, s1_mixed_np, s2_mixed_np, _ = (
                    data_utils.create_mixture_from_sources(
                        s1_np,
                        s2_np,
                        noise_profile=None,  # No specific noise file for general training
                        target_snr_db=None,  # Not using target SNR for general training noise
                        training_noise_level=noise_level,
                    )
                )
                batch_mixtures_np_list.append(mixture_np)
                batch_sources_np_list[0].append(s1_mixed_np)
                batch_sources_np_list[1].append(s2_mixed_np)

            if not batch_mixtures_np_list:
                continue

            mixtures = (
                torch.tensor(np.array(batch_mixtures_np_list), dtype=torch.float32)
                .unsqueeze(1)
                .to(device)
            )  # (batch, 1, time)

            sources_list_torch = []
            for s_idx in range(n_sources):
                sources_list_torch.append(
                    torch.tensor(
                        np.array(batch_sources_np_list[s_idx]), dtype=torch.float32
                    )
                )
            sources = torch.stack(sources_list_torch, dim=1).to(device)

            optimizer.zero_grad()
            estimated_sources = model(mixtures)  # (batch, n_sources, time)

            # Match length before loss calculation
            estimated_sources, sources = match_length(estimated_sources, sources)

            loss_val, si_snr_val = si_snr_loss(
                estimated_sources,
                sources,
                n_sources=n_sources,
                pit=True,
                reduction="mean",
            )
            loss_val.backward()
            optimizer.step()

            total_epoch_loss += loss_val.item()
            total_epoch_si_snr += (
                si_snr_val.item()
            )  # si_snr_val is the actual SI-SNR from the loss function
            num_batches_in_epoch += 1
            progress_bar.set_postfix(
                {"Loss": f"{loss_val.item():.4f}", "SI-SNR": f"{si_snr_val.item():.2f}"}
            )

        avg_epoch_loss = (
            total_epoch_loss / num_batches_in_epoch if num_batches_in_epoch > 0 else 0
        )
        avg_epoch_si_snr = (
            total_epoch_si_snr / num_batches_in_epoch if num_batches_in_epoch > 0 else 0
        )
        all_epoch_losses.append(avg_epoch_loss)
        all_epoch_train_si_snrs.append(avg_epoch_si_snr)

        print(
            f"Epoch {epoch+1} Average Loss: {avg_epoch_loss:.4f}, Average Train SI-SNR: {avg_epoch_si_snr:.2f} dB"
        )
    return all_epoch_losses, all_epoch_train_si_snrs


def evaluate(
    model,
    current_test_files,
    device,
    n_sources=N_SOURCES,
    batch_size=BATCH_SIZE_EVAL,
    target_snr_db=None,
    sample_rate=SAMPLE_RATE,
    duration_samples=DURATION_SAMPLES,
    save_audio_flag=False,
    num_samples_to_save=0,
    output_dir_base=OUTPUT_AUDIO_DIR_BASE,
    condition_name="eval",
):
    model.eval()
    total_si_snr_val = 0
    num_batches = 0
    saved_sample_count = 0

    test_data_pairs = []
    for i in range(0, len(current_test_files) - 1, n_sources):
        if i + n_sources <= len(current_test_files):
            pair = current_test_files[i : i + n_sources]
            if len(pair) == n_sources:
                test_data_pairs.append(tuple(pair))

    if not test_data_pairs:
        print("Not enough test files to create pairs for evaluation.")
        return 0

    with torch.no_grad():
        progress_bar = tqdm(
            range(0, len(test_data_pairs), batch_size),
            desc=f"Evaluating (SNR: {target_snr_db if target_snr_db is not None else 'Clean'})",
        )
        for i in progress_bar:
            batch_pairs = test_data_pairs[i : i + batch_size]
            if not batch_pairs:
                continue

            current_batch_mixtures_np = []
            current_batch_sources_np = [[] for _ in range(n_sources)]
            current_batch_noises_np = []  # For saving noise if needed

            for file_pair_idx, file_pair in enumerate(batch_pairs):
                s1_np = data_utils.load_audio_segment(
                    file_pair[0], sample_rate, duration_samples
                )
                s2_np = data_utils.load_audio_segment(
                    file_pair[1], sample_rate, duration_samples
                )

                if s1_np is None or s2_np is None:
                    print(f"Skipping pair in eval due to loading error: {file_pair}")
                    continue

                mixture_np, s1_mixed_np, s2_mixed_np, noise_added_np = (
                    data_utils.create_mixture_from_sources(
                        s1_np,
                        s2_np,
                        noise_profile=None,  # Can be extended to use specific noise files from a noise dataset
                        target_snr_db=target_snr_db,
                    )
                )
                current_batch_mixtures_np.append(mixture_np)
                current_batch_sources_np[0].append(s1_mixed_np)
                current_batch_sources_np[1].append(s2_mixed_np)
                current_batch_noises_np.append(noise_added_np)  # Store noise for saving

            if not current_batch_mixtures_np:
                continue

            mixtures_tensor = (
                torch.tensor(np.array(current_batch_mixtures_np), dtype=torch.float32)
                .unsqueeze(1)
                .to(device)
            )

            sources_list_torch = []
            for s_idx in range(n_sources):
                sources_list_torch.append(
                    torch.tensor(
                        np.array(current_batch_sources_np[s_idx]), dtype=torch.float32
                    )
                )
            sources_tensor = torch.stack(sources_list_torch, dim=1).to(device)

            estimated_sources_tensor = model(mixtures_tensor)
            estimated_sources_tensor, sources_tensor = match_length(
                estimated_sources_tensor, sources_tensor
            )

            _, si_snr_val_batch = si_snr_loss(
                estimated_sources_tensor,
                sources_tensor,
                n_sources=n_sources,
                pit=True,
                reduction="mean",
            )
            total_si_snr_val += si_snr_val_batch.item()
            num_batches += 1
            progress_bar.set_postfix(
                {"Avg SI-SNR so far": f"{total_si_snr_val / num_batches:.2f} dB"}
            )

            # --- Saving audio samples during evaluation ---
            if save_audio_flag and saved_sample_count < num_samples_to_save:
                # Save samples from the current batch until num_samples_to_save is reached
                for sample_in_batch_idx in range(estimated_sources_tensor.shape[0]):
                    if saved_sample_count < num_samples_to_save:
                        # Extract individual samples from the batch
                        mix_to_save = current_batch_mixtures_np[sample_in_batch_idx]
                        srcs_to_save = [
                            current_batch_sources_np[s_idx][sample_in_batch_idx]
                            for s_idx in range(n_sources)
                        ]
                        est_srcs_to_save = [
                            estimated_sources_tensor[sample_in_batch_idx, s_idx, :]
                            .cpu()
                            .numpy()
                            for s_idx in range(n_sources)
                        ]
                        noise_to_save = current_batch_noises_np[sample_in_batch_idx]

                        data_utils.save_evaluation_audio_samples(
                            mix_to_save,
                            srcs_to_save,
                            est_srcs_to_save,
                            noise_to_save,
                            sample_id=saved_sample_count + 1,  # sample_id is 1-based
                            condition_name=condition_name,
                            base_output_dir=output_dir_base,
                            sr=sample_rate,
                        )
                        saved_sample_count += 1
                    else:
                        break  # Reached num_samples_to_save

    avg_si_snr = total_si_snr_val / num_batches if num_batches > 0 else 0
    return avg_si_snr


# --- Main Execution ---
def main_run():
    # Determine device (this part can be simplified by using config.DEVICE if set after torch import in config)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    # Update config.DEVICE if you want it to be globally accessible via config module after detection
    # config.DEVICE = device

    # Data directories from config
    train_librispeech_dir = config.TRAIN_LIBRISPEECH_DIR
    test_librispeech_dir = config.TEST_LIBRISPEECH_DIR
    noise_dir = config.NOISE_DIR  # For future use with specific noise files

    # Collect all .flac files using data_utils
    train_filepaths = data_utils.load_librispeech_filepaths(
        train_librispeech_dir, max_files=MAX_TRAIN_FILES
    )
    test_filepaths = data_utils.load_librispeech_filepaths(
        test_librispeech_dir, max_files=MAX_TEST_FILES
    )
    print(
        f"Found {len(train_filepaths)} training files and {len(test_filepaths)} test files."
    )

    # Limit number of files for faster execution (optional)
    if MAX_TRAIN_FILES > 0:
        train_filepaths = train_filepaths[:MAX_TRAIN_FILES]
    if MAX_TEST_FILES > 0:
        test_filepaths = test_filepaths[:MAX_TEST_FILES]
    print(
        f"Using {len(train_filepaths)} training files and {len(test_filepaths)} test files for this run."
    )

    if not train_filepaths or not test_filepaths or len(test_filepaths) < N_SOURCES:
        print(
            "Not enough audio files found in the specified directories or for creating pairs. Exiting."
        )
        return

    # Model, Optimizer
    model = ConvTasNet(
        N=N_ENCODER_FILTERS,
        L=L_CONV_KERNEL_SIZE,
        B=B_TCN_CHANNELS,
        H=H_TCN_CHANNELS,
        P=P_TCN_KERNEL_SIZE,
        X=X_TCN_BLOCKS,
        R=R_TCN_REPEATS,
        C=N_SOURCES,
        Sc=Sc_TCN_CHANNELS,
        norm_type=NORM_TYPE,
        causal=CAUSAL_CONV,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    if os.path.exists(MODEL_PATH):
        print(f"Loading pre-trained model from {MODEL_PATH}")
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}. Training from scratch.")
    else:
        print(f"No pre-trained model found at {MODEL_PATH}. Training from scratch.")

    epochs_to_train = 1  # Set to 0 to skip training and only evaluate
    training_losses = []
    training_si_snrs = []

    if epochs_to_train > 0:
        print("Starting training...")
        # Use epochs_to_train for the number of epochs
        training_losses, training_si_snrs = train(
            model,
            train_filepaths,
            optimizer,
            epochs=epochs_to_train,
            batch_size=2,
            device=device,
            n_sources=N_SOURCES,
            noise_level=0.01,
        )
        # Ensure model directory exists
        os.makedirs(MODEL_DIR, exist_ok=True)
        print(f"Saving model to {MODEL_PATH}")
        torch.save(model.state_dict(), MODEL_PATH)
        print("Model saved.")
    else:
        print("Skipping training as epochs_to_train is 0.")

    print("\n--- Starting SNR-based Evaluation ---")
    evaluation_final_si_snrs = {}

    for snr_db_eval in SNR_CONDITIONS_DB:
        snr_label = f"{snr_db_eval}dB" if snr_db_eval is not None else "clean"
        print(f"--- Evaluating for SNR: {snr_label} ---")

        # Pass save_audio_flag and other relevant params to evaluate
        avg_si_snr = evaluate(
            model,
            test_filepaths,
            device=device,
            n_sources=N_SOURCES,
            batch_size=BATCH_SIZE_EVAL,
            target_snr_db=snr_db_eval,
            sample_rate=SAMPLE_RATE,
            duration_samples=DURATION_SAMPLES,
            save_audio_flag=True,  # Enable saving for all conditions
            num_samples_to_save=NUM_SAMPLES_TO_SAVE_EVAL,
            output_dir_base=OUTPUT_AUDIO_DIR_BASE,
            condition_name=snr_label,
        )

        print(f"Average SI-SNR ({snr_label}): {avg_si_snr:.2f} dB")
        evaluation_final_si_snrs[snr_label] = avg_si_snr

    # Saving audio samples is now handled within the evaluate function
    # The old save_audio_samples function at the end of main.py can be removed.

    # Save metrics
    metrics_to_save = {
        "epochs_trained": epochs_to_train,
        "epoch_numbers": list(range(1, len(training_losses) + 1)),
        "training_avg_losses": training_losses,
        "training_avg_si_snrs": training_si_snrs,
        "evaluation_final_si_snrs": evaluation_final_si_snrs,
    }
    metrics_file_path = os.path.join(MODEL_DIR, "training_metrics.json")
    try:
        with open(metrics_file_path, "w") as f:
            json.dump(metrics_to_save, f, indent=4)
        print(f"Training metrics saved to {metrics_file_path}")
    except Exception as e:
        print(f"Error saving metrics to {metrics_file_path}: {e}")

    print("\nAll operations completed.")


if __name__ == "__main__":
    # Ensure model and output directories exist (can be done once at the start)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_AUDIO_DIR_BASE, exist_ok=True)
    main_run()
