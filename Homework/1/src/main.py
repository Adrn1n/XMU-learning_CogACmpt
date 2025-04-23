"""
Audio Processing Script - Based on experiment requirements:
1. Reverse original audio files
2. Slice each audio file and randomly rearrange segments of the same sound
3. Normalize the loudness of all audio files
"""

from pathlib import Path
import os
import librosa
import numpy as np
import soundfile as sf
import random

# Define paths
BASE_DIR = Path("Homework/1/")
ORIGINAL_DIR = BASE_DIR / "assets/audio/org"
REVERSE_DIR = BASE_DIR / "assets/audio/reverse"
RANDOM_DIR = BASE_DIR / "assets/audio/random"

# Create output directories (if they don't exist)
REVERSE_DIR.mkdir(exist_ok=True, parents=True)
RANDOM_DIR.mkdir(exist_ok=True, parents=True)

# Set target loudness parameter (in dB)
TARGET_LOUDNESS = -20.0


def normalize_loudness(audio, target_loudness):
    """Normalize the loudness of the audio"""
    # Calculate current loudness
    y_mono = librosa.to_mono(audio) if audio.ndim > 1 else audio
    current_loudness = librosa.amplitude_to_db(np.abs(y_mono).mean())

    # Calculate gain
    gain = target_loudness - current_loudness

    # Apply gain
    return audio * (10 ** (gain / 20.0))


def reverse_audio(file_path, output_dir):
    """Reverse audio and save"""
    filename = os.path.basename(file_path)
    file_base, file_ext = os.path.splitext(filename)
    output_path = os.path.join(output_dir, f"{file_base}_rev{file_ext}")

    # Load audio
    audio, sr = librosa.load(file_path, sr=None)

    # Reverse audio
    reversed_audio = audio[::-1]

    # Normalize loudness
    normalized_audio = normalize_loudness(reversed_audio, TARGET_LOUDNESS)

    # Save reversed audio
    sf.write(output_path, normalized_audio, sr)

    print(f"Reversed audio saved: {output_path}")
    return normalized_audio, sr, output_path


def create_random_segments(
    file_path, output_dir, segment_duration=0.1, min_segments=10
):
    """Slice and randomly rearrange segments of a single audio file"""
    filename = os.path.basename(file_path)
    file_base, file_ext = os.path.splitext(filename)
    output_path = os.path.join(output_dir, f"{file_base}_random{file_ext}")

    # Load audio
    audio, sr = librosa.load(file_path, sr=None)

    # Normalize loudness
    audio = normalize_loudness(audio, TARGET_LOUDNESS)

    # Calculate segment parameters
    segment_len = int(segment_duration * sr)  # Number of samples per segment
    total_samples = len(audio)

    # Ensure there are enough samples to create the minimum number of segments
    if total_samples < segment_len * min_segments:
        segment_len = total_samples // min_segments
        if segment_len < 1:  # Audio is too short
            print(f"Warning: Audio {filename} is too short for segmentation")
            return

    # Split audio into segments
    segments = []
    for start in range(0, total_samples - segment_len + 1, segment_len):
        segment = audio[start : start + segment_len]
        segments.append(segment)

    # Add the last segment if it's not a full segment_len
    if total_samples % segment_len:
        last_segment = audio[-(total_samples % segment_len) :]
        if (
            len(last_segment) > segment_len // 2
        ):  # Only add if the last segment is more than half the segment length
            segments.append(last_segment)

    # Randomly shuffle segment order
    random.shuffle(segments)

    # Recombine segments
    shuffled_audio = np.concatenate(segments)

    # Save the shuffled audio
    sf.write(output_path, shuffled_audio, sr)

    print(f"Randomly shuffled audio saved: {output_path}")


def main():
    """Main function"""
    # Get paths to all original audio files
    original_files = [
        os.path.join(ORIGINAL_DIR, f)
        for f in os.listdir(ORIGINAL_DIR)
        if f.endswith(".wav") or f.endswith(".mp3")
    ]

    print(f"Found {len(original_files)} original audio files")

    # Process each original audio file
    for file_path in original_files:
        # Reverse audio
        reverse_audio(file_path, REVERSE_DIR)

        # Slice and randomly rearrange the original audio
        create_random_segments(file_path, RANDOM_DIR)

    print("Processing complete!")


if __name__ == "__main__":
    main()
