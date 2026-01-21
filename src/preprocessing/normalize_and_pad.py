import os
import numpy as np

INPUT_DIR = os.path.join("data", "standardized", "spectra_npz")
OUTPUT_DIR = os.path.join("data", "standardized", "spectra_npz_norm")
TARGET_LENGTH = 896
ORIGINAL_LENGTH = 882


def minmax_normalize_per_spectrum(spectra):
    """
    Min–Max normalize each spectrum independently.
    spectra: np.ndarray of shape (N, 882)
    """
    mins = spectra.min(axis=1, keepdims=True)
    maxs = spectra.max(axis=1, keepdims=True)

    # Avoid division by zero (extremely rare but safe)
    denom = np.where((maxs - mins) == 0, 1.0, (maxs - mins))

    return (spectra - mins) / denom


def zero_pad_spectra(spectra, target_length=TARGET_LENGTH):
    """
    Zero-pad spectra on the right: (N, 882) → (N, 896)
    """
    n_samples, current_len = spectra.shape

    if current_len != ORIGINAL_LENGTH:
        raise ValueError(f"Expected length {ORIGINAL_LENGTH}, got {current_len}")

    pad_width = target_length - current_len
    if pad_width < 0:
        raise ValueError("Target length smaller than original length")

    return np.pad(
        spectra,
        pad_width=((0, 0), (0, pad_width)),
        mode="constant",
        constant_values=0.0
    )


def process_file(input_path, output_path):
    data = np.load(input_path, allow_pickle=True)

    spectra = data["spectra"]              # (N, 882)
    wavenumbers = data["wavenumbers"]       # (882,)

    # --- Step 4.1: normalize ---
    spectra_norm = minmax_normalize_per_spectrum(spectra)

    # --- Step 4.2: zero-pad ---
    spectra_pad = zero_pad_spectra(spectra_norm)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    np.savez_compressed(
        output_path,
        spectra=spectra_pad,
        wavenumbers=wavenumbers,
        polymer=data["polymer"],
        condition=data["condition"],
        sample_id=data["sample_id"]
    )

    print(f"Processed → {os.path.basename(output_path)} | shape {spectra_pad.shape}")


def main():

    files = sorted(
        f for f in os.listdir(INPUT_DIR)
        if f.endswith(".npz")
    )

    print(f"Found {len(files)} standardized files")

    for fname in files:
        input_path = os.path.join(INPUT_DIR, fname)
        output_path = os.path.join(OUTPUT_DIR, fname)

        process_file(input_path, output_path)


if __name__ == "__main__":
    main()