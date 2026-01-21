import os
import h5py
import numpy as np

def extract_hdf5_spectra(
    hdf5_path,
    output_dir,
    sample_id,
    polymer,
    condition
):
    """
    Extract all spectra from a single HDF5 file and save as a standardized NPZ.

    Parameters
    ----------
    hdf5_path : str
        Path to input .hdf5 file
    output_dir : str
        Directory to store standardized .npz files
    sample_id : str
        Unique identifier for the plastic sample (e.g. PE_clean_1-1)
    polymer : str
        Polymer label (e.g. PE, PET, Nylon)
    condition : str
        Condition label (clean, high-bkgd, UV_deg, therm_deg, fibers)
    """

    spectra = []
    wavenumbers = None

    with h5py.File(hdf5_path, "r") as f:

        # --- read wavenumbers once ---
        if "wavenumbers" not in f:
            raise KeyError(f"'wavenumbers' dataset not found in {hdf5_path}")

        wavenumbers = np.array(f["wavenumbers"], dtype=np.float64)

        if wavenumbers.shape[0] != 882:
            raise ValueError("Unexpected wavenumber length")

        # --- iterate over spectral datasets ---
        for key in f.keys():

            # skip non-spectral datasets
            if key == "wavenumbers":
                continue

            # numeric dataset names = spectra
            try:
                int(key)
            except ValueError:
                continue

            spectrum = np.array(f[key], dtype=np.float64)

            if spectrum.shape != (882,):
                raise ValueError(f"Unexpected spectrum shape in {hdf5_path}, key={key}")

            spectra.append(spectrum)

    spectra = np.stack(spectra, axis=0)  # (N, 882)

    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"{sample_id}.npz")

    np.savez_compressed(
        output_path,
        spectra=spectra,
        wavenumbers=wavenumbers,
        polymer=polymer,
        condition=condition,
        sample_id=sample_id
    )

    print(
        f"Saved {spectra.shape[0]} spectra "
        f"from '{sample_id}' → {output_path}"
    )

