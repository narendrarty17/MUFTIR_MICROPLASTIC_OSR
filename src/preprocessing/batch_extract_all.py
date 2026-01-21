import os
from extract_spectra import extract_hdf5_spectra

RAW_DIR = os.path.join("data", "raw", "hdf5_original")
OUT_DIR = os.path.join("data", "standardized", "spectra_npz")


def parse_filename(filename):
    """
    Parse polymer and condition from filename.
    This follows the conventions observed in the dataset.
    """
    name = filename.replace(".hdf5", "")

    # condition detection
    if "high-bkgd" in name:
        condition = "high-bkgd"
    elif "UV_deg" in name:
        condition = "UV_deg"
    elif "therm_deg" in name:
        condition = "therm_deg"
    elif "fibers" in name:
        condition = "fibers"
    else:
        condition = "clean"

    # polymer name = first token
    polymer = name.split("_")[0]

    return name, polymer, condition


def main():

    files = sorted(
        f for f in os.listdir(RAW_DIR)
        if f.lower().endswith(".hdf5")
    )

    print(f"Found {len(files)} HDF5 files")

    for fname in files:
        hdf5_path = os.path.join(RAW_DIR, fname)

        sample_id, polymer, condition = parse_filename(fname)

        extract_hdf5_spectra(
            hdf5_path=hdf5_path,
            output_dir=OUT_DIR,
            sample_id=sample_id,
            polymer=polymer,
            condition=condition
        )


if __name__ == "__main__":
    main()