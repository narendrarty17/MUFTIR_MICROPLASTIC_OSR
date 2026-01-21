import os
import h5py

def inspect_hdf5(file_path):
    print("\n" + "=" * 80)
    print(f"Inspecting file: {os.path.basename(file_path)}")
    print("=" * 80)

    with h5py.File(file_path, "r") as f:

        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"[DATASET] {name}")
                print(f"          shape: {obj.shape}")
                print(f"          dtype: {obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"[GROUP]   {name}")

        f.visititems(visitor)

    print("=" * 80)
    print("Inspection complete.\n")


if __name__ == "__main__":

    RAW_HDF5_DIR = os.path.join("data", "raw", "hdf5_original")

    files_to_inspect = [
        "PE_clean_1-1.hdf5",
        "PE_high-bkgd_1.hdf5",
        "PET_fibers_1.hdf5",
        "PE_UV_deg.hdf5"
    ]

    for fname in files_to_inspect:
        file_path = os.path.join(RAW_HDF5_DIR, fname)
        if os.path.isfile(file_path):
            inspect_hdf5(file_path)
        else:
            print(f"⚠️ File not found: {file_path}")