import os
import json
import random
from collections import defaultdict
import numpy as np

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DATA_DIR = os.path.join("data", "standardized", "spectra_npz_norm")
SPLIT_DIR = os.path.join("data", "splits")

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Closed-set training polymers (Big 11)
BIG_11 = {
    "PE", "PP", "PS", "PVC", "PET-PBT",
    "Nylon", "EVA", "PU", "PMMA", "PC", "SA-ABS"
}

# Open-set (never seen during training)
OPEN_SET = {
    "PLA", "PPS", "POM", "CA", "Phenoxy", "Silicone"
}

# PET (non PBT) is used only for stress testing in the paper
PET_STRESS = {"PET"}

# Non-plastic controls
NON_PLASTIC = {
    "Sand", "Cotton", "BSF"
}

# Split ratios (sample-wise)
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# Test = remainder


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def load_metadata(npz_path):
    """Load minimal metadata without touching spectra."""
    d = np.load(npz_path, allow_pickle=True)
    polymer = str(d["polymer"])
    condition = str(d["condition"])
    return polymer, condition


# ---------------------------------------------------------------------
# Main split logic
# ---------------------------------------------------------------------

def main():

    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".npz"))

    # Buckets
    groups = {
        "clean_big11": [],
        "stress_tests": [],
        "open_set": [],
        "non_plastic": []
    }

    # -------------------------------------------------------------
    # First pass: assign experimental role
    # -------------------------------------------------------------
    for fname in files:
        path = os.path.join(DATA_DIR, fname)
        sample_id = fname.replace(".npz", "")

        polymer, condition = load_metadata(path)

        if polymer in NON_PLASTIC:
            groups["non_plastic"].append(sample_id)

        elif polymer in OPEN_SET:
            groups["open_set"].append(sample_id)

        elif polymer in BIG_11:
            if condition == "clean":
                groups["clean_big11"].append(sample_id)
            else:
                groups["stress_tests"].append(sample_id)

        elif polymer in PET_STRESS:
            # PET (not PET-PBT) is stress-test only
            groups["stress_tests"].append(sample_id)

        else:
            raise ValueError(f"Unknown or unclassified polymer: {polymer}")

    # -------------------------------------------------------------
    # Second pass: split clean Big-11 samples sample-wise
    # -------------------------------------------------------------
    train, val, test = [], [], []

    by_polymer = defaultdict(list)
    for sid in groups["clean_big11"]:
        poly = sid.split("_")[0]
        by_polymer[poly].append(sid)

    for poly, samples in by_polymer.items():
        random.shuffle(samples)
        n = len(samples)

        n_train = int(TRAIN_FRAC * n)
        n_val = int(VAL_FRAC * n)

        train.extend(samples[:n_train])
        val.extend(samples[n_train:n_train + n_val])
        test.extend(samples[n_train + n_val:])

    # -------------------------------------------------------------
    # Final split definition
    # -------------------------------------------------------------
    split_definition = {
        "train": sorted(train),
        "validation": sorted(val),
        "test_closed_set": sorted(test),
        "open_set": sorted(groups["open_set"]),
        "stress_tests": sorted(groups["stress_tests"]),
        "non_plastic": sorted(groups["non_plastic"])
    }

    os.makedirs(SPLIT_DIR, exist_ok=True)
    out_path = os.path.join(SPLIT_DIR, "split_definition.json")

    with open(out_path, "w") as f:
        json.dump(split_definition, f, indent=2)

    # -------------------------------------------------------------
    # Summary (human-readable)
    # -------------------------------------------------------------
    print("\n✅ Split definition saved to:", out_path)
    print("\nSplit summary (sample counts):")
    for k, v in split_definition.items():
        print(f"  {k:15s}: {len(v)}")

    print("\nPer-polymer clean sample counts:")
    for poly, samples in sorted(by_polymer.items()):
        print(f"  {poly:10s}: {len(samples)}")


# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()