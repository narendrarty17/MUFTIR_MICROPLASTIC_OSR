import os
import json
import numpy as np
from collections import defaultdict

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

EMB_DIR = "data/standardized/embeddings_npz"
SPLIT_FILE = "data/splits/split_definition.json"

BIG_11 = {
    "PE", "PP", "PS", "PVC", "PET-PBT",
    "Nylon", "EVA", "PU", "PMMA", "PC", "SA-ABS"
}

# Distance percentile for thresholding (baseline choice)
THRESHOLD_PERCENTILE = 95.0

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def load_embeddings(sample_ids):
    """Load and concatenate embeddings for given sample IDs."""
    all_embs = []
    all_labels = []

    for sid in sample_ids:
        path = os.path.join(EMB_DIR, f"{sid}.npz")
        data = np.load(path)

        embs = data["embeddings"]
        polymer = sid.split("_")[0]

        all_embs.append(embs)
        all_labels.extend([polymer] * embs.shape[0])

    return np.vstack(all_embs), np.array(all_labels)


def compute_centroids(embeddings, labels):
    """Compute centroid per polymer."""
    centroids = {}
    for polymer in np.unique(labels):
        centroids[polymer] = embeddings[labels == polymer].mean(axis=0)
    return centroids


def min_distance_to_centroids(x, centroids):
    """Return minimum Euclidean distance to any centroid."""
    return min(
        np.linalg.norm(x - c)
        for c in centroids.values()
    )


# ---------------------------------------------------------------------
# Main OSR pipeline
# ---------------------------------------------------------------------

def main():

    with open(SPLIT_FILE, "r") as f:
        splits = json.load(f)

    # -------------------------------------------------------------
    # 1. Fit centroids on TRAIN (known polymers only)
    # -------------------------------------------------------------
    train_ids = splits["train"]

    X_train, y_train = load_embeddings(train_ids)

    centroids = compute_centroids(X_train, y_train)

    print(f"✅ Computed centroids for {len(centroids)} polymers")

    # -------------------------------------------------------------
    # 2. Determine rejection threshold from training distances
    # -------------------------------------------------------------
    train_distances = [
        min_distance_to_centroids(x, centroids)
        for x in X_train
    ]

    threshold = np.percentile(train_distances, THRESHOLD_PERCENTILE)

    print(f"✅ OSR distance threshold (p{THRESHOLD_PERCENTILE}): {threshold:.4f}")

    # -------------------------------------------------------------
    # 3. Evaluate on all splits
    # -------------------------------------------------------------
    for split_name, sample_ids in splits.items():

        X, y = load_embeddings(sample_ids)

        known_mask = np.isin(y, list(BIG_11))

        distances = np.array([
            min_distance_to_centroids(x, centroids)
            for x in X
        ])

        predicted_known = distances <= threshold

        # Metrics
        if split_name == "test_closed_set":
            acc = (predicted_known == known_mask).mean()
            print(f"\n📊 Closed-set acceptance accuracy: {acc:.4f}")

        else:
            rejection_rate = (~predicted_known).mean()
            print(f"\n📊 {split_name} rejection rate: {rejection_rate:.4f}")


# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()