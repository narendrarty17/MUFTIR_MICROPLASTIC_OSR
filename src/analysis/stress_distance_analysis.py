import os
import json
import numpy as np
from collections import defaultdict

EMB_DIR = "data/standardized/embeddings_npz"
SPLIT_FILE = "data/splits/split_definition.json"

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def load_embeddings(sample_ids):
    X, y = [], []

    for sid in sample_ids:
        data = np.load(os.path.join(EMB_DIR, f"{sid}.npz"))
        embs = data["embeddings"]
        polymer = sid.split("_")[0]

        X.append(embs)
        y.extend([polymer] * embs.shape[0])

    return np.vstack(X), np.array(y)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    with open(SPLIT_FILE, "r") as f:
        splits = json.load(f)

    # Clean training centroids
    X_train, y_train = load_embeddings(splits["train"])
    centroids = {
        p: X_train[y_train == p].mean(axis=0)
        for p in np.unique(y_train)
    }

    # Stress tests
    X_stress, y_stress = load_embeddings(splits["stress_tests"])

    distances = defaultdict(list)

    for x, p in zip(X_stress, y_stress):
        if p in centroids:
            d = np.linalg.norm(x - centroids[p])
            distances[p].append(d)

    print("\n📊 Stress-test distance statistics (to clean centroids):")
    for p, dists in distances.items():
        dists = np.array(dists)
        print(
            f"{p:10s} | mean={dists.mean():.3f} "
            f"| std={dists.std():.3f} "
            f"| max={dists.max():.3f}"
        )


if __name__ == "__main__":
    main()