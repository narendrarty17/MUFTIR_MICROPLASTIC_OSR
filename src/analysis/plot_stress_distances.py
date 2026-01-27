import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

EMB_DIR = "data/standardized/embeddings_npz"
SPLIT_FILE = "data/splits/split_definition.json"
OUT_DIR = "results/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def load_embeddings(sample_ids):
    X, y = [], []
    for sid in sample_ids:
        data = np.load(os.path.join(EMB_DIR, f"{sid}.npz"))
        embs = data["embeddings"]
        polymer = sid.split("_")[0]
        X.append(embs)
        y.extend([polymer] * embs.shape[0])
    return np.vstack(X), np.array(y)

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    with open(SPLIT_FILE, "r") as f:
        splits = json.load(f)

    # Compute clean centroids
    X_train, y_train = load_embeddings(splits["train"])
    centroids = {
        p: X_train[y_train == p].mean(axis=0)
        for p in np.unique(y_train)
    }

    # Stress distances
    X_stress, y_stress = load_embeddings(splits["stress_tests"])
    dist_by_polymer = defaultdict(list)

    for x, p in zip(X_stress, y_stress):
        if p in centroids:
            d = np.linalg.norm(x - centroids[p])
            dist_by_polymer[p].append(d)

    # Prepare plot
    polymers = sorted(dist_by_polymer.keys())
    data = [dist_by_polymer[p] for p in polymers]

    plt.figure(figsize=(12, 6))
    plt.boxplot(data, labels=polymers, showfliers=False)
    plt.ylabel("Distance to Clean Centroid")
    plt.title("Stress-Test Drift in Embedding Space")
    plt.xticks(rotation=45)
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, "stress_distance_boxplot.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"✅ Saved boxplot to {out_path}")

if __name__ == "__main__":
    main()