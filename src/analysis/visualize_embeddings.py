import os
import json
import numpy as np
import matplotlib.pyplot as plt
import umap

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

EMB_DIR = "data/standardized/embeddings_npz"
SPLIT_FILE = "data/splits/split_definition.json"

OUTPUT_DIR = "results/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_POINTS = 5000   # subsample for clarity & speed
RANDOM_SEED = 42

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def load_all_embeddings():
    X, y_poly, y_split = [], [], []

    with open(SPLIT_FILE, "r") as f:
        splits = json.load(f)

    for split_name, sample_ids in splits.items():
        for sid in sample_ids:
            path = os.path.join(EMB_DIR, f"{sid}.npz")
            data = np.load(path)

            embs = data["embeddings"]
            polymer = sid.split("_")[0]

            X.append(embs)
            y_poly.extend([polymer] * embs.shape[0])
            y_split.extend([split_name] * embs.shape[0])

    return np.vstack(X), np.array(y_poly), np.array(y_split)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    np.random.seed(RANDOM_SEED)

    X, polymers, splits = load_all_embeddings()

    # Subsample for visualization
    if X.shape[0] > MAX_POINTS:
        idx = np.random.choice(X.shape[0], MAX_POINTS, replace=False)
        X = X[idx]
        polymers = polymers[idx]
        splits = splits[idx]

    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        random_state=RANDOM_SEED
    )

    X_2d = reducer.fit_transform(X)

    # -------------------------------------------------------------
    # Plot by polymer (clean + open-set)
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 8))
    for p in np.unique(polymers):
        mask = polymers == p
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1], s=8, alpha=0.6, label=p)

    plt.title("UMAP of µFTIR Embeddings (Colored by Polymer)")
    plt.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "umap_by_polymer.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # Plot by split (clean / open / stress / non-plastic)
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 8))
    for s in np.unique(splits):
        mask = splits == s
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1], s=8, alpha=0.6, label=s)

    plt.title("UMAP of µFTIR Embeddings (Colored by Split)")
    plt.legend(markerscale=2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "umap_by_split.png"), dpi=300)
    plt.close()

    print("✅ UMAP visualizations saved to results/figures/")


if __name__ == "__main__":
    main()