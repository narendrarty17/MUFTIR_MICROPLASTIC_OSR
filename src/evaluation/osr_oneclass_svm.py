import os
import json
import numpy as np
from collections import defaultdict
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

EMB_DIR = "data/standardized/embeddings_npz"
SPLIT_FILE = "data/splits/split_definition.json"

BIG_11 = {
    "PE", "PP", "PS", "PVC", "PET-PBT",
    "Nylon", "EVA", "PU", "PMMA", "PC", "SA-ABS"
}

# One-Class SVM hyperparameters (baseline)
NU = 0.05          # fraction of training data allowed outside boundary
GAMMA = "scale"    # kernel width

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def load_embeddings(sample_ids):
    embs = []
    labels = []

    for sid in sample_ids:
        path = os.path.join(EMB_DIR, f"{sid}.npz")
        data = np.load(path)

        X = data["embeddings"]
        polymer = sid.split("_")[0]

        embs.append(X)
        labels.extend([polymer] * X.shape[0])

    return np.vstack(embs), np.array(labels)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    with open(SPLIT_FILE, "r") as f:
        splits = json.load(f)

    # -------------------------------------------------------------
    # 1. Train one OC-SVM per polymer
    # -------------------------------------------------------------
    print("\n🔧 Training One-Class SVMs (per polymer)")
    svms = {}
    scalers = {}

    X_train, y_train = load_embeddings(splits["train"])

    for polymer in sorted(BIG_11):
        Xp = X_train[y_train == polymer]

        scaler = StandardScaler()
        Xp_scaled = scaler.fit_transform(Xp)

        svm = OneClassSVM(
            kernel="rbf",
            nu=NU,
            gamma=GAMMA
        )
        svm.fit(Xp_scaled)

        svms[polymer] = svm
        scalers[polymer] = scaler

        print(f"  ✔ Trained OC-SVM for {polymer} ({Xp.shape[0]} spectra)")

    # -------------------------------------------------------------
    # 2. Evaluation on each split
    # -------------------------------------------------------------
    print("\n📊 Evaluating OSR performance")

    for split_name, sample_ids in splits.items():
        X, y = load_embeddings(sample_ids)

        is_known_gt = np.isin(y, list(BIG_11))
        is_known_pred = []

        for x, polymer in zip(X, y):
            accepted = False

            # Try all known polymer models
            for p in BIG_11:
                x_scaled = scalers[p].transform(x.reshape(1, -1))
                pred = svms[p].predict(x_scaled)  # +1 inlier, -1 outlier

                if pred[0] == 1:
                    accepted = True
                    break

            is_known_pred.append(accepted)

        is_known_pred = np.array(is_known_pred)

        # ---------------------------------------------------------
        # Metrics
        # ---------------------------------------------------------
        if split_name == "test_closed_set":
            acc = (is_known_pred == is_known_gt).mean()
            print(f"\n✔ Closed-set acceptance accuracy: {acc:.4f}")

        else:
            rejection_rate = (~is_known_pred).mean()
            print(f"✔ {split_name} rejection rate: {rejection_rate:.4f}")


# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()