import os
import json
import numpy as np
import torch
from tqdm import tqdm

from src.datasets.spectral_dataset import SpectralDataset
from src.models.embedding_net import SpectralEmbeddingNet

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DATA_DIR = "data/standardized/spectra_npz_norm"
SPLIT_FILE = "data/splits/split_definition.json"
OUT_DIR = "data/standardized/embeddings_npz"
CHECKPOINT = "checkpoints/embedding_epoch_050.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 512

# ---------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------


def extract_split_embeddings(split_name, label_map=None):
    dataset = SpectralDataset(
        data_dir=DATA_DIR,
        split_file=SPLIT_FILE,
        split_name=split_name,
        label_map=label_map,
        return_labels=False
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )

    model = SpectralEmbeddingNet(embedding_dim=128).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    model.eval()

    os.makedirs(OUT_DIR, exist_ok=True)
    sample_buffers = {}

    print(f"\n🔍 STARTING DIAGNOSTICS FOR: {split_name}")
    print(f"----------------------------------------")

    with torch.no_grad():
        # Use enumerate so we can pinpoint the first batch for debugging
        for i, batch in enumerate(tqdm(loader, desc=f"Embedding {split_name}")):
            x = batch[0].to(DEVICE)
            sample_ids = batch[-1]

            # --- [DIAGNOSTIC BLOCK START] ---
            if i == 0:
                print(f"\n[DEBUG] Batch 0 Raw Input Shape: {x.shape}")
                print(f"[DEBUG] Batch 0 Raw Input Dimensions: {x.ndim}")
                if x.ndim == 3:
                    print(f"[DEBUG] 3D detected. Checking last dim: {x.shape[-1]}")
            # --- [DIAGNOSTIC BLOCK END] ---

            # Shape Fix Logic
            if x.ndim == 3 and x.shape[-1] == 1:
                x = x.squeeze(-1)
                if i == 0: print("[DEBUG] ACTION: Squeezed trailing 1. New shape:", x.shape)
            
            elif x.ndim == 1:
                x = x.unsqueeze(0)
                if i == 0: print("[DEBUG] ACTION: Unsqueezed single sample. New shape:", x.shape)

            # --- [FINAL CHECK BEFORE MODEL] ---
            if i == 0:
                print(f"[DEBUG] FINAL Shape entering model: {x.shape}")
                print(f"[DEBUG] Expected by Model: (Batch_Size, 896)\n")
            # ----------------------------------

            # Model Call
            emb = model(x).cpu().numpy()

            # Accumulate results
            for e, sid in zip(emb, sample_ids):
                if sid not in sample_buffers.items(): # Note: fixed .items() check logic below
                   if sid not in sample_buffers:
                       sample_buffers[sid] = []
                sample_buffers[sid].append(e)

    # Save per-sample embeddings
    for sid, embs in sample_buffers.items():
        embs = np.vstack(embs)
        out_path = os.path.join(OUT_DIR, f"{sid}.npz")
        np.savez(out_path, embeddings=embs, sample_id=sid, split=split_name)

    print(f"✅ Saved {len(sample_buffers)} samples for split '{split_name}'")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    with open(SPLIT_FILE, "r") as f:
        split_def = json.load(f)

    for split_name in split_def.keys():
        extract_split_embeddings(split_name)


if __name__ == "__main__":
    main()