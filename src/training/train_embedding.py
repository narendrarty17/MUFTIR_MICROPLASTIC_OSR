import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.datasets.spectral_dataset import SpectralDataset
from src.datasets.polymer_batch_sampler import PolymerBalancedBatchSampler
from src.models.embedding_net import SpectralEmbeddingNet
from src.losses.contrastive_loss import ContrastiveLoss, make_pairs


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DATA_DIR = "data/standardized/spectra_npz_norm"
SPLIT_FILE = "data/splits/split_definition.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 50
LEARNING_RATE = 1e-3
EMBEDDING_DIM = 128
MARGIN = 1.0

POLYMERS_PER_BATCH = 4
SPECTRA_PER_POLYMER = 64   # Batch size = 256


# ---------------------------------------------------------------------
# Label map (Big-11 only)
# ---------------------------------------------------------------------

BIG_11_LABELS = [
    "PE", "PP", "PS", "PVC", "PET-PBT",
    "Nylon", "EVA", "PU", "PMMA", "PC", "SA-ABS"
]

LABEL_MAP = {p: i for i, p in enumerate(BIG_11_LABELS)}


# ---------------------------------------------------------------------
# TRAIN dataset & loader (balanced sampler)
# ---------------------------------------------------------------------

train_dataset = SpectralDataset(
    data_dir=DATA_DIR,
    split_file=SPLIT_FILE,
    split_name="train",
    label_map=LABEL_MAP,
    return_labels=True
)

train_sampler = PolymerBalancedBatchSampler(
    labels=train_dataset.labels,
    polymers_per_batch=POLYMERS_PER_BATCH,
    spectra_per_polymer=SPECTRA_PER_POLYMER
)

train_loader = DataLoader(
    train_dataset,
    batch_sampler=train_sampler,
    num_workers=4
)

# ---------------------------------------------------------------------
# Model, loss, optimizer
# ---------------------------------------------------------------------

model = SpectralEmbeddingNet(embedding_dim=EMBEDDING_DIM).to(DEVICE)
criterion = ContrastiveLoss(margin=MARGIN)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# ---------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------

os.makedirs("checkpoints", exist_ok=True)

for epoch in range(1, EPOCHS + 1):

    # ------------------- TRAIN -------------------
    model.train()
    running_loss = 0.0
    num_batches = 0

    for x, y in train_loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        embeddings = model(x)
        e1, e2, pair_y = make_pairs(embeddings, y)

        # Safety check (should not happen now)
        if e1 is None:
            continue

        loss = criterion(e1, e2, pair_y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    avg_loss = running_loss / max(1, num_batches)

    print(
        f"Epoch [{epoch:03d}/{EPOCHS}] | "
        f"Train contrastive loss: {avg_loss:.4f}"
    )

    # ------------------- SAVE -------------------
    torch.save(
        model.state_dict(),
        f"checkpoints/embedding_epoch_{epoch:03d}.pt"
    )

print("Training complete.")