import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.datasets.spectral_dataset import SpectralDataset
from src.models.embedding_net import SpectralEmbeddingNet

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DATA_DIR = "data/standardized/spectra_npz_norm"
SPLIT_FILE = "data/splits/split_definition.json"
CHECKPOINT = "checkpoints/embedding_epoch_050.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 256
EPOCHS = 20
LEARNING_RATE = 1e-3

BIG_11_LABELS = [
    "PE", "PP", "PS", "PVC", "PET-PBT",
    "Nylon", "EVA", "PU", "PMMA", "PC", "SA-ABS"
]
LABEL_MAP = {p: i for i, p in enumerate(BIG_11_LABELS)}
NUM_CLASSES = len(BIG_11_LABELS)

# ---------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------

train_dataset = SpectralDataset(
    data_dir=DATA_DIR,
    split_file=SPLIT_FILE,
    split_name="train",
    label_map=LABEL_MAP,
    return_labels=True
)

test_dataset = SpectralDataset(
    data_dir=DATA_DIR,
    split_file=SPLIT_FILE,
    split_name="test_closed_set",
    label_map=LABEL_MAP,
    return_labels=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4
)

# ---------------------------------------------------------------------
# Load frozen embedding model
# ---------------------------------------------------------------------

embedding_model = SpectralEmbeddingNet(embedding_dim=128).to(DEVICE)
embedding_model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
embedding_model.eval()

for param in embedding_model.parameters():
    param.requires_grad = False

# ---------------------------------------------------------------------
# Linear classifier on top of embeddings
# ---------------------------------------------------------------------

classifier = nn.Linear(128, NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=LEARNING_RATE)

# ---------------------------------------------------------------------
# Training loop (classifier only)
# ---------------------------------------------------------------------

for epoch in range(1, EPOCHS + 1):
    classifier.train()
    running_loss = 0.0

    for x, y in train_loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        with torch.no_grad():
            emb = embedding_model(x)

        logits = classifier(emb)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(
        f"Epoch [{epoch:02d}/{EPOCHS}] "
        f"Train CE loss: {running_loss / len(train_loader):.4f}"
    )

# ---------------------------------------------------------------------
# Evaluation on closed-set test
# ---------------------------------------------------------------------

classifier.eval()
correct = 0
total = 0

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        emb = embedding_model(x)
        logits = classifier(emb)
        preds = logits.argmax(dim=1)

        correct += (preds == y).sum().item()
        total += y.size(0)

accuracy = correct / total
print(f"\n✅ Closed-set test accuracy: {accuracy:.4f}")