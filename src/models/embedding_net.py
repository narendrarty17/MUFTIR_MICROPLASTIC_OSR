import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralEmbeddingNet(nn.Module):
    """
    1D CNN + ANN embedding network for µFTIR spectra.
    Input:  (B, 896)
    Output: (B, embedding_dim)
    """

    def __init__(self, embedding_dim=128):
        super().__init__()

        # --- CNN feature extractor ---
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool1d(2)

        # --- ANN projection head ---
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, embedding_dim)

    def forward(self, x):
        # x: (B, 896)
        x = x.unsqueeze(1)          # (B, 1, 896)

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # Global average pooling
        x = x.mean(dim=2)           # (B, 128)

        x = F.relu(self.fc1(x))
        embedding = self.fc2(x)     # (B, embedding_dim)

        return embedding
