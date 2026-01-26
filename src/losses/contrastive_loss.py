import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for metric learning.
    """

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, emb1, emb2, label):
        """
        emb1, emb2: (B, D)
        label: 1 = same class, 0 = different class
        """
        dist = F.pairwise_distance(emb1, emb2)

        loss_same = label * dist.pow(2)
        loss_diff = (1 - label) * F.relu(self.margin - dist).pow(2)

        return (loss_same + loss_diff).mean()


def make_pairs(embeddings, labels):
    """
    Create positive and negative embedding pairs from a batch.

    embeddings: (B, D)
    labels: (B,)
    """

    pairs_1, pairs_2, pair_labels = [], [], []

    label_to_indices = {}
    for i, lab in enumerate(labels):
        label_to_indices.setdefault(lab.item(), []).append(i)

    unique_labels = list(label_to_indices.keys())

    for i in range(len(labels)):
        anchor_label = labels[i].item()

        # Positive pair
        if len(label_to_indices[anchor_label]) > 1:
            pos = random.choice(label_to_indices[anchor_label])
            while pos == i:
                pos = random.choice(label_to_indices[anchor_label])

            pairs_1.append(i)
            pairs_2.append(pos)
            pair_labels.append(1)

        # Negative pair
        neg_label = random.choice([l for l in unique_labels if l != anchor_label])
        neg = random.choice(label_to_indices[neg_label])

        pairs_1.append(i)
        pairs_2.append(neg)
        pair_labels.append(0)

    return (
        embeddings[pairs_1],
        embeddings[pairs_2],
        torch.tensor(pair_labels, device=embeddings.device, dtype=torch.float32),
    )