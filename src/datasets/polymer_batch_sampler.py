import random
from torch.utils.data import Sampler


class PolymerBalancedBatchSampler(Sampler):
    """
    Batch sampler that enforces polymer diversity in every batch.

    Each batch contains:
        P polymers
        K spectra per polymer

    Batch size = P * K
    """

    def __init__(self, labels, polymers_per_batch=4, spectra_per_polymer=64):
        """
        Parameters
        ----------
        labels : list or array
            Polymer labels for each spectrum in the dataset
        polymers_per_batch : int
            Number of distinct polymers per batch (P)
        spectra_per_polymer : int
            Number of spectra per polymer per batch (K)
        """
        self.labels = labels
        self.polymers_per_batch = polymers_per_batch
        self.spectra_per_polymer = spectra_per_polymer

        # Build index list per polymer
        self.label_to_indices = {}
        for idx, lab in enumerate(labels):
            self.label_to_indices.setdefault(lab, []).append(idx)

        self.unique_labels = list(self.label_to_indices.keys())

        if len(self.unique_labels) < polymers_per_batch:
            raise ValueError(
                "Not enough polymers to form a balanced batch."
            )

        self.batch_size = polymers_per_batch * spectra_per_polymer

    def __iter__(self):
        # Shuffle indices for each polymer
        for indices in self.label_to_indices.values():
            random.shuffle(indices)

        # Copy index lists so we can pop safely
        available = {
            lab: indices.copy()
            for lab, indices in self.label_to_indices.items()
        }

        while True:
            # Choose P polymers
            chosen_polymers = random.sample(
                self.unique_labels, self.polymers_per_batch
            )

            batch = []

            for poly in chosen_polymers:
                if len(available[poly]) < self.spectra_per_polymer:
                    return  # stop iteration cleanly

                for _ in range(self.spectra_per_polymer):
                    batch.append(available[poly].pop())

            yield batch

    def __len__(self):
        # Conservative estimate
        min_batches = min(
            len(v) // self.spectra_per_polymer
            for v in self.label_to_indices.values()
        )
        return min_batches