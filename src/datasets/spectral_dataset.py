import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset

class SpectralDataset(Dataset):
    """
    PyTorch Dataset for µFTIR spectra.

    Each item corresponds to ONE spectrum.
    Sample-wise splitting is respected via split_definition.json.
    """

    def __init__(
        self,
        data_dir,
        split_file,
        split_name,
        label_map=None,
        return_labels=True
    ):
        """
        Parameters
        ----------
        data_dir : str
            Path to spectra_npz_norm directory
        split_file : str
            Path to split_definition.json
        split_name : str
            One of:
              'train', 'validation', 'test_closed_set',
              'open_set', 'stress_tests', 'non_plastic'
        label_map : dict or None
            Mapping polymer -> integer label (Big-11 only)
        return_labels : bool
            Whether to return labels (False for OSR / stress tests)
        """

        self.data_dir = data_dir
        self.split_name = split_name
        self.label_map = label_map
        self.return_labels = return_labels

        # ---------------------------------------------------------
        # Load split definition
        # ---------------------------------------------------------
        with open(split_file, "r") as f:
            splits = json.load(f)

        if split_name not in splits:
            raise ValueError(f"Unknown split: {split_name}")

        self.sample_ids = splits[split_name]

        # ---------------------------------------------------------
        # Load all spectra into indexable lists
        # ---------------------------------------------------------
        self.spectra = []
        self.labels = []
        self.sample_index = []

        for sid in self.sample_ids:
            npz_path = os.path.join(data_dir, f"{sid}.npz")
            data = np.load(npz_path, allow_pickle=True)

            spectra = data["spectra"]      # (N, 896)
            polymer = str(data["polymer"])

            for i in range(spectra.shape[0]):
                self.spectra.append(spectra[i])

                if self.return_labels:
                    if label_map is None:
                        raise ValueError("label_map must be provided when return_labels=True")
                    self.labels.append(label_map[polymer])

                self.sample_index.append(sid)

        self.spectra = np.asarray(self.spectra, dtype=np.float32)

        if self.return_labels:
            self.labels = np.asarray(self.labels, dtype=np.int64)

    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.spectra[idx])
        sid = self.sample_index[idx]  # <-- STRING like "PET_fibers_2"

        if self.return_labels:
            y = self.labels[idx]
            return x, y, sid
        else:
            return x, sid

