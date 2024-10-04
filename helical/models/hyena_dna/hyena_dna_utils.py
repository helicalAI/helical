from torch.utils.data import Dataset

class HyenaDNADataset(Dataset):
    """HyenaDNA dataset.

    Parameters
    ----------
        sequences: list
            The list of sequences.
        labels: list, default = None
            The list of labels.
    """
    def __init__(self, sequences, labels=None):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if self.labels is None:
            return self.sequences[idx]
        else:
            return self.sequences[idx], self.labels[idx]

    def set_labels(self, labels):
        self.labels = labels