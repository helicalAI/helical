from torch.utils.data import Dataset

class HyenaDNADataset(Dataset):
    """HyenaDNA dataset.

    Parameters
    ----------
        sequences: list
            The list of sequences.
        labels: list
            The list
    """
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]