from torch.utils.data import Dataset
import torch


class HelixmRNADataset(Dataset):
    def __init__(self, sequences, tokenizer):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = len(max(sequences, key=len)) + 10
        self.labels = None

    def set_labels(self, labels):
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seqs = self.sequences[idx]

        # Tokenize all sequences in the batch
        encoded = self.tokenizer(
            seqs,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )

        # Prepare output dictionary
        output = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "special_tokens_mask": encoded["special_tokens_mask"].squeeze(0),
        }

        # Add labels if they exist
        if self.labels is not None:
            output["labels"] = torch.tensor(self.labels[idx])

        return output
