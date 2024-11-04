from datasets import Dataset
import torch

class HelixRDataset(Dataset):
    def __init__(self, sequences, tokenizer):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = len(max(sequences, key=len))+10
        self.labels = None

    def set_labels(self, labels):
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        encoded = self.tokenizer(
            seq.astype(str).tolist(),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_special_tokens_mask=True,
            return_tensors='pt'
        )
        if self.labels is not None:
            return {
                'input_ids': encoded['input_ids'],
                'special_tokens_mask': encoded['special_tokens_mask'],
                'labels': torch.tensor(self.labels[idx])
            }
        else:
            return {
                'input_ids': encoded['input_ids'],
                'special_tokens_mask': encoded['special_tokens_mask'],
            }