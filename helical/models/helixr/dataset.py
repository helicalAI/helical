from datasets import Dataset
import torch

class HelixRDataset(Dataset):
    def __init__(self, sequences, tokenizer):
        self.sequences = sequences
        # self.values = values
        self.tokenizer = tokenizer
        self.max_length = 0

        # if len(self.sequences) != len(self.values):
        #     raise ValueError("The number of sequences and values must be the same.")

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

        return {
            'input_ids': encoded['input_ids'],
            'special_tokens_mask': encoded['special_tokens_mask'],
            # 'labels': torch.tensor(self.values[idx], dtype=torch.long)
        }