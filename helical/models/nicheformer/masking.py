import torch
import random

MASK_TOKEN = 0
PAD_TOKEN = 1
CLS_TOKEN = 2


def complete_masking(batch, masking_p, n_tokens):
    """Apply masking to input batch for masked language modeling.

    Args:
        batch (dict): Input batch containing 'input_ids' and 'attention_mask'
        masking_p (float): Probability of masking a token
        n_tokens (int): Total number of tokens in vocabulary

    Returns:
        dict: Batch with masked indices and masking information
    """
    device = batch["input_ids"].device
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    # Create mask tensor (1 for tokens to be masked, 0 otherwise)
    prob = torch.rand(input_ids.shape, device=device)
    mask = (prob < masking_p) & (input_ids != PAD_TOKEN) & (input_ids != CLS_TOKEN)

    # For masked tokens:
    # - 80% replace with MASK token
    # - 10% replace with random token
    # - 10% keep unchanged
    masked_indices = input_ids.clone()

    # Calculate number of tokens to be masked
    num_tokens_to_mask = mask.sum().item()

    # Determine which tokens get which type of masking
    mask_mask = torch.rand(num_tokens_to_mask, device=device) < 0.8
    random_mask = (torch.rand(num_tokens_to_mask, device=device) < 0.5) & ~mask_mask

    # Apply MASK token (80% of masked tokens)
    masked_indices[mask] = torch.where(
        mask_mask,
        torch.tensor(MASK_TOKEN, device=device, dtype=torch.long),
        masked_indices[mask],
    )

    # Apply random tokens (10% of masked tokens)
    random_tokens = torch.randint(
        3,
        n_tokens,  # Start from 3 to avoid special tokens
        (random_mask.sum(),),
        device=device,
        dtype=torch.long,
    )
    masked_indices[mask][random_mask] = random_tokens

    # 10% remain unchanged

    return {
        "masked_indices": masked_indices,
        "attention_mask": attention_mask,
        "mask": mask,
        "input_ids": input_ids,
    }
