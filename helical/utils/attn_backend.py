import torch
from transformers.utils import is_flash_attn_2_available


def select_attn_backend(device, output_attentions: bool = False):
    """Select attention implementation and dtype for transformers from_pretrained.

    Returns
    -------
    (attn_impl, dtype) :
        ``("eager", float32)`` if attention weights are needed;
        ``("flash_attention_2", bfloat16)`` on CUDA when flash_attn is installed;
        ``("sdpa", float32)`` otherwise.
    """
    if output_attentions:
        return "eager", torch.float32
    if is_flash_attn_2_available() and torch.device(device).type == "cuda":
        return "flash_attention_2", torch.bfloat16
    return "sdpa", torch.float32
