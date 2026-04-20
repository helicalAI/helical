import torch
from transformers.utils import is_flash_attn_2_available


def select_attn_backend(
    device, output_attentions: bool = False, supports_fa2: bool = False
):
    """Select attention implementation and dtype for transformers from_pretrained.

    Parameters
    ----------
    supports_fa2 : bool
        Whether the target model class declares Flash Attention 2 support via HF's
        ``_supports_flash_attn``/``_supports_flash_attn_2`` dispatcher. Required for the
        ``"flash_attention_2"`` branch to be selected; models without it must wire
        flash_attn directly at the consumer site.

    Returns
    -------
    (attn_impl, dtype) :
        ``("eager", float32)`` if attention weights are needed;
        ``("flash_attention_2", bfloat16)`` on CUDA when flash_attn is installed and the
        model supports it;
        ``("sdpa", float32)`` otherwise.
    """
    if output_attentions:
        return "eager", torch.float32
    if (
        supports_fa2
        and is_flash_attn_2_available()
        and torch.device(device).type == "cuda"
    ):
        return "flash_attention_2", torch.bfloat16
    return "sdpa", torch.float32
