import torch
from typing import Optional
from flash_attn import bert_padding

def gen_flash_attn_padding_info(
    bsz: int,
    S: int,
    past_key_len: int,
    device: torch.device,
    attention_mask_in_length: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
):
    flash_attn_padding_info = {}
    if attention_mask_in_length is None:
        key_padding_mask = attention_mask
        if key_padding_mask is None:
            key_padding_mask = torch.ones((bsz, past_key_len + S),
                                          dtype=torch.bool,
                                          device=device)
        query_padding_mask = key_padding_mask[:, -S:]
        unpadding_function = bert_padding.unpad_input
    else:
        key_padding_mask = attention_mask_in_length
        query_padding_mask = attention_mask_in_length
        unpadding_function = bert_padding.unpad_input_for_concatenated_sequences

    _, indices_q, cu_seqlens_q, max_seqlen_q, *_ = unpadding_function(
        torch.empty(bsz, S, 1, device=device),
        query_padding_mask,
    )
    _, indices_k, cu_seqlens_k, max_seqlen_k, *_ = unpadding_function(
        torch.empty(bsz, past_key_len + S, 1, device=device),
        key_padding_mask,
    )
    _, indices_v, *_ = unpadding_function(
        torch.empty(bsz, past_key_len + S, 1, device=device),
        key_padding_mask,
    )

    flash_attn_padding_info['indices_q'] = indices_q
    flash_attn_padding_info['indices_k'] = indices_k
    flash_attn_padding_info['indices_v'] = indices_v
    flash_attn_padding_info['cu_seqlens_q'] = cu_seqlens_q
    flash_attn_padding_info['cu_seqlens_k'] = cu_seqlens_k
    flash_attn_padding_info['max_seqlen_q'] = max_seqlen_q
    flash_attn_padding_info['max_seqlen_k'] = max_seqlen_k
    return flash_attn_padding_info