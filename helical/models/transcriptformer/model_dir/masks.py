def pad_mask_factory(mask):
    def pad_mask(b, h, q_idx, kv_idx):
        if mask is None:
            return True
        m = mask[b, kv_idx]
        return m

    return pad_mask


def causal_mask_factory(start_row=0, start_col=0, offset=0):
    def causal_mask(b, h, q_idx, kv_idx):
        m = (q_idx >= kv_idx + offset) & (q_idx >= start_row) & (kv_idx >= start_col)
        return m

    return causal_mask
