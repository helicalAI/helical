import numpy as np
import pytest
import torch
import torch.nn as nn

from helical.utils.transformer_encoder import (
    LoraCompatibleSelfAttention,
    TransformerEncoder,
    TransformerEncoderLayer,
)

EMBED_DIM = 16
NUM_HEADS = 4
BATCH = 3
SEQ_LEN = 5


def _copy_weights(stock: nn.MultiheadAttention, custom: LoraCompatibleSelfAttention):
    with torch.no_grad():
        custom.in_proj_weight.copy_(stock.in_proj_weight)
        custom.in_proj_bias.copy_(stock.in_proj_bias)
        custom.out_proj.weight.copy_(stock.out_proj.weight)
        custom.out_proj.bias.copy_(stock.out_proj.bias)


@pytest.fixture
def attn_pair():
    stock = nn.MultiheadAttention(EMBED_DIM, NUM_HEADS, dropout=0.0, batch_first=True)
    custom = LoraCompatibleSelfAttention(
        EMBED_DIM, NUM_HEADS, dropout=0.0, batch_first=True
    )
    _copy_weights(stock, custom)
    stock.eval()
    custom.eval()
    return stock, custom


def test_self_attention_matches_stock_no_mask(attn_pair):
    stock, custom = attn_pair
    x = torch.randn(BATCH, SEQ_LEN, EMBED_DIM)

    stock_out, stock_w = stock(x, x, x, need_weights=True, average_attn_weights=False)
    custom_out, custom_w = custom(
        x, x, x, need_weights=True, average_attn_weights=False
    )

    np.testing.assert_allclose(
        custom_out.detach().numpy(), stock_out.detach().numpy(), atol=1e-5
    )
    np.testing.assert_allclose(
        custom_w.detach().numpy(), stock_w.detach().numpy(), atol=1e-5
    )


def test_self_attention_matches_stock_with_key_padding_mask(attn_pair):
    stock, custom = attn_pair
    x = torch.randn(BATCH, SEQ_LEN, EMBED_DIM)
    key_padding_mask = torch.zeros(BATCH, SEQ_LEN, dtype=torch.bool)
    key_padding_mask[:, -2:] = True  # last two positions are padding

    stock_out, stock_w = stock(
        x,
        x,
        x,
        key_padding_mask=key_padding_mask,
        need_weights=True,
        average_attn_weights=False,
    )
    custom_out, custom_w = custom(
        x,
        x,
        x,
        key_padding_mask=key_padding_mask,
        need_weights=True,
        average_attn_weights=False,
    )

    np.testing.assert_allclose(
        custom_out.detach().numpy(), stock_out.detach().numpy(), atol=1e-5
    )
    np.testing.assert_allclose(
        custom_w.detach().numpy(), stock_w.detach().numpy(), atol=1e-5
    )


def test_self_attention_matches_stock_need_weights_false(attn_pair):
    """The fused SDPA path (need_weights=False) must match the stock module's
    fast path numerically, not just the manual need_weights=True path."""
    stock, custom = attn_pair
    x = torch.randn(BATCH, SEQ_LEN, EMBED_DIM)

    stock_out, _ = stock(x, x, x, need_weights=False)
    custom_out, custom_w = custom(x, x, x, need_weights=False)

    assert custom_w is None
    np.testing.assert_allclose(
        custom_out.detach().numpy(), stock_out.detach().numpy(), atol=1e-5
    )


def test_self_attention_causal_mask_matches_stock(attn_pair):
    """is_causal is a hint, not a standalone flag: stock nn.MultiheadAttention
    raises unless an explicit causal attn_mask is also supplied."""
    stock, custom = attn_pair
    x = torch.randn(BATCH, SEQ_LEN, EMBED_DIM)
    causal_mask = nn.Transformer.generate_square_subsequent_mask(SEQ_LEN)

    stock_out, _ = stock(
        x, x, x, attn_mask=causal_mask, is_causal=True, need_weights=False
    )
    custom_out, _ = custom(
        x, x, x, attn_mask=causal_mask, is_causal=True, need_weights=False
    )

    np.testing.assert_allclose(
        custom_out.detach().numpy(), stock_out.detach().numpy(), atol=1e-5
    )


def test_self_attention_need_weights_shape_parity_in_training_mode():
    """In training mode (dropout > 0) the manual and fused paths draw from
    different RNG streams and cannot match by value -- assert shapes only."""
    custom = LoraCompatibleSelfAttention(
        EMBED_DIM, NUM_HEADS, dropout=0.1, batch_first=True
    )
    custom.train()
    x = torch.randn(BATCH, SEQ_LEN, EMBED_DIM)

    out_weighted, weights = custom(
        x, x, x, need_weights=True, average_attn_weights=False
    )
    out_fast, none_weights = custom(x, x, x, need_weights=False)

    assert out_weighted.shape == (BATCH, SEQ_LEN, EMBED_DIM)
    assert out_fast.shape == (BATCH, SEQ_LEN, EMBED_DIM)
    assert weights.shape == (BATCH, NUM_HEADS, SEQ_LEN, SEQ_LEN)
    assert none_weights is None


def test_lora_hook_fires_on_out_proj():
    """Proves the structural fix directly: a forward hook on out_proj (the
    mechanism PEFT's LoRA relies on) must actually fire during a normal
    forward pass -- independent of PEFT itself, which lives in bio-agent."""
    custom = LoraCompatibleSelfAttention(EMBED_DIM, NUM_HEADS, batch_first=True)
    calls = []
    custom.out_proj.register_forward_hook(lambda module, inp, out: calls.append(out))

    x = torch.randn(BATCH, SEQ_LEN, EMBED_DIM)
    custom(x, x, x)

    assert len(calls) == 1


def test_transformer_encoder_layer_output_attentions_shape():
    layer = TransformerEncoderLayer(
        d_model=EMBED_DIM, nhead=NUM_HEADS, dim_feedforward=32, batch_first=True
    )
    x = torch.randn(BATCH, SEQ_LEN, EMBED_DIM)

    out, attn_map = layer(x, output_attentions=True)

    assert out.shape == (BATCH, SEQ_LEN, EMBED_DIM)
    assert attn_map.shape == (BATCH, NUM_HEADS, SEQ_LEN, SEQ_LEN)


def test_transformer_encoder_collects_attn_maps_per_layer():
    num_layers = 3
    layer = TransformerEncoderLayer(
        d_model=EMBED_DIM, nhead=NUM_HEADS, dim_feedforward=32, batch_first=True
    )
    encoder = TransformerEncoder(
        layer, num_layers=num_layers, enable_nested_tensor=False
    )
    x = torch.randn(BATCH, SEQ_LEN, EMBED_DIM)

    out, attn_maps = encoder(x, output_attentions=True)

    assert out.shape == (BATCH, SEQ_LEN, EMBED_DIM)
    assert len(attn_maps) == num_layers
    for attn_map in attn_maps:
        assert attn_map.shape == (BATCH, NUM_HEADS, SEQ_LEN, SEQ_LEN)


def test_out_proj_gradient_flows_from_encoder_output():
    """Direct regression guard for helicalAI/bio-agent#1015: out_proj must be
    on the gradient path, since that's exactly what LoRA relies on."""
    layer = TransformerEncoderLayer(
        d_model=EMBED_DIM, nhead=NUM_HEADS, dim_feedforward=32, batch_first=True
    )
    x = torch.randn(BATCH, SEQ_LEN, EMBED_DIM)

    out = layer(x)
    out.sum().backward()

    assert layer.self_attn.out_proj.weight.grad is not None
    assert torch.any(layer.self_attn.out_proj.weight.grad != 0)


def test_checkpoint_key_names_match_stock_multihead_attention():
    """Guards against the silent-skip failure mode in scGPT's load_pretrained
    (strict=False, matches state-dict keys by name+shape): if this module's
    parameter names ever drift from stock MultiheadAttention's, a real
    pretrained checkpoint would silently fail to load self_attn weights."""
    stock = nn.MultiheadAttention(EMBED_DIM, NUM_HEADS, batch_first=True)
    custom = LoraCompatibleSelfAttention(EMBED_DIM, NUM_HEADS, batch_first=True)

    stock_keys = set(stock.state_dict().keys())
    custom_keys = set(custom.state_dict().keys())

    assert stock_keys == custom_keys
    for key in stock_keys:
        assert stock.state_dict()[key].shape == custom.state_dict()[key].shape
