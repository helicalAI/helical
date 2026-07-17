import torch

from helical.models.nicheformer.configuration_nicheformer import NicheformerConfig
from helical.models.nicheformer.modeling_nicheformer import NicheformerModel
from helical.utils.transformer_encoder import LoraCompatibleSelfAttention

_DIM = 8
_NHEADS = 2
_NLAYERS = 2
_N_TOKENS = 20
_CONTEXT_LEN = 10
_BATCH = 2
# NicheformerModel's learnable positional embedding is a fixed-size buffer of
# length context_length, added unconditionally -- inputs must match it exactly.
_SEQ_LEN = _CONTEXT_LEN


def _tiny_model() -> NicheformerModel:
    config = NicheformerConfig(
        dim_model=_DIM,
        nheads=_NHEADS,
        dim_feedforward=16,
        nlayers=_NLAYERS,
        n_tokens=_N_TOKENS,
        context_length=_CONTEXT_LEN,
    )
    return NicheformerModel(config)


def test_encoder_uses_lora_compatible_self_attention():
    """Regression guard for helicalAI/bio-agent#1015: NicheformerModel must build
    its encoder from the shared, fixed TransformerEncoderLayer, not stock
    nn.TransformerEncoderLayer/nn.MultiheadAttention."""
    model = _tiny_model()

    for layer in model.encoder.layers:
        assert isinstance(layer.self_attn, LoraCompatibleSelfAttention)


def test_out_proj_reachable_by_name_and_receives_gradient():
    model = _tiny_model()
    # NicheformerModel also keeps an unused prototype self.encoder_layer attribute
    # (the un-cloned template handed to TransformerEncoder's _get_clones) -- it's
    # never part of forward() and never receives gradient, by construction, both
    # before and after this fix. Scope the check to the real, forward-participating
    # stack under encoder.layers.
    module_names = [name for name, _ in model.encoder.named_modules()]
    assert any(name.endswith("out_proj") for name in module_names)

    input_ids = torch.randint(0, _N_TOKENS, (_BATCH, _SEQ_LEN))
    attention_mask = torch.ones(_BATCH, _SEQ_LEN, dtype=torch.bool)

    out = model(input_ids, attention_mask=attention_mask)
    out.sum().backward()

    for name, module in model.encoder.named_modules():
        if name.endswith("out_proj"):
            assert module.weight.grad is not None
            assert torch.any(module.weight.grad != 0)


def test_extract_attention_weights_call_pattern_still_works():
    """Nicheformer's wrapper (model.py::Nicheformer._extract_attention_weights)
    reaches directly into encoder.layers[i].self_attn/.norm1/.norm_first without
    going through NicheformerModel.forward -- confirm that access pattern still
    works unchanged against the fixed encoder layer."""
    model = _tiny_model()
    input_ids = torch.randint(0, _N_TOKENS, (_BATCH, _SEQ_LEN))
    attention_mask = torch.ones(_BATCH, _SEQ_LEN, dtype=torch.bool)
    padding_mask = ~attention_mask.bool()

    token_embedding = model.embeddings(input_ids)
    pos_embedding = model.positional_embedding(model.pos.to(token_embedding.device))
    x = model.dropout(token_embedding + pos_embedding)

    enc_layer = model.encoder.layers[0]
    query = enc_layer.norm1(x) if enc_layer.norm_first else x
    output, attn_weights = enc_layer.self_attn(
        query,
        query,
        query,
        key_padding_mask=padding_mask,
        need_weights=True,
        average_attn_weights=False,
    )

    assert output.shape == (_BATCH, _SEQ_LEN, _DIM)
    assert attn_weights.shape == (_BATCH, _NHEADS, _SEQ_LEN, _SEQ_LEN)
