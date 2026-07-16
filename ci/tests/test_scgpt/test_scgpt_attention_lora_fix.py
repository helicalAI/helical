import torch

from helical.models.scgpt.model_dir.transformer import (
    TransformerEncoder,
    TransformerEncoderLayer,
)
from helical.utils.transformer_encoder import LoraCompatibleSelfAttention

_D_MODEL = 8
_NHEAD = 2
_NUM_LAYERS = 2
_BATCH = 2
_SEQ_LEN = 5


def _tiny_encoder() -> TransformerEncoder:
    layer = TransformerEncoderLayer(
        d_model=_D_MODEL, nhead=_NHEAD, dim_feedforward=16, batch_first=True
    )
    return TransformerEncoder(layer, num_layers=_NUM_LAYERS)


def test_scgpt_import_path_uses_lora_compatible_self_attention():
    """Regression guard for helicalAI/bio-agent#1015: scGPT's TransformerEncoder/
    TransformerEncoderLayer (imported from model_dir.transformer, now a re-export
    shim) must build self_attn from the shared, fixed module, not stock
    nn.MultiheadAttention."""
    encoder = _tiny_encoder()
    for layer in encoder.layers:
        assert isinstance(layer.self_attn, LoraCompatibleSelfAttention)


def test_out_proj_reachable_by_name_and_receives_gradient():
    encoder = _tiny_encoder()
    module_names = [name for name, _ in encoder.named_modules()]
    assert any(name.endswith("out_proj") for name in module_names)

    x = torch.randn(_BATCH, _SEQ_LEN, _D_MODEL)
    out = encoder(x)
    out.sum().backward()

    for name, module in encoder.named_modules():
        if name.endswith("out_proj"):
            assert module.weight.grad is not None
            assert torch.any(module.weight.grad != 0)
