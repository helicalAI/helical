import pytest
import torch
from helical.utils.attn_backend import select_attn_backend


@pytest.fixture
def fa2_available(mocker):
    return mocker.patch(
        "helical.utils.attn_backend.is_flash_attn_2_available", return_value=True
    )


@pytest.mark.parametrize("supports_fa2", [False, True])
def test_output_attentions_forces_eager(supports_fa2, fa2_available):
    attn, dtype = select_attn_backend(
        "cuda", output_attentions=True, supports_fa2=supports_fa2
    )
    assert attn == "eager"
    assert dtype == torch.float32


def test_fa2_selected_when_supported_and_available(fa2_available):
    attn, dtype = select_attn_backend(
        "cuda", output_attentions=False, supports_fa2=True
    )
    assert attn == "flash_attention_2"
    assert dtype == torch.bfloat16


def test_sdpa_when_model_does_not_support_fa2(fa2_available):
    """Regression guard: models without HF FA2 dispatch (e.g. BertForMaskedLM) must
    never receive attn_implementation='flash_attention_2', even when flash_attn is
    installed on CUDA. Transformers raises a hard ValueError otherwise."""
    attn, _ = select_attn_backend("cuda", output_attentions=False, supports_fa2=False)
    assert attn == "sdpa"


def test_sdpa_when_fa2_unavailable(mocker):
    mocker.patch(
        "helical.utils.attn_backend.is_flash_attn_2_available", return_value=False
    )
    attn, _ = select_attn_backend("cuda", output_attentions=False, supports_fa2=True)
    assert attn == "sdpa"


def test_sdpa_on_cpu_even_with_fa2_available(fa2_available):
    attn, _ = select_attn_backend("cpu", output_attentions=False, supports_fa2=True)
    assert attn == "sdpa"
