from helical.models.hyena_dna.model import HyenaDNA,HyenaDNAConfig
import pytest

@pytest.mark.parametrize("model_name, d_model, d_inner", [
    ("hyenadna-tiny-1k-seqlen", 128, 512),
    ("hyenadna-tiny-1k-seqlen-d256", 256, 1024)
])
def test_hyena_dna__valid_model_names(model_name, d_model, d_inner):
    """
    Test case for the HyenaDNA class initialization.

    Args:
        model_name (str): The name of the model.
        d_model (int): The dimensionality of the model.
        d_inner (int): The dimensionality of the inner layers.
    """
    configurer = HyenaDNAConfig(model_name=model_name)
    model = HyenaDNA(configurer=configurer)
    assert model.config["model_name"] == model_name
    assert model.config["d_model"] == d_model
    assert model.config["d_inner"] == d_inner

@pytest.mark.parametrize("model_name", [
    ("wrong_name")
])
def test_hyena_dna__invalid_model_names(model_name):
    """
    Test case when an invalid model name is provided.
    Verifies that a ValueError is raised when an invalid model name is passed to the HyenaDNAConfig constructor.

    Parameters:
    - model_name (str): The invalid model name.

    Raises:
    - ValueError: If the model name is invalid.
    """
    with pytest.raises(ValueError):
        HyenaDNAConfig(model_name=model_name)
