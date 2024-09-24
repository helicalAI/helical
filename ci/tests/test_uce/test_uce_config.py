from helical.models.uce.model import UCEConfig
import pytest

@pytest.mark.parametrize("model_name, n_layers", [
    ("33l_8ep_1024t_1280", 33),
    ("4layer_model", 4)
])

def test_uce_config_valid_model_names(model_name, n_layers):
    """
    Test case for the UCE config initialization.

    Args:
        model_name (str): The name of the model.
        n_layers (int): The number of layers of the model.
    """
    configurer = UCEConfig(model_name=model_name)
    assert configurer.config["model_path"].name == f"{model_name}.torch"
    assert configurer.config["n_layers"] == n_layers

@pytest.mark.parametrize("model_name", [
    ("wrong_name")
])
def test_uce_config__invalid_model_names(model_name):
    """
    Test case when an invalid model name is provided.
    Verifies that a ValueError is raised when an invalid model name is passed to the UCEConfig constructor.

    Parameters:
    - model_name (str): The invalid model name.

    Raises:
    - ValueError: If the model name is invalid.
    """
    with pytest.raises(ValueError):
        UCEConfig(model_name=model_name)

