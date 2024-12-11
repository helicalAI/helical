from helical.models.hyena_dna.model import HyenaDNAConfig
import pytest
import torch
from helical.models.hyena_dna.model import HyenaDNA
from helical.models.hyena_dna.hyena_dna_utils import HyenaDNADataset

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
    assert configurer.config["model_path"].name == f"{model_name}.ckpt"
    assert configurer.config["d_model"] == d_model
    assert configurer.config["d_inner"] == d_inner

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

@pytest.mark.parametrize("input_sequence, expected_output", [
    # Valid DNA sequences
    ("", [0, 1]),
    ("A", [0, 7, 1]),
    ("CC", [0, 8, 8, 1]),
    # ("GGG", [0, 9, 9, 9, 1]),
    ("TTTT", [0, 10, 10, 10, 10, 1]),
    ("ACGTN", [0, 7, 8, 9, 10, 11, 1]),
    ("ACGT" * 256, [0] + [7, 8, 9, 10] * 256 + [1]),
    # Invalid sequences / sequences with uncertain 'N' nucleodites 
    ("BHIK", [0, 6, 6, 6, 6, 1]),
    ("ANNTBH", [0, 7, 11, 11, 10, 6, 6, 1]),

])
def test_hyena_dna_process_data(input_sequence, expected_output):
    """
    Test the process_data method of the HyenaDNA model.
    The input DNA sequence is tokenized and the output is compared to the expected output.

    Args:
        input_sequence (str): The input DNA sequence to be processed.
        expected_output (int): The expected output of the process_data method.

    Returns:
        None

    Raises:
        AssertionError: If the output of the process_data method does not match the expected output.
    """
    model = HyenaDNA()
    output = model.process_data([input_sequence])
    expected = torch.tensor([expected_output])
    assert torch.equal(output.sequences, expected)
