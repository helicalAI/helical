from helical.models.hyena_dna import HyenaDNAConfig, HyenaDNA
import pytest
import numpy as np
from pandas import DataFrame


@pytest.mark.parametrize(
    "model_name, d_model, d_inner",
    [
        ("hyenadna-tiny-1k-seqlen", 128, 512),
        ("hyenadna-tiny-1k-seqlen-d256", 256, 1024),
    ],
)
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


@pytest.mark.parametrize("model_name", ["wrong_name"])
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


@pytest.mark.parametrize(
    "input_sequence, expected_output",
    [
        # Valid DNA sequences
        ("", [0, 1]),
        ("A", [0, 7, 1]),
        ("CC", [0, 8, 8, 1]),
        ("TTTT", [0, 10, 10, 10, 10, 1]),
        ("ACGTN", [0, 7, 8, 9, 10, 11, 1]),
        ("ACGT" * 256, [0] + [7, 8, 9, 10] * 256 + [1]),
    ],
)
def test_hyena_dna_process_data(input_sequence, expected_output):
    model = HyenaDNA()
    output = model.process_data([input_sequence])
    expected = np.array(expected_output)
    assert np.all(np.equal(np.array(output["input_ids"][0]), expected))


@pytest.mark.parametrize(
    "input_sequence, expected_output",
    [
        (
            ["A", "CC", "TTTT", "ACGTN", "ACGT"],
            [
                [4, 4, 4, 4, 0, 7, 1],
                [4, 4, 4, 0, 8, 8, 1],
                [4, 0, 10, 10, 10, 10, 1],
                [0, 7, 8, 9, 10, 11, 1],
                [4, 0, 7, 8, 9, 10, 1],
            ],
        )
    ],
)
def test_hyena_dna_process_data_variable_length_sequences(
    input_sequence, expected_output
):
    model = HyenaDNA()
    dataset = model.process_data(input_sequence)
    assert np.all(np.equal(np.array(expected_output), np.array(dataset["input_ids"])))


@pytest.mark.parametrize(
    "data, raise_exception",
    [
        (DataFrame({"Sequence": ["ACGTN", "ACGTN"]}), False),
        (DataFrame({"invalid": ["ACGTN", "ACGTN"]}), True),
    ],
)
def test_process_data_w_dataframe_raises_appropriate_errors(data, raise_exception):
    model = HyenaDNA()
    if raise_exception:
        with pytest.raises(
            KeyError, match=r"The DataFrame must have a column named 'Sequence'."
        ):
            model.process_data(data)
    else:
        model.process_data(data)
