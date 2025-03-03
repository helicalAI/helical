import pytest
import numpy as np
import torch

try:
    from helical.models.caduceus import Caduceus, CaduceusConfig

    caduceus_unavailable = False  # only run tests if able to import the package
except:
    caduceus_unavailable = True


@pytest.mark.skipif(caduceus_unavailable, reason="No Caduceus module present")
@pytest.mark.parametrize(
    "model_name",
    [
        "caduceus-ph-4L-seqlen-1k-d118",
        "caduceus-ph-4L-seqlen-1k-d256",
        "caduceus-ph-16L-seqlen-131k-d256",
        "caduceus-ps-4L-seqlen-1k-d118",
        "caduceus-ps-4L-seqlen-1k-d256",
        "caduceus-ps-16L-seqlen-131k-d256",
    ],
)
def test_caduceus_valid_model_names(model_name):
    """
    Test case for the Caduceus class initialization.
    """
    CaduceusConfig(model_name=model_name)


@pytest.mark.skipif(caduceus_unavailable, reason="No Caduceus module present")
@pytest.mark.parametrize(
    "model_name",
    [
        "caduceus-pq-4L-seqlen-1k-d118",
        "caduceus-ph-9L-seqlen-1k-d256",
    ],
)
def test_caduceus_invalid_model_names(model_name):
    """
    Test case when an invalid model name is provided.
    Verifies that a ValueError is raised when an invalid model name is passed to the CaduceusConfig constructor.
    """
    with pytest.raises(ValueError):
        CaduceusConfig(model_name=model_name)


@pytest.mark.skipif(caduceus_unavailable, reason="No Caduceus module present")
def test_runtime_error_when_cuda_unavailable(mocker):
    """
    Test that an error is raised when CUDA is not available since the model can't run without it.
    """
    mocker.patch.object(torch.cuda, "is_available", return_value=False)
    with pytest.raises(RuntimeError):
        Caduceus(CaduceusConfig())


@pytest.mark.skipif(caduceus_unavailable, reason="No Caduceus module present")
@pytest.mark.parametrize(
    "input_sequence, expected_sequence",
    [
        # Valid DNA sequences
        ("A", [7, 1]),
        ("CC", [8, 8, 1]),
        ("TTTT", [10, 10, 10, 10, 1]),
        ("ACGTN", [7, 8, 9, 10, 11, 1]),
        ("ACG" * 256, [7, 8, 9] * 256 + [1]),
    ],
)
def test_caduceus_process_data(input_sequence, expected_sequence, mocker):
    """
    Test the process_data method of the Caduceus model.
    The input DNA sequence is tokenized and the output shape is verified.
    """
    mocker.patch.object(torch.cuda, "is_available", return_value=True)
    model = Caduceus(CaduceusConfig(model_name="caduceus-ph-4L-seqlen-1k-d118"))
    dataset = model.process_data([input_sequence])

    assert len(max(dataset["input_ids"], key=len)) <= model.config["input_size"]
    assert np.all(
        np.equal(np.array(expected_sequence), np.array(dataset["input_ids"][0]))
    )


@pytest.mark.skipif(caduceus_unavailable, reason="No Caduceus module present")
@pytest.mark.parametrize(
    "input_sequences, expected_sequences",
    [
        (
            ["A", "CC", "TTTT", "ACGTN", "ACG"],
            [
                [4, 4, 4, 4, 7, 1],
                [4, 4, 4, 8, 8, 1],
                [4, 10, 10, 10, 10, 1],
                [7, 8, 9, 10, 11, 1],
                [4, 4, 7, 8, 9, 1],
            ],
        )
    ],
)
def test_caduceus_process_data_variable_length_sequences(
    input_sequences, expected_sequences, mocker
):
    """
    Test the process_data method of the Caduceus model.
    The input DNA sequence is tokenized and the output shape is verified.
    """
    mocker.patch.object(torch.cuda, "is_available", return_value=True)
    model = Caduceus(CaduceusConfig(model_name="caduceus-ph-4L-seqlen-1k-d118"))
    dataset = model.process_data(input_sequences)

    assert len(max(dataset["input_ids"], key=len)) <= model.config["input_size"]
    assert np.all(
        np.equal(np.array(expected_sequences), np.array(dataset["input_ids"]))
    )
