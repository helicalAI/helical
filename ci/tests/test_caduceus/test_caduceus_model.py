from helical import Caduceus, CaduceusConfig
import pytest
import torch

@pytest.mark.parametrize("model_name", [
    "caduceus-ph-4L-seqlen-1k-d118", 
    "caduceus-ph-4L-seqlen-1k-d256", 
    "caduceus-ph-16L-seqlen-131k-d256", 
    "caduceus-ps-4L-seqlen-1k-d118",
    "caduceus-ps-4L-seqlen-1k-d256", 
    "caduceus-ps-16L-seqlen-131k-d256"
])
def test_caduceus__valid_model_names(model_name):
    """
    Test case for the Caduceus class initialization.

    Args:
        model_name (str): The name of the model.
    """
    CaduceusConfig(model_name=model_name, device="cuda")

@pytest.mark.parametrize("model_name", [
    ("wrong_name")
])
def test_caduceus__invalid_model_names(model_name):
    """
    Test case when an invalid model name is provided.
    Verifies that a ValueError is raised when an invalid model name is passed to the CaduceusConfig constructor.

    Parameters:
    - model_name (str): The invalid model name.

    Raises:
    - ValueError: If the model name is invalid.
    """
    with pytest.raises(ValueError):
        CaduceusConfig(model_name=model_name)

@pytest.mark.parametrize("input_sequence", [
    # Valid DNA sequences
    "A",
    "CC",
    "TTTT", 
    "ACGTN",
    "ACGT" * 256
])
def test_caduceus_process_data(input_sequence):
    """
    Test the process_data method of the Caduceus model.
    The input DNA sequence is tokenized and the output shape is verified.

    Args:
        input_sequence (str): The input DNA sequence to be processed.

    Raises:
        AssertionError: If the output shape doesn't match expected dimensions.
    """
    model = Caduceus(CaduceusConfig(model_name="caduceus-ph-4L-seqlen-1k-d118", device="cuda"))
    dataset = model.process_data([input_sequence])
    assert len(max(dataset["input_ids"], key=len)) <= model.config["input_size"]


@pytest.mark.parametrize("input_sequence,model_name", [
    ("ACTG", "caduceus-ph-4L-seqlen-1k-d118"),
    ("ACTG", "caduceus-ph-4L-seqlen-1k-d256"),
    ("ACTG", "caduceus-ph-16L-seqlen-131k-d256"),
])
def test_caduceus_ph_get_embeddings(input_sequence, model_name):
    model = Caduceus(CaduceusConfig(model_name=model_name, device="cuda"))
    dataset = model.process_data(input_sequence)
    embeddings = model.get_embeddings(dataset)
    assert embeddings.shape[1] == model.config["embedding_size"]

@pytest.mark.parametrize("input_sequence,model_name", [
    ("A", "caduceus-ps-4L-seqlen-1k-d118"),
    ("TTTT", "caduceus-ps-4L-seqlen-1k-d256"),
    ("ACGTN", "caduceus-ps-16L-seqlen-131k-d256"),
])
def test_caduceus_ps_get_embeddings(input_sequence, model_name):
    model = Caduceus(CaduceusConfig(model_name=model_name, device="cuda"))
    dataset = model.process_data(input_sequence)
    embeddings = model.get_embeddings(dataset)
    assert embeddings.shape[1] == model.config["embedding_size"]*2
