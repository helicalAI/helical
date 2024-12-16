import pytest
import numpy as np
import torch

try:
    from helical import Caduceus, CaduceusConfig
    skip = False # only run tests if able to import the package
except:
    skip = True


@pytest.mark.skipif(skip, reason="No Caduceus module present")
@pytest.mark.parametrize("model_name", [
    "caduceus-ph-4L-seqlen-1k-d118", 
    "caduceus-ph-4L-seqlen-1k-d256", 
    "caduceus-ph-16L-seqlen-131k-d256", 
    "caduceus-ps-4L-seqlen-1k-d118",
    "caduceus-ps-4L-seqlen-1k-d256", 
    "caduceus-ps-16L-seqlen-131k-d256"
])
def test_caduceus_valid_model_names(model_name):
    """
    Test case for the Caduceus class initialization.
    """
    CaduceusConfig(model_name=model_name, device="cuda")

@pytest.mark.skipif(skip, reason="No Caduceus module present")
@pytest.mark.parametrize("model_name", [
    "caduceus-pq-4L-seqlen-1k-d118", 
    "caduceus-ph-9L-seqlen-1k-d256", 
])
def test_caduceus_invalid_model_names(model_name):
    """
    Test case when an invalid model name is provided.
    Verifies that a ValueError is raised when an invalid model name is passed to the CaduceusConfig constructor.
    """
    with pytest.raises(ValueError):
        CaduceusConfig(model_name=model_name)

@pytest.mark.skipif(skip, reason="No Caduceus module present")
def test_runtime_error_when_cuda_unavailable(mocker):
    """
    Test that an error is raised when CUDA is not available since the model can't run without it.
    """
    mocker.patch.object(torch.cuda, "is_available", return_value=False)
    with pytest.raises(RuntimeError):
        Caduceus(CaduceusConfig())

@pytest.mark.skipif(skip, reason="No Caduceus module present")
@pytest.mark.parametrize("input_sequence", [
    # Valid DNA sequences
    "A",
    "CC",
    "TTTT", 
    "ACGTN",
    "ACG" * 256
])
def test_caduceus_process_data(input_sequence, mocker):
    """
    Test the process_data method of the Caduceus model.
    The input DNA sequence is tokenized and the output shape is verified.
    """
    mocker.patch.object(torch.cuda, "is_available", return_value=True)
    model = Caduceus(CaduceusConfig(model_name="caduceus-ph-4L-seqlen-1k-d118", device="cuda"))
    dataset = model.process_data([input_sequence])
    expected_tokenized_sequence = [model.tokenizer.pad_token_id]*(model.config["input_size"]-len(input_sequence)-1)
    for ch in input_sequence:
        expected_tokenized_sequence.append(model.tokenizer._vocab_str_to_int[ch])
    expected_tokenized_sequence.append(model.tokenizer.sep_token_id)
    assert len(max(dataset["input_ids"], key=len)) <= model.config["input_size"]
    assert np.all(np.equal(np.array(expected_tokenized_sequence), np.array(dataset["input_ids"][0][0])))

