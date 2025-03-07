from helical.models.mamba2_mrna import Mamba2mRNA, Mamba2mRNAConfig
import pytest
import torch


class TestHelixmRNAModel:
    @pytest.fixture
    def mamba2mRNA(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        config = Mamba2mRNAConfig(batch_size=1, device=self.device, max_length=100)
        return Mamba2mRNA(configurer=config)

    @pytest.fixture
    def mock_data(self, mamba2mRNA):
        input_sequences = [
            "AAAA",
            "CCCC",
            "UUUU" * 30,
            "ACGU" * 100,
            "ACGN",
            "ANNU" * 20,
        ]
        tokenized_sequences = mamba2mRNA.process_data(input_sequences)
        return tokenized_sequences

    def test_invalid_sequences(self, mamba2mRNA):
        input_sequences = [
            "AAQA",
            "CCCZ",
            "FJAK" * 30,
            "QWER" * 100,
            "JFHS",
            "OFNW" * 20,
        ]
        with pytest.raises(ValueError, match=r"Invalid RNA sequence:*"):
            mamba2mRNA.process_data(input_sequences)

    def test_helix_mrna_get_embeddings(self, mock_data, mamba2mRNA):
        embeddings = mamba2mRNA.get_embeddings(mock_data)

        assert (
            embeddings is not None
        ), f"Embeddings should not be None for sequence: {mock_data}"
