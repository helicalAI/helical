from helical import HelixmRNA, HelixmRNAConfig
import pytest
import torch

class TestHelixmRNAModel:
    @pytest.fixture(params=["helical-ai/Helix-mRNA"])
    def helixmRNA(self, request):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        config = HelixmRNAConfig(model_name=request.param, batch_size=1, device=self.device)
        return HelixmRNA(configurer=config)

    @pytest.fixture
    def mock_data(self, helixmRNA):
        input_sequences = ["AAAA", "CCCC", "UUUU" * 30, "ACGU" * 100, "ACGN", "ANNU" * 20]
        tokenized_sequences = helixmRNA.process_data(input_sequences)
        return tokenized_sequences

    def test_invalid_sequences(self, helixmRNA):
        input_sequences = ["AAQA", "CCCZ", "FJAK" * 30, "QWER" * 100, "JFHS", "OFNW" * 20]
        with pytest.raises(ValueError, match=r"Invalid RNA sequence:*"):
            helixmRNA.process_data(input_sequences)

    def test_helix_mrna_get_embeddings(self, mock_data, helixmRNA):
        embeddings = helixmRNA.get_embeddings(mock_data)

        assert embeddings is not None, f"Embeddings should not be None for sequence: {mock_data}"