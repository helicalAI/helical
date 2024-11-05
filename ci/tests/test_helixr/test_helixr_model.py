from helical import HelixR, HelixRConfig
import pytest
import torch

class TestHelixRModel:
    @pytest.fixture(params=["helixR-8L"])
    def helixR(self, request):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        config = HelixRConfig(model_name=request.param, batch_size=1, device=self.device)
        return HelixR(configurer=config)

    @pytest.fixture
    def mock_data(self, helixR):
        input_sequences = ["AAAA", "CCCC", "UUUU" * 30, "ACGU" * 100, "ACGN", "ANNU" * 20]
        tokenized_sequences = helixR.process_data(input_sequences)
        return tokenized_sequences

    def test_invalid_sequences(self, helixR):
        input_sequences = ["AAQA", "CCCZ", "FJAK" * 30, "QWER" * 100, "JFHS", "OFNW" * 20]
        with pytest.raises(ValueError, match=r"Invalid RNA sequence:*"):
            helixR.process_data(input_sequences)

    def test_helixr_get_embeddings(self, mock_data, helixR):
        embeddings = helixR.get_embeddings(mock_data)

        assert embeddings is not None, f"Embeddings should not be None for sequence: {mock_data}"