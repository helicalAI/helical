from helical.models.helix_mrna import HelixmRNA, HelixmRNAConfig
import pytest
from pandas import DataFrame
import torch


class TestHelixmRNAModel:
    @pytest.fixture
    def helixmRNA(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        config = HelixmRNAConfig(batch_size=1, device=self.device, max_length=100)
        return HelixmRNA(configurer=config)

    @pytest.fixture
    def helixmRNA_eager(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        config = HelixmRNAConfig(
            batch_size=1,
            device=self.device,
            max_length=100,
            output_attentions=True,
        )
        return HelixmRNA(configurer=config)

    @pytest.fixture
    def mock_data(self, helixmRNA):
        input_sequences = [
            "AAAA",
            "CCCC",
            "UUUU" * 30,
            "ACGU" * 100,
            "ACGN",
            "ANNU" * 20,
        ]
        tokenized_sequences = helixmRNA.process_data(input_sequences)
        return tokenized_sequences

    def test_invalid_sequences(self, helixmRNA):
        input_sequences = [
            "AAQA",
            "CCCZ",
            "FJAK" * 30,
            "QWER" * 100,
            "JFHS",
            "OFNW" * 20,
        ]
        with pytest.raises(ValueError, match=r"Invalid RNA sequence:*"):
            helixmRNA.process_data(input_sequences)

    def test_helix_mrna_get_embeddings(self, mock_data, helixmRNA):
        embeddings = helixmRNA.get_embeddings(mock_data)

        assert (
            embeddings is not None
        ), f"Embeddings should not be None for sequence: {mock_data}"

    def test_get_embeddings_output_attentions_requires_eager_config(
        self, mock_data, helixmRNA
    ):
        with pytest.raises(
            ValueError,
            match=r"output_attentions=True requires the model to be loaded with eager",
        ):
            helixmRNA.get_embeddings(mock_data, output_attentions=True)

    def test_get_embeddings_output_attentions_returns_attention_tensors(
        self, helixmRNA_eager
    ):
        sequences = ["EACUEGG", "EACUEGG"]
        dataset = helixmRNA_eager.process_data(sequences)

        result = helixmRNA_eager.get_embeddings(dataset, output_attentions=True)

        assert isinstance(result, tuple), "Expected tuple when output_attentions=True"
        assert len(result) == 2
        embeddings, attentions = result
        assert isinstance(attentions, list)
        assert len(attentions) >= 1
        for attn in attentions:
            assert attn.ndim == 4, (
                f"Expected attention shape (batch, heads, seq, seq); "
                f"got shape {attn.shape}"
            )

    @pytest.mark.parametrize(
        "data, raise_exception",
        [
            (
                DataFrame(
                    {
                        "Sequence": [
                            "EACU" * 20,
                            "EAUG" * 20,
                            "EAUG" * 20,
                            "EACU" * 20,
                            "EAUU" * 20,
                        ]
                    }
                ),
                False,
            ),
            (
                DataFrame(
                    {
                        "invalid": [
                            "EACU" * 20,
                            "EAUG" * 20,
                            "EAUG" * 20,
                            "EACU" * 20,
                            "EAUU" * 20,
                        ]
                    }
                ),
                True,
            ),
        ],
    )
    def test_process_data_w_dataframe_raises_appropriate_errors(
        self, helixmRNA, data, raise_exception
    ):
        if raise_exception:
            with pytest.raises(
                KeyError, match=r"The DataFrame must have a column named 'Sequence'."
            ):
                helixmRNA.process_data(data)
        else:
            helixmRNA.process_data(data)
