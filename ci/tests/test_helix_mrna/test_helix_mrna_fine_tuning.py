import pytest
import torch
from helical import HelixmRNAConfig, HelixmRNAFineTuningModel


class TestHelixmRNAFineTuning:
    @pytest.fixture
    def helixmRNAFineTune(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        config = HelixmRNAConfig(batch_size=1, device=self.device, max_length=20)
        return HelixmRNAFineTuningModel(
            helix_mrna_config=config, fine_tuning_head="classification", output_size=1
        )

    @pytest.fixture
    def mock_data(self, helixmRNAFineTune):
        input_sequences = ["AAAA", "CCCC", "UUUU", "ACGU", "ACGN", "ANNU"]
        labels = [0, 0, 0, 0, 0, 0]
        tokenized_sequences = helixmRNAFineTune.process_data(input_sequences)
        return tokenized_sequences, labels

    def test_output_dimensionality_of_fine_tuned_model(
        self, helixmRNAFineTune, mock_data
    ):
        input_sequences, labels = mock_data
        helixmRNAFineTune.train(
            train_dataset=input_sequences,
            train_labels=labels,
            validation_dataset=input_sequences,
            validation_labels=labels,
        )
        outputs = helixmRNAFineTune.get_outputs(input_sequences)
        assert outputs.shape == (len(input_sequences), 1)
