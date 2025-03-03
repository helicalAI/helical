import pytest
import torch
from helical.models.mamba2_mrna import Mamba2mRNAConfig, Mamba2mRNAFineTuningModel


class TestHelixmRNAFineTuning:
    @pytest.fixture
    def mamba2mRNAFineTune(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        config = Mamba2mRNAConfig(batch_size=1, device=self.device, max_length=20)
        return Mamba2mRNAFineTuningModel(
            mamba2_mrna_config=config, fine_tuning_head="classification", output_size=1
        )

    @pytest.fixture
    def mock_data(self, mamba2mRNAFineTune):
        input_sequences = ["AAAA", "CCCC", "UUUU", "ACGU", "ACGN", "ANNU"]
        labels = [0, 0, 0, 0, 0, 0]
        tokenized_sequences = mamba2mRNAFineTune.process_data(input_sequences)
        return tokenized_sequences, labels

    def test_output_dimensionality_of_fine_tuned_model(
        self, mamba2mRNAFineTune, mock_data
    ):
        input_sequences, labels = mock_data
        mamba2mRNAFineTune.train(
            train_dataset=input_sequences,
            train_labels=labels,
            validation_dataset=input_sequences,
            validation_labels=labels,
        )
        outputs = mamba2mRNAFineTune.get_outputs(input_sequences)
        assert outputs.shape == (len(input_sequences), 1)
