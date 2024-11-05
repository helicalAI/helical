import pytest
import torch
from helical import HelixRConfig, HelixRFineTuningModel

class TestHelixRFineTuning:
    @pytest.fixture(params=["helixR-8L"])
    def helixRFineTune(self, request):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        config = HelixRConfig(model_name=request.param, batch_size=1, device=self.device)
        return HelixRFineTuningModel(helixr_config=config, fine_tuning_head="classification", output_size=1)

    @pytest.fixture
    def mock_data(self, helixRFineTune):
        input_sequences = ["AAAA", "CCCC", "UUUU", "ACGU", "ACGN", "ANNU"]
        labels = [0, 0, 0, 0, 0, 0, 0]
        tokenized_sequences = helixRFineTune.process_data(input_sequences)
        return tokenized_sequences, labels

    def test_output_dimensionality_of_fine_tuned_model(self, helixRFineTune, mock_data):
        input_sequences, labels = mock_data
        helixRFineTune.train(train_dataset=input_sequences, train_labels=labels, validation_dataset=input_sequences, validation_labels=labels)
        outputs = helixRFineTune.get_outputs(input_sequences)
        assert outputs.shape == (len(input_sequences), 1)