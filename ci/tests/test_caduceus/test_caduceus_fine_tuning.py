import pytest
import torch
from helical import CaduceusConfig, CaduceusFineTuningModel

class TestCaduceusFineTuning:
    @pytest.fixture(params=["caduceus-ph-4L-seqlen-1k-d118", "caduceus-ps-4L-seqlen-1k-d118"])
    def caduceusFineTune(self, request):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        config = CaduceusConfig(model_name=request.param, batch_size=1, device=self.device)
        return CaduceusFineTuningModel(caduceus_config=config, fine_tuning_head="classification", output_size=1)

    @pytest.fixture
    def mock_data(self, caduceusFineTune):
        input_sequences = ["AAAA", "CCCC", "TTTT", "ACGT", "ACGN"]
        labels = [0, 0, 0, 0, 0]
        tokenized_sequences = caduceusFineTune.process_data(input_sequences)
        return tokenized_sequences, labels

    def test_output_dimensionality_of_fine_tuned_model(self, caduceusFineTune, mock_data):
        input_sequences, labels = mock_data
        caduceusFineTune.train(train_dataset=input_sequences, train_labels=labels, validation_dataset=input_sequences, validation_labels=labels)
        outputs = caduceusFineTune.get_outputs(input_sequences)
        assert outputs.shape == (len(input_sequences), 1)