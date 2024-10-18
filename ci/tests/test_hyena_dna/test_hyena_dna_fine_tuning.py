import pytest
import torch
from helical import HyenaDNAConfig, HyenaDNAFineTuningModel

class TestHyenaDNAFineTuning:
    @pytest.fixture(params=["hyenadna-tiny-1k-seqlen", "hyenadna-tiny-1k-seqlen-d256"])
    def hyenaDNAFineTune(self, request):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        config = HyenaDNAConfig(model_name=request.param, batch_size=1, device=self.device)
        return HyenaDNAFineTuningModel(hyena_config=config, fine_tuning_head="classification", output_size=1)

    @pytest.fixture
    def mock_data(self, hyenaDNAFineTune):
        input_sequences = ["AAAA", "CCCC", "TTTT", "ACGT", "ACGN", "BHIK", "ANNT"]
        labels = [0, 0, 0, 0, 0, 0, 0]
        tokenized_sequences = hyenaDNAFineTune.process_data(input_sequences)
        return tokenized_sequences, labels

    def test_output_dimensionality_of_fine_tuned_model(self, hyenaDNAFineTune, mock_data):
        input_sequences, labels = mock_data
        hyenaDNAFineTune.train(train_input_data=input_sequences, train_labels=labels, validation_input_data=input_sequences, validation_labels=labels)
        outputs = hyenaDNAFineTune.get_outputs(input_sequences)
        assert outputs.shape == (len(input_sequences), 1)