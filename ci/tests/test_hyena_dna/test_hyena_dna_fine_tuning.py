import pytest
import torch
from helical.models.hyena_dna import HyenaDNAConfig, HyenaDNAFineTuningModel
from helical.models.fine_tune.fine_tuning_heads import ClassificationHead
import os

class TestHyenaDNAFineTuning:
    @pytest.fixture(params=["hyenadna-tiny-1k-seqlen", "hyenadna-tiny-1k-seqlen-d256"])
    def hyenaDNAFineTune(self, request):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        config = HyenaDNAConfig(
            model_name=request.param, batch_size=1, device=self.device
        )
        return HyenaDNAFineTuningModel(
            hyena_config=config, fine_tuning_head="classification", output_size=1
        )

    @pytest.fixture
    def mock_data(self, hyenaDNAFineTune):
        input_sequences = ["AAAA", "CCCC", "TTTT", "ACGT", "ACGN", "ANNT"]
        labels = [0, 0, 0, 0, 0, 0]
        tokenized_sequences = hyenaDNAFineTune.process_data(input_sequences)
        return tokenized_sequences, labels

    def test_output_dimensionality_of_fine_tuned_model(
        self, hyenaDNAFineTune, mock_data
    ):
        input_sequences, labels = mock_data
        hyenaDNAFineTune.train(
            train_dataset=input_sequences,
            train_labels=labels,
            validation_dataset=input_sequences,
            validation_labels=labels,
        )
        outputs = hyenaDNAFineTune.get_outputs(input_sequences)
        assert outputs.shape == (len(input_sequences), 1)

    def test_save_and_load_model(self, hyenaDNAFineTune):

        try:
            hyenaDNAFineTune.save_model("./hyena_dna_fine_tuned_model.pt")
            hyenaDNAFineTune.load_model("./hyena_dna_fine_tuned_model.pt")
            assert not hyenaDNAFineTune.model.training, "Model should be in eval mode"
            assert not hyenaDNAFineTune.fine_tuning_head.training, "Fine-tuning head should be in eval mode"
            assert hyenaDNAFineTune.model is not None
            assert hyenaDNAFineTune.fine_tuning_head.output_size == 1, "Output size should be 1"
            assert type(hyenaDNAFineTune.fine_tuning_head) == ClassificationHead, "Fine-tuning head should be a ClassificationHead"
            assert hyenaDNAFineTune.model.state_dict() is not None
        finally:
            if os.path.exists("./hyena_dna_fine_tuned_model.pt"):
                os.remove("./hyena_dna_fine_tuned_model.pt")