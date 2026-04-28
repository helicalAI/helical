from helical.models.geneformer.fine_tuning_model import GeneformerFineTuningModel
from helical.models.geneformer.model import GeneformerConfig
from helical.models.fine_tune.fine_tuning_heads import ClassificationHead
import os
import tempfile
import torch

def test_save_and_load_model():

    try:
        model = GeneformerFineTuningModel(GeneformerConfig(), "classification", 10)
        model.save_model("./geneformer_fine_tuned_model.pt")
        model.load_model("./geneformer_fine_tuned_model.pt")
        assert not model.model.training, "Model should be in eval mode"
        assert not model.fine_tuning_head.training, "Fine-tuning head should be in eval mode"
        assert model.fine_tuning_head.output_size == 10, "Output size should be 10"
        assert type(model.fine_tuning_head) == ClassificationHead, "Fine-tuning head should be a ClassificationHead"
        assert model.model is not None
        assert model.model.state_dict() is not None
    finally:
        if os.path.exists("./geneformer_fine_tuned_model.pt"):
            os.remove("./geneformer_fine_tuned_model.pt")

def test_save_and_load_preserves_fine_tuning_head_weights():
    """Verify that save_model persists fine-tuning head weights and
    load_model restores them exactly."""
    config = GeneformerConfig()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.pt")

        # Create model and set recognizable head weights.
        model = GeneformerFineTuningModel(config, "classification", 10)
        with torch.no_grad():
            for param in model.fine_tuning_head.parameters():
                param.fill_(42.0)
        saved_head_state = {
            k: v.clone()
            for k, v in model.fine_tuning_head.state_dict().items()
        }

        model.save_model(path)

        # Create a fresh model to load into.
        model2 = GeneformerFineTuningModel(config, "classification", 10)
        model2.load_model(path)

        loaded_head_state = model2.fine_tuning_head.state_dict()
        for key in saved_head_state:
            assert key in loaded_head_state, f"Missing head key: {key}"
            assert torch.equal(saved_head_state[key], loaded_head_state[key]), (
                f"Head weight mismatch for {key}"
            )
