from helical.models.mamba2_mrna.fine_tuning_model import Mamba2mRNAFineTuningModel
from helical.models.mamba2_mrna.model import Mamba2mRNAConfig
from helical.models.fine_tune.fine_tuning_heads import ClassificationHead
import os

def test_save_and_load_model():

    try:
        model = Mamba2mRNAFineTuningModel(Mamba2mRNAConfig(), "classification", 10)
        model.save_model("./mamba2_mrna_fine_tuned_model.pt")
        model.load_model("./mamba2_mrna_fine_tuned_model.pt")
        assert not model.model.training, "Model should be in eval mode"
        assert not model.fine_tuning_head.training, "Fine-tuning head should be in eval mode"
        assert model.fine_tuning_head.output_size == 10, "Output size should be 10"
        assert type(model.fine_tuning_head) == ClassificationHead, "Fine-tuning head should be a ClassificationHead"
        assert model.model is not None
        assert model.model.state_dict() is not None
    finally:
        if os.path.exists("./mamba2_mrna_fine_tuned_model.pt"):
            os.remove("./mamba2_mrna_fine_tuned_model.pt")
