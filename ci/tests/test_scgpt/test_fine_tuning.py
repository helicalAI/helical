from helical.models.scgpt.fine_tuning_model import scGPTFineTuningModel
from helical.models.scgpt.model import scGPTConfig
from helical.models.fine_tune.fine_tuning_heads import ClassificationHead
import os

def test_save_and_load_model():

    try:
        model = scGPTFineTuningModel(scGPTConfig(), "classification", 10)
        model.save_model("./scgpt_fine_tuned_model.pt")
        model.load_model("./scgpt_fine_tuned_model.pt")
        assert model.model is not None
        assert model.vocab is not None
        assert model.model.state_dict() is not None
        assert not model.model.training, "Model should be in eval mode"
        assert not model.fine_tuning_head.training, "Fine-tuning head should be in eval mode"
        assert model.fine_tuning_head.output_size == 10, "Output size should be 10"
        assert type(model.fine_tuning_head) == ClassificationHead, "Fine-tuning head should be a ClassificationHead"
    finally:
        if os.path.exists("./scgpt_fine_tuned_model.pt"):
            os.remove("./scgpt_fine_tuned_model.pt")
