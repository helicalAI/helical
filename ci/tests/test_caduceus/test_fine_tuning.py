try:
    from helical.models.caduceus.fine_tuning_model import CaduceusFineTuningModel
    from helical.models.caduceus.model import CaduceusConfig
    caduceus_unavailable = False  # only run tests if able to import the package
except:
    caduceus_unavailable = True

from helical.models.fine_tune.fine_tuning_heads import RegressionHead
import os
import pytest

@pytest.mark.skipif(caduceus_unavailable, reason="No Caduceus module present")
def test_save_and_load_model():

    try:
        model = CaduceusFineTuningModel(CaduceusConfig(), "regression", 10)
        model.save_model("./caduceus_fine_tuned_model.pt")
        model.load_model("./caduceus_fine_tuned_model.pt")
        assert not model.model.training, "Model should be in eval mode"
        assert not model.fine_tuning_head.training, "Fine-tuning head should be in eval mode"
        assert model.fine_tuning_head.output_size == 10, "Output size should be 10"
        assert type(model.fine_tuning_head) == RegressionHead, "Fine-tuning head should be a RegressionHead"
        assert model.model is not None
        assert model.model.state_dict() is not None
    finally:
        if os.path.exists("./caduceus_fine_tuned_model.pt"):
            os.remove("./caduceus_fine_tuned_model.pt")
