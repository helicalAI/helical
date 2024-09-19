from anndata import AnnData
from helical.models.uce.model import UCEConfig, UCE
from helical.models.uce.fine_tuning_model import UCEFineTuningModel
import pytest
import torch


class TestUCEModel:
    @pytest.fixture(params=["4layer_model"])
    def uce(self, request):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        config = UCEConfig(model_name=request.param, batch_size=10, device=self.device)
        return UCE(config)

    @pytest.fixture
    def mock_data(self):
        data = AnnData()
        data.var['gene_symbols'] = ['SAMD11', 'PLEKHN1', 'HES4']
        data.var['ensembl_id'] = ['ENSG00000187634', 'ENSG00000187583', 'ENSG00000188290']
        data.obs["cell_type"] = ["CD4 T cells"]
        data.X = [[1, 2, 5]]
        return data

    @pytest.fixture
    def fine_tune_mock_data(self):
        labels = list([0])
        return labels

    def test_fine_tune_classification(self, uce, mock_data, fine_tune_mock_data):
        pytest.skip("Test takes long to run")
        tokenized_dataset = uce.process_data(mock_data)
        fine_tuned_model = UCEFineTuningModel(uce, fine_tuning_head="classification", output_size=1)
        fine_tuned_model.fine_tune(train_input_data=tokenized_dataset, train_labels=fine_tune_mock_data)
        assert fine_tuned_model is not None
        assert isinstance(fine_tuned_model, UCEFineTuningModel)
