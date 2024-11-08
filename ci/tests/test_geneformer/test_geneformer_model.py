import pytest
import torch
from helical.models.geneformer.model import Geneformer
from helical.models.geneformer.geneformer_config import GeneformerConfig
from helical.models.geneformer.geneformer_utils import get_embs, load_model
from helical.models.geneformer.fine_tuning_model import GeneformerFineTuningModel
from anndata import AnnData

class TestGeneformerModel:
    @pytest.fixture(params=["gf-12L-30M-i2048", "gf-12L-95M-i4096"])
    def geneformer(self, request):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        config = GeneformerConfig(model_name=request.param, device=self.device)
        return Geneformer(config)

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

    def test_pass_invalid_model_name(self):
        with pytest.raises(ValueError):
            geneformer_config = GeneformerConfig(model_name='InvalidName')
        

    def test_process_data_mapping_to_ensemble_ids(self, geneformer, mock_data):
        assert mock_data.var['ensembl_id'][0] == 'ENSG00000187634'
        # is the same as the above line but more verbose (linking the gene symbol to the ensembl id)
        assert mock_data.var[mock_data.var['gene_symbols'] == 'SAMD11']['ensembl_id'].values[0] == 'ENSG00000187634'
        assert mock_data.var[mock_data.var['gene_symbols'] == 'PLEKHN1']['ensembl_id'].values[0] == 'ENSG00000187583'
        assert mock_data.var[mock_data.var['gene_symbols'] == 'HES4']['ensembl_id'].values[0] == 'ENSG00000188290'

    def test_process_data_padding_and_masking_ids(self, geneformer, mock_data):
        # for this token mapping, the padding token is 0 and the mask token is 1
        geneformer.process_data(mock_data, gene_names='gene_symbols')
        assert geneformer.gene_token_dict.get("<pad>") == 0
        assert geneformer.gene_token_dict.get("<mask>") == 1

    def test_ensure_data_validity_raising_error_with_missing_ensembl_id_column(self, geneformer, mock_data):
        del mock_data.var['ensembl_id']
        with pytest.raises(KeyError):
            geneformer.ensure_rna_data_validity(mock_data, "ensembl_id")
    
    @pytest.mark.parametrize("gene_symbols, raises_error",
                             [
                                (['ENSGSAMD11', 'ENSGPLEKHN1', 'ENSGHES4'], True), # humans
                                (['ENSMUSG00000021033', 'ENSMUSG00000021033', 'ENSMUSG00000021033'], True), # mice
                                (['SAMD11', 'None', 'HES4'], True),
                                (['SAMD11', 'PLEKHN1', 'HES4'], False),
                             ]
    )
    def test_ensembl_data_is_caught(self, geneformer, mock_data, gene_symbols, raises_error):
        mock_data.var['gene_symbols'] = gene_symbols
        if raises_error:
            with pytest.raises(ValueError):
                geneformer.process_data(mock_data, "gene_symbols")
        else:
            geneformer.process_data(mock_data, "gene_symbols")

    def test_cls_mode_with_v1_model_config(self, geneformer, mock_data):
        if geneformer.config["special_token"]:
            pytest.skip("This test is only for v1 models and should thus be only executed once.")
        with pytest.raises(ValueError):
            config = GeneformerConfig(model_name="gf-12L-30M-i2048", device="cpu", emb_mode='cls')

    def test_get_embs_cell_mode(self, geneformer, mock_data):
        tokenized_dataset = geneformer.process_data(mock_data, gene_names='gene_symbols')
        model = load_model("Pretrained", geneformer.files_config["model_files_dir"], self.device)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        embs = get_embs(
            model,
            tokenized_dataset,
            emb_mode="cell",
            layer_to_quant=-1,
            pad_token_id=geneformer.pad_token_id,
            forward_batch_size=1,
            token_gene_dict=geneformer.gene_token_dict,
            device=device
        )
        assert embs.shape == (1, model.config.hidden_size)

    def test_get_embs_cls_mode(self, geneformer, mock_data):
        if not geneformer.config["special_token"]:
            pytest.skip("This test is only for models with special tokens (v2)")
        tokenized_dataset = geneformer.process_data(mock_data, gene_names='gene_symbols')
        model = load_model("Pretrained", geneformer.files_config["model_files_dir"], self.device)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        embs = get_embs(
            model,
            tokenized_dataset,
            emb_mode="cls",
            layer_to_quant=-1,
            pad_token_id=geneformer.pad_token_id,
            forward_batch_size=1,
            gene_token_dict=geneformer.gene_token_dict,
            device=device
        )
        assert embs.shape == (1, model.config.hidden_size)

    def test_get_embs_gene_mode(self, geneformer, mock_data):
        tokenized_dataset = geneformer.process_data(mock_data, gene_names='gene_symbols')
        model = load_model("Pretrained", geneformer.files_config["model_files_dir"], self.device)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        embs = get_embs(
            model,
            tokenized_dataset,
            emb_mode="gene",
            layer_to_quant=-1,
            pad_token_id=geneformer.pad_token_id,
            forward_batch_size=1,
            gene_token_dict=geneformer.gene_token_dict,
            device=device
        )
        assert embs.shape[0] == 1
        assert embs.shape[2] == model.config.hidden_size

    def test_get_embs_different_layer(self, geneformer, mock_data):
        tokenized_dataset = geneformer.process_data(mock_data, gene_names='gene_symbols')
        model = load_model("Pretrained", geneformer.files_config["model_files_dir"], self.device)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        embs_last = get_embs(
            model,
            tokenized_dataset,
            emb_mode="cell",
            layer_to_quant=-1,
            pad_token_id=geneformer.pad_token_id,
            forward_batch_size=1,
            gene_token_dict=geneformer.gene_token_dict,
            device=device
        )
        embs_first = get_embs(
            model,
            tokenized_dataset,
            emb_mode="cell",
            layer_to_quant=0,
            pad_token_id=geneformer.pad_token_id,
            forward_batch_size=1,
            gene_token_dict=geneformer.gene_token_dict,
            device=device
        )
        assert not torch.allclose(embs_last, embs_first)

    def test_get_embs_cell_mode(self, geneformer, mock_data):
        tokenized_dataset = geneformer.process_data(mock_data, gene_names='gene_symbols')
        model = load_model("Pretrained", geneformer.files_config["model_files_dir"], self.device)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        embs = get_embs(
            model,
            tokenized_dataset,
            emb_mode="cell",
            layer_to_quant=-1,
            pad_token_id=geneformer.pad_token_id,
            forward_batch_size=1,
            gene_token_dict=geneformer.gene_token_dict,
            device=device
        )
        assert embs.shape == (1, model.config.hidden_size)

    def test_cls_eos_tokens_presence(self, geneformer, mock_data):
        geneformer.process_data(mock_data, gene_names='gene_symbols')
        if geneformer.config["special_token"]:
            assert "<cls>" in geneformer.tk.gene_token_dict
            assert "<eos>" in geneformer.tk.gene_token_dict
        else:
            assert "<cls>" not in geneformer.tk.gene_token_dict
            assert "<eos>" not in geneformer.tk.gene_token_dict

    def test_model_input_size(self, geneformer):
        assert geneformer.config["input_size"] == geneformer.configurer.model_map[geneformer.config["model_name"]]['input_size']

    def test_fine_tune_classifier_returns_correct_shape(self, mock_data, fine_tune_mock_data):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        fine_tuned_model = GeneformerFineTuningModel(GeneformerConfig(device=device), fine_tuning_head="classification", output_size=1)
        tokenized_dataset = fine_tuned_model.process_data(mock_data, gene_names='gene_symbols')
        tokenized_dataset = tokenized_dataset.add_column('labels', fine_tune_mock_data)
        
        fine_tuned_model.train(train_dataset=tokenized_dataset, label='labels')
        assert fine_tuned_model is not None
        outputs = fine_tuned_model.get_outputs(tokenized_dataset)
        assert outputs.shape == (len(mock_data), len(fine_tune_mock_data))

            