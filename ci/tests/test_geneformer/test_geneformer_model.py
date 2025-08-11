import pytest
from helical.models.geneformer import (
    GeneformerConfig,
    Geneformer,
    GeneformerFineTuningModel,
)
from anndata import AnnData
import torch
import pandas as pd
import numpy as np


class TestGeneformer:
    @pytest.fixture
    def mock_data(self):
        data = AnnData()
        data.var["gene_symbols"] = ["HES4", "PLEKHN1", "SAMD11"]
        data.obs["cell_type"] = ["CD4 T cells"]
        data.X = [[1, 2, 5]]
        return data

    @pytest.fixture
    def mock_embeddings_v1(self, mocker):
        embs = mocker.Mock()
        embs.hidden_states = [
            torch.tensor(
                [
                    [
                        [5.0, 5.0, 5.0, 5.0, 5.0],
                        [1.0, 2.0, 3.0, 2.0, 1.0],
                        [6.0, 6.0, 6.0, 6.0, 6.0],
                    ]
                ]
            )
        ] * 12
        return embs

    @pytest.fixture
    def mock_embeddings_v2(self, mocker):
        embs = mocker.Mock()
        embs.hidden_states = torch.tensor(
            [
                [
                    [6.0, 5.0, 7.0, 5.0, 5.0],
                    [5.0, 5.0, 5.0, 5.0, 5.0],
                    [1.0, 2.0, 3.0, 2.0, 1.0],
                    [6.0, 6.0, 6.0, 6.0, 6.0],
                    [6.0, 6.0, 1.0, 6.0, 2.0],
                ]
            ]
        ).repeat(12, 1, 1, 1)
        return embs

    @pytest.fixture(params=["gf-12L-40M-i2048", "gf-12L-38M-i4096"])
    def geneformer(self, request):
        config = GeneformerConfig(model_name=request.param, batch_size=5)
        geneformer = Geneformer(config)
        return geneformer

    def test_process_data_mapping_to_ensemble_ids(self, geneformer, mock_data):
        # geneformer modifies the anndata in place and maps the gene names to ensembl id
        geneformer.process_data(mock_data, gene_names="gene_symbols")
        assert mock_data.var["ensembl_id"][0] == "ENSG00000188290"
        # is the same as the above line but more verbose (linking the gene symbol to the ensembl id)
        assert (
            mock_data.var[mock_data.var["gene_symbols"] == "SAMD11"][
                "ensembl_id"
            ].values[0]
            == "ENSG00000187634"
        )
        assert (
            mock_data.var[mock_data.var["gene_symbols"] == "PLEKHN1"][
                "ensembl_id"
            ].values[0]
            == "ENSG00000187583"
        )
        assert (
            mock_data.var[mock_data.var["gene_symbols"] == "HES4"]["ensembl_id"].values[
                0
            ]
            == "ENSG00000188290"
        )

    def test_process_data_mapping_to_ensemble_ids_resulting_in_0_genes(
        self, geneformer, mock_data
    ):
        # provide a gene that does not exist in the ensembl database
        mock_data.var["gene_symbols"] = ["1", "2", "3"]
        with pytest.raises(ValueError):
            geneformer.process_data(mock_data, gene_names="gene_symbols")

    @pytest.mark.parametrize(
        "invalid_model_names", ["gf-12L-35M-i2048", "gf-34L-30M-i5000"]
    )
    def test_pass_invalid_model_name(self, invalid_model_names):
        with pytest.raises(ValueError):
            GeneformerConfig(model_name=invalid_model_names)

    def test_ensure_data_validity_raising_error_with_missing_ensembl_id_column(
        self, geneformer, mock_data
    ):
        geneformer.process_data(mock_data, gene_names="gene_symbols")
        del mock_data.var["ensembl_id"]
        with pytest.raises(KeyError):
            geneformer.ensure_rna_data_validity(mock_data, "ensembl_id")

    @pytest.mark.parametrize(
        "gene_symbols, raises_error",
        [
            (["ENSGSAMD11", "ENSGPLEKHN1", "ENSGHES4"], True),  # humans
            (
                ["ENSMUSG00000021033", "ENSMUSG00000021033", "ENSMUSG00000021033"],
                True,
            ),  # mice
            (["SAMD11", "None", "HES4"], True),
            (["SAMD11", "PLEKHN1", "HES4"], False),
        ],
    )
    def test_ensembl_data_is_caught(
        self, geneformer, mock_data, gene_symbols, raises_error
    ):
        mock_data.var["gene_symbols"] = gene_symbols
        if raises_error:
            with pytest.raises(ValueError):
                geneformer.process_data(mock_data, "gene_symbols")
        else:
            geneformer.process_data(mock_data, "gene_symbols")

    def test_cls_mode_with_v1_model_config(self, geneformer):
        if geneformer.config["special_token"]:
            pytest.skip(
                "This test is only for v1 models and should thus be only executed once."
            )
        with pytest.raises(ValueError):
            GeneformerConfig(model_name="gf-12L-40M-i2048", emb_mode="cls")

    @pytest.mark.parametrize("emb_mode", ["cell", "gene"])
    def test_get_embeddings_of_different_modes_v1(
        self, emb_mode, mock_data, mock_embeddings_v1, mocker
    ):
        config = GeneformerConfig(
            model_name="gf-12L-40M-i2048", batch_size=5, emb_mode=emb_mode
        )
        geneformer = Geneformer(config)
        mocker.patch.object(
            geneformer.model, "forward", return_value=mock_embeddings_v1
        )

        dataset = geneformer.process_data(mock_data, gene_names="gene_symbols")
        embeddings = geneformer.get_embeddings(dataset)
        if emb_mode == "gene":
            data_list = pd.Series(
                {
                    "ENSG00000187583": np.array([1.0, 2.0, 3.0, 2.0, 1.0]),
                    "ENSG00000187634": np.array([5.0, 5.0, 5.0, 5.0, 5.0]),
                    "ENSG00000188290": np.array([6.0, 6.0, 6.0, 6.0, 6.0]),
                }
            )
            for key in data_list.index:
                assert np.all(np.equal(embeddings[0][key], data_list[key]))

        if emb_mode == "cell":
            expected = np.array([[4, 4.333333, 4.666667, 4.333333, 4]])
            np.testing.assert_allclose(embeddings, expected, rtol=1e-4, atol=1e-4)

    def test_get_embeddings_with_output_genes(
        self, mock_data, mock_embeddings_v1, mocker
    ):
        config = GeneformerConfig(
            model_name="gf-12L-40M-i2048", batch_size=5, emb_mode="cell"
        )
        geneformer = Geneformer(config)
        mocker.patch.object(
            geneformer.model, "forward", return_value=mock_embeddings_v1
        )

        dataset = geneformer.process_data(mock_data, gene_names="gene_symbols")
        embeddings, genes = geneformer.get_embeddings(dataset, output_genes=True)

        expected = np.array([[4, 4.333333, 4.666667, 4.333333, 4]])
        np.testing.assert_allclose(embeddings, expected, rtol=1e-4, atol=1e-4)
        for gene_list in genes:
            assert len(gene_list) == 3
            assert "ENSG00000187583" in gene_list
            assert "ENSG00000187634" in gene_list
            assert "ENSG00000188290" in gene_list

    @pytest.mark.parametrize("emb_mode", ["cell", "gene", "cls"])
    def test_get_embeddings_of_different_modes_v2(
        self, emb_mode, mock_data, mock_embeddings_v2, mocker
    ):
        config = GeneformerConfig(
            model_name="gf-12L-38M-i4096", batch_size=5, emb_mode=emb_mode
        )
        geneformer = Geneformer(config)
        mocker.patch.object(
            geneformer.model, "forward", return_value=mock_embeddings_v2
        )

        dataset = geneformer.process_data(mock_data, gene_names="gene_symbols")
        embeddings = geneformer.get_embeddings(dataset)
        if emb_mode == "gene":
            data_list = pd.Series(
                {
                    "ENSG00000187583": np.array([1.0, 2.0, 3.0, 2.0, 1.0]),
                    "ENSG00000187634": np.array([5.0, 5.0, 5.0, 5.0, 5.0]),
                    "ENSG00000188290": np.array([6.0, 6.0, 6.0, 6.0, 6.0]),
                }
            )
            for key in data_list.index:
                assert np.all(np.equal(embeddings[0][key], data_list[key]))

        if emb_mode == "cls":
            assert (embeddings == np.array([6.0, 5.0, 7.0, 5.0, 5.0])).all()
        if emb_mode == "cell":
            expected = np.array([[4, 4.333333, 4.666667, 4.333333, 4]])
            np.testing.assert_allclose(embeddings, expected, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("emb_mode", ["cell", "gene"])
    def test_fine_tune_classifier_returns_correct_shape(self, emb_mode, mock_data):
        fine_tuned_model = GeneformerFineTuningModel(
            GeneformerConfig(emb_mode=emb_mode),
            fine_tuning_head="classification",
            output_size=1,
        )
        tokenized_dataset = fine_tuned_model.process_data(
            mock_data, gene_names="gene_symbols"
        )
        tokenized_dataset = tokenized_dataset.add_column("labels", list([0]))

        fine_tuned_model.train(train_dataset=tokenized_dataset, label="labels")

        outputs = fine_tuned_model.get_outputs(tokenized_dataset)
        assert outputs.shape == (len(mock_data), 1)

    def test_fine_tune_classifier_cls_returns_correct_shape(self, mock_data):
        fine_tuned_model = GeneformerFineTuningModel(
            GeneformerConfig(model_name="gf-12L-38M-i4096", emb_mode="cls"),
            fine_tuning_head="classification",
            output_size=1,
        )
        tokenized_dataset = fine_tuned_model.process_data(
            mock_data, gene_names="gene_symbols"
        )
        tokenized_dataset = tokenized_dataset.add_column("labels", [0])

        fine_tuned_model.train(train_dataset=tokenized_dataset, label="labels")

        outputs = fine_tuned_model.get_outputs(tokenized_dataset)
        assert outputs.shape == (len(mock_data), 1)

    @pytest.mark.parametrize(
        "model_name,emb_layer,expected_error",
        [
            ("gf-6L-10M-i2048", -1, "No Error"),
            ("gf-6L-10M-i2048", 7, "Error"),
            ("gf-12L-40M-i2048", 6, "No Error"),
            ("gf-20L-151M-i4096", 23, "Error"),
        ],
    )
    def test_embedding_layer_error(self, model_name, emb_layer, expected_error):
        config = GeneformerConfig(
            model_name=model_name, batch_size=5, emb_layer=emb_layer
        )

        if expected_error == "Error":
            with pytest.raises(ValueError):
                Geneformer(config)
        else:
            try:
                Geneformer(config)
            except Exception as error:
                pytest.fail(f"Unexpected error: {error}")

    @pytest.mark.parametrize(
        "model_name,emb_layer",
        [
            ("gf-6L-10M-i2048", -1),
            ("gf-6L-10M-i2048", 3),
            ("gf-12L-40M-i2048", 6),
            ("gf-20L-151M-i4096", -1),
        ],
    )
    def test_layer_to_quant(self, model_name, emb_layer):
        config = GeneformerConfig(
            model_name=model_name, batch_size=5, emb_layer=emb_layer
        )
        geneformer = Geneformer(config)

        assert geneformer.layer_to_quant == emb_layer

    @pytest.mark.parametrize(
        "old_model_name,new_model_name",
        [
            ("gf-6L-30M-i2048", "gf-6L-10M-i2048"),
            ("gf-12L-30M-i2048", "gf-12L-40M-i2048"),
            ("gf-12L-95M-i4096", "gf-12L-38M-i4096"),
            ("gf-12L-95M-i4096-CLcancer", "gf-12L-38M-i4096-CLcancer"),
            ("gf-20L-95M-i4096", "gf-20L-151M-i4096"),
        ],
    )
    def test_model_name_mapping(self, old_model_name, new_model_name):
        config = GeneformerConfig(model_name=old_model_name)

        assert config.config["model_name"] == new_model_name
