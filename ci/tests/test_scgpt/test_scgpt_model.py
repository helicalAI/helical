from helical.models.scgpt import scGPT, scGPTConfig, scGPTFineTuningModel
from anndata import AnnData
import pytest
import anndata as ad
import numpy as np
from scipy.sparse import csr_matrix
import torch
import pandas as pd


class TestSCGPTModel:
    scgpt = scGPT()

    # Create a dummy AnnData object
    data = AnnData()
    data.var["gene_names"] = ["SAMD11", "PLEKHN1", "NOT_IN_VOCAB", "HES4"]
    data.obs["cell_type"] = ["CD4 T cells"]

    vocab = {"SAMD11": 0, "PLEKHN1": 1, "HES4": 2, "<pad>": 3, "<cls>": 4}
    scgpt.vocab = vocab
    scgpt.vocab_id_to_str = {value: key for key, value in scgpt.vocab.items()}

    data.X = [[1, 2, 5, 6]]

    def test_process_data(self):
        dataset = self.scgpt.process_data(self.data, gene_names="gene_names")

        assert self.scgpt.gene_names == "gene_names"
        assert self.scgpt.gene_names in self.data.var

        # asserts that all the genes in gene_names have been correctly translated
        # to the corresponding ids based on the vocabulary
        assert (self.data.var["id_in_vocab"] == [0, 1, -1, 2]).all()

        # make sure that the genes not present in the vocabulary are filtered out
        # meaning -1 is not present in the gene_ids
        assert (dataset.gene_ids == [0, 1, 2]).all()

        assert (dataset.count_matrix == [1, 2, 6]).all()

    def test_correct_handling_of_batch_ids(self):
        batch_id_array = [1]
        self.data.obs["batch_id"] = batch_id_array
        dataset = self.scgpt.process_data(
            self.data, gene_names="gene_names", use_batch_labels=True
        )
        assert (dataset.batch_ids == batch_id_array).all()

    def test_direct_assignment_of_genes_to_index(self):
        self.data.var.index = ["SAMD11", "PLEKHN1", "NOT_IN_VOCAB", "HES4"]
        self.scgpt.process_data(self.data, gene_names="index")

        # as set above, the gene column can also be direclty assigned to the index column
        assert self.scgpt.gene_names == "index"
        assert self.scgpt.gene_names in self.data.var

    dummy_data = ad.read_h5ad("ci/tests/data/cell_type_sample.h5ad")

    @pytest.mark.parametrize(
        "data, gene_names, batch_labels",
        [
            #  missing gene_names in data.var
            (AnnData(), "gene_names", False),
            #  missing batch_id in data.obs
            (dummy_data, "index", True),
        ],
    )
    def test_ensure_data_validity__key_error(self, data, gene_names, batch_labels):
        with pytest.raises(KeyError):
            self.scgpt.ensure_data_validity(data, gene_names, batch_labels)

    err_np_arr_data = ad.read_h5ad("ci/tests/data/cell_type_sample.h5ad")
    err_np_arr_data.X.dtype = float
    err_np_arr_data.X[0, 0] = 0.5

    err_csr_data = ad.read_h5ad("ci/tests/data/cell_type_sample.h5ad")
    err_csr_data.X = csr_matrix(np.random.rand(100, 5), dtype=np.float32)

    @pytest.mark.parametrize(
        "data",
        [
            (err_np_arr_data),
            (err_csr_data),
        ],
    )
    def test_ensure_data_validity__value_error(self, data):
        """The data in X must be ints. Test an error is raised for both np.ndarray and csr_matrix."""
        with pytest.raises(ValueError):
            self.scgpt.ensure_data_validity(data, "index", False)
        assert "total_counts" in data.obs

    def test_process_data_no_matching_genes(self):
        self.dummy_data.var["gene_ids"] = [1] * self.dummy_data.n_vars
        model = scGPT()

        with pytest.raises(ValueError):
            model.process_data(self.dummy_data, gene_names="gene_ids")

    np_arr_data = ad.read_h5ad("ci/tests/data/cell_type_sample.h5ad")
    csr_data = ad.read_h5ad("ci/tests/data/cell_type_sample.h5ad")
    csr_data.X = csr_matrix(np.random.poisson(1, size=(100, 5)), dtype=np.float32)

    @pytest.mark.parametrize(
        "data",
        [
            (np_arr_data),
            (csr_data),
        ],
    )
    def test_ensure_data_validity__no_error(self, data):
        """The data in X must be ints. Test no error is raised for both np.ndarray and csr_matrix."""
        self.scgpt.ensure_data_validity(data, "index", False)
        assert "total_counts" in data.obs

    def test_fine_tune_classification_returns_correct_shape(self):
        labels = list([0])
        fine_tuned_model = scGPTFineTuningModel(
            scGPTConfig(), fine_tuning_head="classification", output_size=1
        )
        tokenized_dataset = fine_tuned_model.process_data(self.data)
        fine_tuned_model.train(train_input_data=tokenized_dataset, train_labels=labels)
        assert fine_tuned_model is not None
        outputs = fine_tuned_model.get_outputs(tokenized_dataset)
        assert outputs.shape == (len(self.data), len(labels))

    @pytest.mark.parametrize("emb_mode", ["cell", "gene", "cls"])
    def test_get_embeddings_of_different_modes(self, mocker, emb_mode):
        self.scgpt.config["emb_mode"] = emb_mode
        self.scgpt.config["embsize"] = 5

        # Mock the method directly on the instance
        mocked_embeddings = torch.tensor(
            [
                [
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [5.0, 5.0, 5.0, 5.0, 5.0],
                    [1.0, 2.0, 3.0, 2.0, 1.0],
                    [6.0, 6.0, 6.0, 6.0, 6.0],
                ],
            ]
        )
        mocker.patch.object(self.scgpt.model, "_encode", return_value=mocked_embeddings)

        # mocking the normalization of embeddings makes it easier to test the output
        mocker.patch.object(
            self.scgpt, "_normalize_embeddings", side_effect=lambda x: x
        )

        dataset = self.scgpt.process_data(self.data, gene_names="gene_names")
        embeddings = self.scgpt.get_embeddings(dataset)
        if emb_mode == "gene":
            data_list = pd.Series(
                {
                    "SAMD11": np.array([5.0, 5.0, 5.0, 5.0, 5.0]),
                    "PLEKHN1": np.array([1.0, 2.0, 3.0, 2.0, 1.0]),
                    "HES4": np.array([6.0, 6.0, 6.0, 6.0, 6.0]),
                }
            )
            assert data_list.equals(embeddings[0])

        if emb_mode == "cls":
            assert (embeddings == np.array([1.0, 1.0, 1.0, 1.0, 1.0])).all()
        if emb_mode == "cell":
            # average column wise excluding first row
            expected = np.array([[4.0, 4.3333335, 4.6666665, 4.3333335, 4.0]])
            np.testing.assert_allclose(
                embeddings,
                expected,
                rtol=1e-4,  # relative tolerance
                atol=1e-4,  # absolute tolerance
            )

    @pytest.mark.parametrize("emb_mode", ["cell", "cls"])
    def test_get_embeddings_with_gene_outputs(self, mocker, emb_mode):
        self.scgpt.config["emb_mode"] = emb_mode
        self.scgpt.config["embsize"] = 5

        # Mock the method directly on the instance
        mocked_embeddings = torch.tensor(
            [
                [
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [5.0, 5.0, 5.0, 5.0, 5.0],
                    [1.0, 2.0, 3.0, 2.0, 1.0],
                    [6.0, 6.0, 6.0, 6.0, 6.0],
                ],
            ]
        )
        mocker.patch.object(self.scgpt.model, "_encode", return_value=mocked_embeddings)

        # mocking the normalization of embeddings makes it easier to test the output
        mocker.patch.object(
            self.scgpt, "_normalize_embeddings", side_effect=lambda x: x
        )

        dataset = self.scgpt.process_data(self.data, gene_names="gene_names")
        embeddings, genes = self.scgpt.get_embeddings(dataset, output_genes=True)

        if emb_mode == "cls":
            assert (embeddings == np.array([1.0, 1.0, 1.0, 1.0, 1.0])).all()
            assert len(genes) == len(embeddings)
            for gene_list in genes:
                for gene in gene_list:
                    assert gene in ["SAMD11", "PLEKHN1", "HES4"]
        if emb_mode == "cell":
            # average column wise excluding first row
            expected = np.array([[4.0, 4.3333335, 4.6666665, 4.3333335, 4.0]])
            np.testing.assert_allclose(
                embeddings,
                expected,
                rtol=1e-4,  # relative tolerance
                atol=1e-4,  # absolute tolerance
            )
            assert len(genes) == len(embeddings)
            for gene_list in genes:
                for gene in gene_list:
                    assert gene in ["SAMD11", "PLEKHN1", "HES4"]

    @pytest.mark.parametrize("emb_mode", ["cls", "cell"])
    def test_normalization_cell_and_cls(self, emb_mode):
        mocked_embeddings = np.array(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [5.0, 5.0, 5.0, 5.0, 5.0],
                [1.0, 2.0, 3.0, 2.0, 1.0],
                [6.0, 6.0, 6.0, 6.0, 6.0],
            ]
        )

        expected_normalized_embeddings = np.array(
            [
                [0.4472, 0.4472, 0.4472, 0.4472, 0.4472],
                [0.4472, 0.4472, 0.4472, 0.4472, 0.4472],
                [0.2294, 0.4588, 0.6882, 0.4588, 0.2294],
                [0.4472, 0.4472, 0.4472, 0.4472, 0.4472],
            ]
        )

        self.scgpt.config["emb_mode"] = emb_mode
        normalized_embeddings = np.around(
            self.scgpt._normalize_embeddings(mocked_embeddings), decimals=4
        )
        assert np.all(np.equal(normalized_embeddings, expected_normalized_embeddings))

    def test_normalization_of_gene(self):
        mocked_embeddings = [
            pd.Series(
                {
                    "SAMD11": np.array([5.0, 5.0, 5.0, 5.0, 5.0]),
                    "PLEKHN1": np.array([1.0, 2.0, 3.0, 2.0, 1.0]),
                    "HES4": np.array([6.0, 6.0, 6.0, 6.0, 6.0]),
                }
            )
        ]
        expected_normalized_embeddings = [
            pd.Series(
                {
                    "SAMD11": np.array([0.4472, 0.4472, 0.4472, 0.4472, 0.4472]),
                    "PLEKHN1": np.array([0.2294, 0.4588, 0.6882, 0.4588, 0.2294]),
                    "HES4": np.array([0.4472, 0.4472, 0.4472, 0.4472, 0.4472]),
                }
            )
        ]

        self.scgpt.config["emb_mode"] = "gene"
        normalized_embeddings = self.scgpt._normalize_embeddings(mocked_embeddings)

        for expected_emb, emb in zip(
            expected_normalized_embeddings, normalized_embeddings
        ):
            for true_index, index in zip(expected_emb.keys(), emb.keys()):
                assert np.all(
                    np.equal(
                        expected_emb[true_index], np.around(emb[index], decimals=4)
                    )
                )
