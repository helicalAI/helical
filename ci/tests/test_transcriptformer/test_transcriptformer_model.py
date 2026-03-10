import numpy as np
import h5py
import pytest
import torch
from helical.models.transcriptformer.model import TranscriptFormer
from helical.models.transcriptformer.transcriptformer_config import (
    TranscriptFormerConfig,
)
from anndata import AnnData


def _write_dummy_embedding_h5(path, gene_names, emb_dim=2560):
    """Write a minimal HDF5 embedding file with random embeddings."""
    with h5py.File(path, "w") as f:
        f.create_dataset("keys", data=np.array(gene_names, dtype="S"))
        arrays_group = f.create_group("arrays")
        rng = np.random.default_rng(seed=0)
        for gene in gene_names:
            arrays_group.create_dataset(gene, data=rng.random(emb_dim).astype(np.float32))


class TestTranscriptFormerModel:
    configurer = TranscriptFormerConfig(emb_mode="gene")
    transcriptformer = TranscriptFormer(configurer)

    # Create a dummy AnnData object
    data = AnnData()
    data.obs["cell_type"] = ["CD4 T cells"]
    gene_names = ["A1BG", "ZZZ3", "NOT_IN_VOCAB", "ZZEF1"]
    data.var["gene_names"] = gene_names
    data.X = [[1, 2, 5, 6]]
    data.var_names = gene_names

    def test_process_data__correct_ensembl_ids(self):
        # test that the data is correctly mapped to ensembl ids
        dataset = self.transcriptformer.process_data([self.data])
        assert len(dataset) == 1
        assert all(
            dataset.files_list[0].var["ensembl_id"].values
            == [
                "ENSG00000121410",
                "ENSG00000036549",
                None,
                "ENSG00000074755",
            ]
        )

    def test_get_embeddings__in_gene_mode(self):
        dataset = self.transcriptformer.process_data([self.data])
        embeddings = self.transcriptformer.get_embeddings(dataset)

        # every mapped ensembl id gene should be present
        # and have 2048 embeddings
        assert embeddings[0]["ENSG00000121410"].shape == (2048,)
        assert embeddings[0]["ENSG00000036549"].shape == (2048,)
        assert embeddings[0]["ENSG00000074755"].shape == (2048,)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestTranscriptFormerPretainedEmbeddingList:
    """Tests that a list of pretrained embedding paths is accepted and applied correctly."""

    GENES_FILE_1 = ["ENSG00000121410", "ENSG00000036549"]
    GENES_FILE_2 = ["ENSG00000074755", "ENSG00000078808"]

    def test_model_loads_with_list_of_pretrained_embeddings(self, tmp_path):
        path1 = str(tmp_path / "embeddings_1.h5")
        path2 = str(tmp_path / "embeddings_2.h5")
        _write_dummy_embedding_h5(path1, self.GENES_FILE_1)
        _write_dummy_embedding_h5(path2, self.GENES_FILE_2)

        configurer = TranscriptFormerConfig(
            emb_mode="gene",
            pretrained_embedding=[path1, path2],
        )
        model = TranscriptFormer(configurer)

        # All genes from both embedding files should be present in the updated vocab
        for gene in self.GENES_FILE_1 + self.GENES_FILE_2:
            assert gene in model.gene_vocab
