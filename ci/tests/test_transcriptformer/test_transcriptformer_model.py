from helical.models.transcriptformer.model import TranscriptFormer
from helical.models.transcriptformer.transcriptformer_config import (
    TranscriptFormerConfig,
)
from anndata import AnnData


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
