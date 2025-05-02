from omegaconf import OmegaConf
from typing import Literal


class TranscriptFormerConfig:
    """
    TranscriptFormerConfig constructor.

    Parameters
    ----------
        model_name: Literal["tf_sapiens", "tf_metazoa", "tf_exemplar"] = "tf_metazoa"
            The name of the model to use.
        inference_config: dict = {
            "_target_": "helical.models.transcriptformer.data.dataclasses.InferenceConfig",
            "batch_size": 8,
            "output_keys": ["embeddings"],
            "obs_keys": ["all"],
            "data_files": [None],
            "output_path": "./inference_results",
            "load_checkpoint": None,
            "pretrained_embedding": None,
            "precision": "16-mixed"
            }
            The inference configuration.
        data_config: dict = {
            "_target_": "helical.models.transcriptformer.data.dataclasses.DataConfig",
            "gene_col_name": "ensembl_id",
            "clip_counts": 30,
            "filter_to_vocabs": True,
            "filter_outliers": 0.0,
            "normalize_to_scale": 0,
            "sort_genes": False,
            "randomize_genes": False,
            "min_expressed_genes": 0
            }
            The data configuration.
    """

    def __init__(
        self,
        model_name: Literal["tf_sapiens", "tf_metazoa", "tf_exemplar"] = "tf_sapiens",
        inference_config: dict = {
            "_target_": "helical.models.transcriptformer.data.dataclasses.InferenceConfig",
            "batch_size": 8,
            "output_keys": ["embeddings"],
            "obs_keys": ["all"],
            "data_files": [None],
            "output_path": "./inference_results",
            "load_checkpoint": None,
            "pretrained_embedding": None,
            "precision": "16-mixed",
        },
        data_config: dict = {
            "_target_": "helical.models.transcriptformer.data.dataclasses.DataConfig",
            "gene_col_name": "index",
            "clip_counts": 30,
            "filter_to_vocabs": True,
            "filter_outliers": 0.0,
            "normalize_to_scale": 0,
            "sort_genes": False,
            "randomize_genes": False,
            "min_expressed_genes": 0,
        },
    ):
        self.config = OmegaConf.create(
            {
                "model": {
                    "inference_config": inference_config,
                    "data_config": data_config,
                }
            }
        )

        if model_name not in ["tf_sapiens", "tf_metazoa", "tf_exemplar"]:
            raise ValueError(
                f"Model name {model_name} not supported. Only tf_sapiens, tf_metazoa, and tf_exemplar are supported."
            )

        if model_name == "tf_sapiens":
            self.list_of_files_to_download = [
                "transcriptformer/tf_sapiens/config.json",
                "transcriptformer/tf_sapiens/model_weights.pt",
                "transcriptformer/tf_sapiens/vocabs/assay_vocab.json",
                "transcriptformer/tf_sapiens/vocabs/homo_sapiens_gene.h5",
            ]
        elif model_name == "tf_metazoa":
            self.list_of_files_to_download = [
                "transcriptformer/tf_metazoa/config.json",
                "transcriptformer/tf_metazoa/model_weights.pt",
                "transcriptformer/tf_metazoa/vocabs/assay_vocab.json",
                "transcriptformer/tf_metazoa/vocabs/drosophila_melanogaster_gene.h5",
                "transcriptformer/tf_metazoa/vocabs/lytechinus_variegatus_gene.h5",
                "transcriptformer/tf_metazoa/vocabs/plasmodium_falciparum_gene.h5",
                "transcriptformer/tf_metazoa/vocabs/xenopus_laevis_gene.h5",
                "transcriptformer/tf_metazoa/vocabs/caenorhabditis_elegans_gene.h5",
                "transcriptformer/tf_metazoa/vocabs/gallus_gallus_gene.h5",
                "transcriptformer/tf_metazoa/vocabs/mus_musculus_gene.h5",
                "transcriptformer/tf_metazoa/vocabs/saccharomyces_cerevisiae_gene.h5",
                "transcriptformer/tf_metazoa/vocabs/danio_rerio_gene.h5",
                "transcriptformer/tf_metazoa/vocabs/oryctolagus_cuniculus_gene.h5",
                "transcriptformer/tf_metazoa/vocabs/spongilla_lacustris_gene.h5",
                "transcriptformer/tf_metazoa/vocabs/homo_sapiens_gene.h5",
            ]
        elif model_name == "tf_exemplar":
            self.list_of_files_to_download = [
                "transcriptformer/tf_exemplar/config.json",
                "transcriptformer/tf_exemplar/model_weights.pt",
                "transcriptformer/tf_exemplar/vocabs/assay_vocab.json",
                "transcriptformer/tf_exemplar/vocabs/danio_rerio_gene.h5",
                "transcriptformer/tf_exemplar/vocabs/drosophila_melanogaster_gene.h5",
                "transcriptformer/tf_exemplar/vocabs/homo_sapiens_gene.h5",
                "transcriptformer/tf_exemplar/vocabs/mus_musculus_gene.h5",
                "transcriptformer/tf_exemplar/vocabs/caenorhabditis_elegans_gene.h5",
            ]

        self.model_name = model_name
