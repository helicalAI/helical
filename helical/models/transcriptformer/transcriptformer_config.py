from omegaconf import OmegaConf
from typing import Literal, List


class TranscriptFormerConfig:
    """
    TranscriptFormerConfig constructor.

    Parameters
    ----------
        model_name: Literal["tf_sapiens", "tf_metazoa", "tf_exemplar"] = "tf_metazoa"
            The name of the model to use.
        batch_size: int = 8
            The number of samples to process in each batch.
        emb_mode: Literal["gene", "cell"] = "cell"
            The mode to use for the embeddings.
        output_keys: List[Literal["gene_llh", "llh"]] = ["gene_llh"]
            The keys to output.
        obs_keys: List[str] = ["all"]
            The keys to include in the output.
        data_files: List[str] = [None]
            Path to input AnnData file(s)
        output_path: str = "./inference_results"
            Directory where results will be saved
        load_checkpoint: str = None
            Path to model weights file (automatically set by inference.py)
        pretrained_embedding: str = None
            Path to pretrained embeddings for out-of-distribution species
        gene_col_name: str = "ensembl_id"
            Column name in AnnData.var containing gene names which will be mapped to ensembl ids. If index is set, .var_names will be used.
        clip_counts: int = 30
            Maximum count value (higher values will be clipped)
        filter_to_vocabs: bool = True
            Whether to filter genes to only those in the vocabulary
        filter_outliers: float = 0.0
            Standard deviation threshold for filtering outlier cells (0.0 = no filtering)
        normalize_to_scale: float = 0
            Scale factor for count normalization (0 = no normalization)
        sort_genes: bool = False
            Whether to sort the genes.
        randomize_genes: bool = False
            Whether to randomize the genes.
        min_expressed_genes: int = 0
            Minimum number of expressed genes required per cell

    """

    def __init__(
        self,
        model_name: Literal["tf_sapiens", "tf_metazoa", "tf_exemplar"] = "tf_sapiens",
        batch_size: int = 8,
        emb_mode: Literal["gene", "cell"] = "cell",
        output_keys: List[Literal["gene_llh", "llh"]] = [
            "llh",
        ],
        obs_keys: List[str] = ["all"],
        data_files: List[str] = [None],
        output_path: str = "./inference_results",
        load_checkpoint: str = None,
        pretrained_embedding: str = None,
        gene_col_name: str = "index",
        clip_counts: int = 30,
        filter_to_vocabs: bool = True,
        filter_outliers: float = 0.0,
        normalize_to_scale: float = 0,
        sort_genes: bool = False,
        randomize_genes: bool = False,
        min_expressed_genes: int = 0,
    ):

        inference_config: dict = {
            "batch_size": batch_size,
            "output_keys": output_keys,
            "obs_keys": obs_keys,
            "data_files": data_files,
            "output_path": output_path,
            "load_checkpoint": load_checkpoint,
            "device": "cuda",
            "pretrained_embedding": pretrained_embedding,
            "emb_mode": emb_mode,
        }

        data_config: dict = {
            "gene_col_name": gene_col_name,
            "clip_counts": clip_counts,
            "filter_to_vocabs": filter_to_vocabs,
            "filter_outliers": filter_outliers,
            "normalize_to_scale": normalize_to_scale,
            "sort_genes": sort_genes,
            "randomize_genes": randomize_genes,
            "min_expressed_genes": min_expressed_genes,
        }

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
