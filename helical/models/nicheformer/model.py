from helical.models.base_models import HelicalRNAModel
from helical.models.nicheformer.nicheformer_config import NicheformerConfig
from helical.utils.downloader import Downloader
from anndata import AnnData
from datasets import Dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer
import numpy as np
import torch
import logging

LOGGER = logging.getLogger(__name__)


class Nicheformer(HelicalRNAModel):
    """Nicheformer Model.

    Nicheformer is a transformer-based foundation model for single-cell and spatial
    omics data. It is pre-trained with a masked language modelling (MLM) objective
    on a large corpus of human and mouse transcriptomics data spanning dissociated
    and spatial technologies (MERFISH, CosMx, Xenium, 10x Genomics, CITE-seq,
    Smart-seq v4). The 49 M-parameter encoder (12 layers, 512 hidden dim, 16 heads)
    produces 512-dimensional cell-level embeddings that capture both intrinsic gene
    expression and spatial niche context.

    Example
    -------
    ```python
    from helical.models.nicheformer import Nicheformer, NicheformerConfig
    import anndata as ad

    model_config = NicheformerConfig(batch_size=32)
    nicheformer = Nicheformer(model_config)

    ann_data = ad.read_h5ad("spatial_data.h5ad")
    dataset = nicheformer.process_data(ann_data)
    embeddings = nicheformer.get_embeddings(dataset)
    print("Nicheformer embeddings shape:", embeddings.shape)
    ```

    Parameters
    ----------
    configurer : NicheformerConfig, optional, default=default_configurer
        The model configuration.

    Notes
    -----
    We use the weights from the <a href="https://huggingface.co/theislab/Nicheformer">Nicheformer</a>
    HuggingFace repository, which accompanies the paper:
    Schaar, A.C., Tejada-Lapuerta, A., et al.
    <a href="https://doi.org/10.1101/2024.04.15.589472">Nicheformer: a foundation model for
    single-cell and spatial omics.</a> bioRxiv (2024).

    The tokenizer reads optional ``adata.obs`` columns to prepend context tokens:

    - ``modality``: ``"dissociated"`` or ``"spatial"``
    - ``specie``: ``"human"`` / ``"Homo sapiens"`` or ``"mouse"`` / ``"Mus musculus"``
    - ``assay``: technology string, e.g. ``"10x 3' v3"``, ``"MERFISH"``

    These columns are not required; missing metadata is silently ignored.
    """

    default_configurer = NicheformerConfig()

    def __init__(self, configurer: NicheformerConfig = default_configurer) -> None:
        super().__init__()
        self.configurer = configurer
        self.config = configurer.config
        self.files_config = configurer.files_config
        self.device = self.config["device"]

        downloader = Downloader()
        for file_path, url in self.configurer.list_of_files_to_download:
            downloader.download_via_link(file_path, url)

        model_files_dir = str(self.files_config["model_files_dir"])

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_files_dir, trust_remote_code=True
        )
        self.tokenizer.name_or_path = self.config["model_name"]

        technology_mean = self.config["technology_mean"]
        if technology_mean is not None:
            self.tokenizer._load_technology_mean(technology_mean)

        self.model = AutoModelForMaskedLM.from_pretrained(
            model_files_dir, trust_remote_code=True
        )
        self.model.eval()
        self.model.to(self.device)

        LOGGER.info("Nicheformer model finished initialising.")

    def process_data(
        self,
        adata: AnnData,
        gene_names: str = "index",
        use_raw_counts: bool = True,
    ) -> Dataset:
        """Processes the data for the Nicheformer model.

        Parameters
        ----------
        adata : AnnData
            The AnnData object containing single-cell or spatial gene expression data.
            Supported species are human and mouse. The tokenizer aligns genes against
            the Nicheformer reference vocabulary automatically.
            Optional ``adata.obs`` columns ``modality``, ``specie``, and ``assay``
            are used to prepend context tokens when present.
        gene_names : str, optional, default="index"
            The column in ``adata.var`` that contains gene names. If set to
            ``"index"``, the index of ``adata.var`` is used.
        use_raw_counts : bool, optional, default=True
            Whether to validate that the expression matrix contains raw integer
            counts.

        Returns
        -------
        Dataset
            A Hugging Face ``Dataset`` with ``input_ids`` and ``attention_mask``
            columns, ready for :meth:`get_embeddings`.
        """
        LOGGER.info("Processing data for Nicheformer.")
        self.ensure_rna_data_validity(adata, gene_names, use_raw_counts)

        inputs = self.tokenizer(adata)

        dataset = Dataset.from_dict(
            {
                "input_ids": inputs["input_ids"].numpy(),
                "attention_mask": inputs["attention_mask"].numpy(),
            }
        )

        LOGGER.info("Successfully processed data for Nicheformer.")
        return dataset

    def get_embeddings(self, dataset: Dataset) -> np.ndarray:
        """Extracts cell embeddings from a processed dataset using Nicheformer.

        Embeddings are obtained by mean-pooling over the sequence dimension at
        the configured transformer layer.

        Parameters
        ----------
        dataset : Dataset
            The processed dataset returned by :meth:`process_data`.

        Returns
        -------
        np.ndarray
            Cell embeddings of shape ``(n_cells, 512)``.
        """
        LOGGER.info("Started getting embeddings for Nicheformer.")

        batch_size = self.config["batch_size"]
        layer = self.config["layer"]
        with_context = self.config["with_context"]

        dataset.set_format(type="torch")
        all_embeddings = []

        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            with torch.no_grad():
                embeddings = self.model.get_embeddings(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    layer=layer,
                    with_context=with_context,
                )

            all_embeddings.append(embeddings.cpu().numpy())

        LOGGER.info("Finished getting embeddings for Nicheformer.")
        return np.concatenate(all_embeddings, axis=0)
