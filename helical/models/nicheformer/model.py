from helical.models.base_models import HelicalRNAModel
from helical.models.nicheformer.nicheformer_config import NicheformerConfig
from helical.utils.downloader import Downloader
from helical.utils.mapping import map_gene_symbols_to_ensembl_ids
from anndata import AnnData
from datasets import Dataset
from helical.models.nicheformer.modeling_nicheformer import NicheformerForMaskedLM
from helical.models.nicheformer.tokenization_nicheformer import (
    NicheformerTokenizer,
)
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

        self.tokenizer = NicheformerTokenizer.from_pretrained(
            model_files_dir
        )

        technology_mean = self.config["technology_mean"]
        if technology_mean is not None:
            self.tokenizer._load_technology_mean(technology_mean)

        self.model = NicheformerForMaskedLM.from_pretrained(
            model_files_dir
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
            ``"index"``, the index of ``adata.var`` is used. If set to
            ``"ensembl_id"``, no symbol-to-Ensembl mapping is performed and
            ``adata.var_names`` must already be Ensembl IDs. Otherwise the
            symbols in the given column are mapped to Ensembl IDs via the
            static BioMart table.
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
        # When gene_names="ensembl_id" the IDs live in var_names (the index),
        # not in a separate column, so validate against "index".
        self.ensure_rna_data_validity(
            adata, "index" if gene_names == "ensembl_id" else gene_names, use_raw_counts
        )

        if gene_names != "ensembl_id":
            col = adata.var[gene_names]
            if not col.str.startswith("ENS").all():
                adata = map_gene_symbols_to_ensembl_ids(
                    adata, gene_names if gene_names != "index" else None
                )
                if adata.var["ensembl_id"].isnull().all():
                    message = "All gene symbols could not be mapped to Ensembl IDs. Please check the input data."
                    LOGGER.error(message)
                    raise ValueError(message)
                adata = adata[:, adata.var["ensembl_id"].notnull()]
                adata.var_names = adata.var["ensembl_id"].values

        ref_genes = {k for k in self.tokenizer.get_vocab() if not k.startswith("[")}
        _original_gene_count = len(adata.var_names)
        adata = adata[:, adata.var_names[adata.var_names.isin(ref_genes)]]
        LOGGER.info(
            f"Filtering out {_original_gene_count - adata.shape[1]} genes to a total of {adata.shape[1]} genes with an ID in the Nicheformer vocabulary."
        )

        if adata.shape[1] == 0:
            _message = "No matching genes were found in Nicheformer vocabulary"
            LOGGER.error(_message)
            raise ValueError(_message)

        inputs = self.tokenizer(adata)

        dataset = Dataset.from_dict(
            {
                "input_ids": inputs["input_ids"].numpy(),
                "attention_mask": inputs["attention_mask"].numpy(),
            }
        )

        LOGGER.info("Successfully processed data for Nicheformer.")
        return dataset

    def get_embeddings(
        self,
        dataset: Dataset,
        output_attentions: bool = False,
    ) -> np.ndarray:
        """Extracts cell embeddings from a processed dataset using Nicheformer.

        Embeddings are obtained by mean-pooling over the sequence dimension at
        the configured transformer layer.

        Parameters
        ----------
        dataset : Dataset
            The processed dataset returned by :meth:`process_data`.
        output_attentions : bool, optional, default=False
            Whether to return per-head attention weights from the target
            transformer layer. When ``True`` a second array is returned with
            shape ``(n_cells, n_heads, seq_length, seq_length)``.

        Returns
        -------
        np.ndarray
            Cell embeddings of shape ``(n_cells, 512)``.
        np.ndarray, optional
            Attention weights of shape ``(n_cells, n_heads, seq_length,
            seq_length)``, only returned when ``output_attentions=True``.
        """
        LOGGER.info("Started getting embeddings for Nicheformer.")

        batch_size = self.config["batch_size"]
        layer = self.config["layer"]
        with_context = self.config["with_context"]

        dataset.set_format(type="torch")
        all_embeddings = []
        all_attentions = [] if output_attentions else None

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

                if output_attentions:
                    attn = self._extract_attention_weights(
                        input_ids, attention_mask, layer
                    )
                    all_attentions.append(attn.cpu().numpy())

            all_embeddings.append(embeddings.cpu().numpy())

        LOGGER.info("Finished getting embeddings for Nicheformer.")
        embeddings_out = np.concatenate(all_embeddings, axis=0)
        if output_attentions:
            return embeddings_out, np.concatenate(all_attentions, axis=0)
        return embeddings_out

    def _extract_attention_weights(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        layer: int,
    ) -> torch.Tensor:
        """Return per-head attention weights from the target encoder layer.

        Runs the embedding preparation and all encoder layers up to and
        including ``layer``, extracting the self-attention weights at that
        layer via ``MultiheadAttention`` with ``need_weights=True``.

        Parameters
        ----------
        input_ids : torch.Tensor
            Token IDs of shape ``(batch, seq_len)``.
        attention_mask : torch.Tensor
            Boolean mask of shape ``(batch, seq_len)`` — ``True`` for real
            tokens.
        layer : int
            Target layer index (negative values count from the last layer).

        Returns
        -------
        torch.Tensor
            Attention weights of shape ``(batch, n_heads, seq_len, seq_len)``.
        """
        bert = self.model.bert
        layer_idx = bert.config.nlayers + layer if layer < 0 else layer

        token_embedding = bert.embeddings(input_ids)
        if bert.config.learnable_pe:
            pos_embedding = bert.positional_embedding(
                bert.pos.to(token_embedding.device)
            )
            x = bert.dropout(token_embedding + pos_embedding)
        else:
            x = bert.positional_embedding(token_embedding)

        padding_mask = ~attention_mask.bool()

        for i in range(layer_idx + 1):
            if i == layer_idx:
                x_in = x
            x = bert.encoder.layers[i](
                x, src_key_padding_mask=padding_mask, is_causal=False
            )

        enc_layer = bert.encoder.layers[layer_idx]
        query = enc_layer.norm1(x_in) if enc_layer.norm_first else x_in
        _, attn_weights = enc_layer.self_attn(
            query,
            query,
            query,
            key_padding_mask=padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        return attn_weights
