from helical.models.base_models import HelicalRNAModel
import logging
from pathlib import Path
import numpy as np
from anndata import AnnData
import torch
from typing import Optional, Union
from torch.utils.data import DataLoader
from helical.models.tahoe.tahoe_config import TahoeConfig
from helical.utils.mapping import map_gene_symbols_to_ensembl_ids

LOGGER = logging.getLogger(__name__)


class Tahoe(HelicalRNAModel):
    """Tahoe-1x Model.

    The Tahoe-1x Model is a transformer-based foundation model designed for single-cell
    RNA-seq data. It can extract cell and gene embeddings from raw count data.
    The model is available in three sizes:

    - 70m: 12-layer transformer with 512 embedding dimensions
    - 1b: 24-layer transformer with 1024 embedding dimensions (coming soon)
    - 3b: 36-layer transformer with 1536 embedding dimensions (coming soon)

    Example
    -------
    ```python
    from helical.models.tahoe import Tahoe, TahoeConfig
    import anndata as ad

    # Example configuration
    tahoe_config = TahoeConfig(model_size="70m", batch_size=8)
    tahoe = Tahoe(configurer=tahoe_config)

    # Load and process data - returns a DataLoader
    ann_data = ad.read_h5ad("anndata_file.h5ad")
    dataloader = tahoe.process_data(ann_data)

    # Get embeddings from the DataLoader
    embeddings = tahoe.get_embeddings(dataloader)
    print("Tahoe embeddings shape:", embeddings.shape)

    # Get both cell and gene embeddings
    cell_embeddings, gene_embeddings = tahoe.get_embeddings(dataloader, return_gene_embeddings=True)
    print("Cell embeddings shape:", cell_embeddings.shape)
    print("Gene embeddings shape:", gene_embeddings.shape)

    # Get attention weights (requires attn_impl='torch')
    tahoe_config_attn = TahoeConfig(model_size="70m", batch_size=8, attn_impl='torch')
    tahoe_attn = Tahoe(configurer=tahoe_config_attn)
    dataloader_attn = tahoe_attn.process_data(ann_data)
    cell_embeddings, attentions = tahoe_attn.get_embeddings(dataloader_attn, output_attentions=True)
    print("Attention shape:", attentions.shape)  # (n_cells, n_heads, seq_len, seq_len) - last layer only
    ```

    Parameters
    ----------
    configurer : TahoeConfig, optional
        The model configuration. Defaults to TahoeConfig() with default parameters.

    Notes
    -----
    The Tahoe-1x model uses Ensembl IDs to identify genes and currently supports only
    human genes. The model is published by Tahoe Therapeutics and available on
    Hugging Face at https://huggingface.co/tahoebio/Tahoe-x1.

    By default, the model uses Flash Attention (attn_impl='flash') for efficient inference.
    To extract attention weights, use attn_impl='torch' when creating the TahoeConfig,
    though this will be slower and use more memory.
    """

    default_configurer = TahoeConfig()

    def __init__(self, configurer: TahoeConfig = default_configurer) -> None:
        super().__init__()
        self.configurer = configurer
        self.config = configurer.config
        self.device = torch.device(self.config["device"])

        # Import tahoe_x1 modules from local copy
        from helical.models.tahoe.tahoe_x1.model import ComposerTX

        LOGGER.info(
            f"Loading Tahoe model (size: {self.config['model_size']}) from Hugging Face..."
        )

        # Load model from Hugging Face
        self.model, self.vocab, self.model_cfg, self.collator_cfg = (
            ComposerTX.from_hf(
                repo_id=self.config["hf_repo_id"],
                model_size=self.config["model_size"],
                return_gene_embeddings=(self.config["emb_mode"] == "gene"),
                attn_impl=self.config["attn_impl"],
            )
        )

        self.model.to(self.device)
        self.model.eval()

        LOGGER.info(
            f"Model loaded with {self.model.model.n_layers} transformer layers."
        )
        LOGGER.info(
            f"Tahoe model is in 'eval' mode, on device '{self.device}' with embedding mode '{self.config['emb_mode']}' "
            f"and attention implementation '{self.config['attn_impl']}'."
        )

    def process_data(
        self,
        adata: AnnData,
        gene_names: str = "index",
        use_raw_counts: bool = True,
    ) -> DataLoader:
        """
        Processes the data for the Tahoe model and returns a DataLoader.

        Parameters
        ----------
        adata : AnnData
            The AnnData object containing the data to be processed. Tahoe uses Ensembl IDs
            to identify genes and currently supports only human genes. If the AnnData object
            already has an 'ensembl_id' column, the mapping step can be skipped.
        gene_names : str, optional, default="index"
            The column in `adata.var` that contains the gene names. If set to a value other
            than "ensembl_id", the gene symbols in that column will be mapped to Ensembl IDs
            using the 'pyensembl' package.
            - If set to "index", the index of the AnnData object will be used and mapped to Ensembl IDs.
            - If set to "ensembl_id", no mapping will occur.
        use_raw_counts : bool, optional, default=True
            Determines whether raw counts should be used.

        Returns
        -------
        DataLoader
            A PyTorch DataLoader ready for inference.
        """
        LOGGER.info("Processing data for Tahoe.")
        self.ensure_rna_data_validity(adata, gene_names, use_raw_counts)

        # Map gene symbols to Ensembl IDs if provided
        if gene_names != "ensembl_id":
            if (adata.var[gene_names].str.startswith("ENS").all()) or (
                adata.var[gene_names].str.startswith("None").any()
            ):
                message = (
                    "It seems an anndata with 'ensemble ids' and/or 'None' was passed. "
                    "Please set gene_names='ensembl_id' and remove 'None's to skip mapping."
                )
                LOGGER.error(message)
                raise ValueError(message)
            adata = map_gene_symbols_to_ensembl_ids(adata, gene_names)

            if adata.var["ensembl_id"].isnull().all():
                message = "All gene symbols could not be mapped to Ensembl IDs. Please check the input data."
                LOGGER.error(message)
                raise ValueError(message)

        gene_id_key = "ensembl_id"

        # Map genes to vocabulary
        adata.var["id_in_vocab"] = [
            self.vocab[gene] if gene in self.vocab else -1
            for gene in adata.var[gene_id_key]
        ]

        gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
        n_matched = np.sum(gene_ids_in_vocab >= 0)
        n_total = len(gene_ids_in_vocab)

        LOGGER.info(
            f"Matched {n_matched}/{n_total} genes in vocabulary of size {len(self.vocab)}."
        )

        # Filter to genes in vocabulary
        adata = adata[:, adata.var["id_in_vocab"] >= 0]

        genes = adata.var[gene_id_key].tolist()
        gene_ids = np.array([self.vocab[gene] for gene in genes], dtype=int)

        if not np.all(gene_ids >= 0):
            raise ValueError("Some genes are not in the vocabulary after filtering.")

        # Create DataLoader from AnnData
        from helical.models.tahoe.tahoe_x1.utils.util import loader_from_adata

        dataloader = loader_from_adata(
            adata=adata,
            collator_cfg=self.collator_cfg,
            vocab=self.vocab,
            batch_size=self.config["batch_size"],
            max_length=self.config["max_length"],
            gene_ids=gene_ids,
            num_workers=self.config["num_workers"],
            prefetch_factor=self.config["prefetch_factor"],
        )

        LOGGER.info("Successfully processed the data for Tahoe.")
        return dataloader

    def get_embeddings(
        self,
        dataloader: DataLoader,
        return_gene_embeddings: bool = False,
        output_attentions: bool = False,
    ) -> Union[np.ndarray, tuple]:
        """Gets the embeddings from the Tahoe model.

        Parameters
        ----------
        dataloader : DataLoader
            The DataLoader returned from process_data().
        return_gene_embeddings : bool, optional, default=False
            Whether to return gene embeddings in addition to cell embeddings.
        output_attentions : bool, optional, default=False
            Whether to return attention weights from all transformer layers.
            Note: This requires the model to be initialized with attn_impl='torch'.
            The default Flash Attention (attn_impl='flash') does not support attention
            weight extraction for efficiency reasons.

        Returns
        -------
        np.ndarray or tuple
            Depending on the combination of flags:
            - If both False: cell_embeddings (n_cells, embedding_dim)
            - If return_gene_embeddings=True only: (cell_embeddings, gene_embeddings)
            - If output_attentions=True only: (cell_embeddings, attentions)
            - If both True: (cell_embeddings, gene_embeddings, attentions)

            Where attentions is a numpy array of attention weights from the last transformer layer.
            Shape: (n_cells, n_heads, seq_length, seq_length).
            Only the last transformer layer's attention is returned to conserve memory.
        """
        LOGGER.info("Extracting embeddings from Tahoe model...")

        # Check if attention extraction is requested but not supported
        if output_attentions:
            attn_impl = self.model_cfg.get("attn_config", {}).get("attn_impl", "flash")
            if attn_impl in ["flash", "triton"]:
                raise RuntimeError(
                    f"Attention weight extraction is not supported with attn_impl='{attn_impl}'. "
                    "Flash Attention is optimized for speed and memory efficiency and does not "
                    "compute/store attention weights. To extract attention weights, initialize the model "
                    "with attn_impl='torch':\n\n"
                    "    tahoe_config = TahoeConfig(model_size='70m', attn_impl='torch')\n"
                    "    tahoe = Tahoe(configurer=tahoe_config)"
                )

        from typing import List
        from tqdm.auto import tqdm

        device = self.device
        model = self.model.model
        model.return_gene_embeddings = return_gene_embeddings

        cell_embs: List[torch.Tensor] = []
        all_attentions: List[torch.Tensor] = [] if output_attentions else None

        if return_gene_embeddings:
            gene_array = torch.zeros(
                len(self.vocab),
                self.model_cfg["d_model"],
                dtype=torch.float32,
                device=device,
            )
            gene_array_counts = torch.zeros(
                len(self.vocab),
                dtype=torch.float32,
                device=device,
            )

        dtype_from_string = {
            "fp32": torch.float32,
            "amp_bf16": torch.bfloat16,
            "amp_fp16": torch.float16,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }

        with (
            torch.no_grad(),
            torch.amp.autocast(
                enabled=True,
                dtype=dtype_from_string[self.model_cfg["precision"]],
                device_type=device.type,
            ),
        ):
            pbar = tqdm(total=len(dataloader), desc="Embedding cells")

            for data_dict in dataloader:
                input_gene_ids = data_dict["gene"].to(device)
                src_key_padding_mask = ~input_gene_ids.eq(self.collator_cfg["pad_token_id"])

                output = model(
                    genes=input_gene_ids,
                    values=data_dict["expr"].to(device),
                    gen_masks=data_dict["gen_mask"].to(device),
                    key_padding_mask=src_key_padding_mask,
                    drug_ids=(
                        data_dict["drug_ids"].to(device)
                        if "drug_ids" in data_dict
                        else None
                    ),
                    skip_decoders=True,
                    output_attentions=output_attentions,
                )

                cell_embs.append(output["cell_emb"].to("cpu").to(dtype=torch.float32))

                if output_attentions:
                    # Only keep last layer attention to save memory
                    # Shape: (batch, n_heads, seq_len, seq_len)
                    # Convert to float32 for numpy compatibility
                    last_layer_attn = output["attentions"][-1].cpu().to(torch.float32)
                    all_attentions.append(last_layer_attn)

                if return_gene_embeddings:
                    gene_embs = output.get("gene_emb").to(torch.float32)
                    flat_gene_ids = input_gene_ids.view(-1)
                    flat_embeddings = gene_embs.view(-1, gene_embs.shape[-1])

                    valid = flat_gene_ids != self.collator_cfg["pad_token_id"]
                    flat_gene_ids = flat_gene_ids[valid]
                    flat_embeddings = flat_embeddings[valid].to(gene_embs.dtype)

                    gene_array.index_add_(0, flat_gene_ids, flat_embeddings)
                    gene_array_counts.index_add_(
                        0,
                        flat_gene_ids,
                        torch.ones_like(flat_gene_ids, dtype=gene_embs.dtype),
                    )

                pbar.update(1)

        # Normalize cell embeddings
        cell_array = torch.cat(cell_embs, dim=0).numpy()
        cell_array = cell_array / np.linalg.norm(
            cell_array,
            axis=1,
            keepdims=True,
        )

        # Prepare gene embeddings if requested
        if return_gene_embeddings:
            # Average gene embeddings
            gene_array = gene_array.to("cpu").to(torch.float32).numpy()
            gene_array_counts = gene_array_counts.to("cpu").to(torch.float32).numpy()
            gene_array_counts = np.expand_dims(gene_array_counts, axis=1)

            gene_array = np.divide(
                gene_array,
                gene_array_counts,
                out=np.ones_like(gene_array) * np.nan,
                where=gene_array_counts != 0,
            )

            gene2idx = self.vocab.get_stoi()
            all_gene_ids = np.array(list(gene2idx.values()))
            gene_array = gene_array[all_gene_ids, :]

        # Prepare attention arrays if requested
        if output_attentions:
            # Concatenate all batches and convert to numpy
            # Shape: (total_cells, n_heads, seq_len, seq_len)
            attention_array = torch.cat(all_attentions, dim=0).numpy()

        # Return based on requested outputs
        log_msg = f"Finished extracting embeddings. Cell shape: {cell_array.shape}"
        if return_gene_embeddings:
            log_msg += f", Gene shape: {gene_array.shape}"
        if output_attentions:
            log_msg += f", Attention shape: {attention_array.shape}"
        LOGGER.info(log_msg)

        # Return appropriate combination
        if return_gene_embeddings and output_attentions:
            return cell_array, gene_array, attention_array
        elif return_gene_embeddings:
            return cell_array, gene_array
        elif output_attentions:
            return cell_array, attention_array
        else:
            return cell_array
