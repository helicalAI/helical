import os
import logging
import torch
import anndata
import h5py as h5
import numpy as np
from typing import Optional

from tqdm import tqdm
from torch import nn

from .model_dir.embed_utils import StateEmbeddingModel
from .model_dir.embed_utils import create_dataloader
from .model_dir.embed_utils.utils import get_embedding_cfg, get_precision_config

from helical.models.base_models import HelicalBaseFoundationModel
from helical.models.state.state_config import StateConfig
from helical.utils.downloader import Downloader
from omegaconf import OmegaConf
from scipy.sparse import csr_matrix, issparse

LOGGER = logging.getLogger(__name__)

class StateEmbed(HelicalBaseFoundationModel):
    def __init__(self, configurer: StateConfig = None) -> None:
        super().__init__()

        self.model = None
        self.collator = None

        if configurer is None:
            configurer = StateConfig()
        self.config = configurer.config

        downloader = Downloader()
        for file in self.config["embed_files_to_download"]:
            downloader.download_via_name(file)

        self.model_dir = self.config["embed_dir"]
        self.ckpt_path = os.path.join(self.model_dir, "se600m_model_weights.pt")
        
        if not os.path.exists(self.ckpt_path):
            raise FileNotFoundError(f"Model checkpoint not found at {self.ckpt_path}")

        LOGGER.info(f"Using model checkpoint: {self.ckpt_path}")

        embedding_file = os.path.join(self.model_dir, "protein_embeddings.pt")
        
        self.protein_embeds = (
            torch.load(embedding_file, weights_only=False, map_location="cpu")
            if os.path.exists(embedding_file)
            else None
        )

        self.model_conf = OmegaConf.load(os.path.join(self.model_dir, "config.yaml"))
        self.batch_size = self.config["batch_size"]
        self.load_model()

    def load_model(self):
        if self.model:
            raise ValueError("Model already initialized")

        # First, initialize the model with the config
        self.model = StateEmbeddingModel(
            token_dim=self.model_conf.tokenizer.token_dim,  
            d_model=self.model_conf.model.emsize,
            nhead=self.model_conf.model.nhead,
            d_hid=self.model_conf.model.d_hid,
            nlayers=self.model_conf.model.nlayers,
            output_dim=self.model_conf.model.output_dim,
            compiled=self.model_conf.experiment.compiled,
            cfg=self.model_conf,
        )

        # LOGGER.info("number of free parameters: %s", sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        loaded_weights = torch.load(self.ckpt_path, weights_only=False)

        # missing_keys are keys that are in the model but NOT in the loaded weights, so they would not be initialized for inference properly.
        missing_keys, _ = self.model.load_state_dict(loaded_weights, strict=False)
        # LOGGER.info(f"Missing keys: {missing_keys}")

        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        precision = get_precision_config(device_type=device_type)
        self.model = self.model.to(precision)

        all_pe = self.protein_embeds or StateEmbed.load_esm2_embeddings(self.model_conf)
        if isinstance(all_pe, dict):
            all_pe = torch.vstack(list(all_pe.values()))

        self.model.pe_embedding = nn.Embedding.from_pretrained(all_pe)
        self.model.pe_embedding.to(self.model.device, dtype=precision)
        self.model.to(device_type)
        self.model.binary_decoder.requires_grad = False
        self.model.eval()

        if self.protein_embeds is None:
            self.protein_embeds = torch.load(
                get_embedding_cfg(self.model_conf).all_embeddings, weights_only=False
            )
        LOGGER.info("Successfully loaded model")

    def process_data(
        self,
        adata: anndata.AnnData,
    ):

        shape_dict = self.__load_dataset_meta_from_adata(adata)
        adata = self._convert_to_csr(adata)
        gene_column: Optional[str] = self._auto_detect_gene_column(adata)

        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        precision = get_precision_config(device_type=device_type)

        dataloader = create_dataloader(
            self.model_conf,
            adata=adata,
            adata_name="inference",
            shape_dict=shape_dict,
            data_dir=None,
            shuffle=False,
            protein_embeds=self.protein_embeds,
            precision=precision,
            gene_column=gene_column,
            batch_size=self.batch_size,
        )

        return dataloader

    def get_embeddings(self, dataloader):
        all_embeddings = []
        all_ds_embeddings = []
        for embeddings, ds_embeddings in tqdm(
            self.encode(dataloader), total=len(dataloader), desc="Encoding"
        ):
            all_embeddings.append(embeddings)
            if ds_embeddings is not None:
                all_ds_embeddings.append(ds_embeddings)

        all_embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
        if len(all_ds_embeddings) > 0:
            all_ds_embeddings = np.concatenate(all_ds_embeddings, axis=0).astype(
                np.float32
            )

            all_embeddings = np.concatenate(
                [all_embeddings, all_ds_embeddings], axis=-1
            )

        # if output_adata_path is provided, write the adata to the file
        # if output_adata_path is not None:
        #     adata.obsm[emb_key] = all_embeddings
        #     adata.write_h5ad(output_adata_path)

        return all_embeddings

    # def __load_dataset_meta(self, adata_path):
    #     with h5.File(adata_path) as h5f:
    #         attrs = dict(h5f["X"].attrs)
    #         if "encoding-type" in attrs:  # Fixed: was checking undefined 'adata'
    #             if attrs["encoding-type"] in ["csr_matrix", "csc_matrix"]:
    #                 num_cells = attrs["shape"][0]
    #                 num_genes = attrs["shape"][1]
    #             elif attrs["encoding-type"] == "array":
    #                 num_cells = h5f["X"].shape[0]
    #                 num_genes = h5f["X"].shape[1]
    #             else:
    #                 raise ValueError("Input file contains count mtx in non-csr matrix")
    #         else:
    #             # No encoding-type specified, try to infer from dataset structure
    #             if hasattr(h5f["X"], "shape") and len(h5f["X"].shape) == 2:
    #                 # Treat as dense array - get shape directly from dataset
    #                 num_cells = h5f["X"].shape[0]
    #                 num_genes = h5f["X"].shape[1]
    #             elif all(key in h5f["X"] for key in ["indptr", "indices", "data"]):
    #                 # Looks like sparse CSR format
    #                 num_cells = len(h5f["X"]["indptr"]) - 1
    #                 num_genes = attrs.get(
    #                     "shape",
    #                     [
    #                         0,
    #                         (
    #                             h5f["X"]["indices"][:].max() + 1
    #                             if len(h5f["X"]["indices"]) > 0
    #                             else 0
    #                         ),
    #                     ],
    #                 )[1]
    #             else:
    #                 raise ValueError(
    #                     "Cannot determine matrix format - no encoding-type and unrecognized structure"
    #                 )

    #     return {Path(adata_path).stem: (num_cells, num_genes)}

    def __load_dataset_meta_from_adata(self, adata, dataset_name=None):
        """
        Extract dataset metadata directly from an AnnData object.

        Args:
            adata: AnnData object
            dataset_name: Optional name for the dataset. If None, uses 'inference'

        Returns:
            dict: Dictionary with dataset name as key and (num_cells, num_genes) as value
        """
        num_cells = adata.n_obs
        num_genes = adata.n_vars

        if dataset_name is None:
            dataset_name = "inference"

        return {dataset_name: (num_cells, num_genes)}

    def _save_data(self, input_adata_path, output_adata_path, obsm_key, data):
        """
        Save data in the output file. This function addresses following cases:
        - output_adata_path does not exist:
          In this case, the function copies the rest of the input file to the
          output file then adds the data to the output file.
        - output_adata_path exists but the dataset does not exist:
          In this case, the function adds the dataset to the output file.
        - output_adata_path exists and the dataset exists:
          In this case, the function resizes the dataset and appends the data to
          the dataset.
        """
        if not os.path.exists(output_adata_path):
            os.makedirs(os.path.dirname(output_adata_path), exist_ok=True)
            # Copy rest of the input file to output file
            with h5.File(input_adata_path) as input_h5f:
                with h5.File(output_adata_path, "a") as output_h5f:
                    # Replicate the input data to the output file
                    for _, obj in input_h5f.items():
                        input_h5f.copy(obj, output_h5f)
                    output_h5f.create_dataset(
                        f"/obsm/{obsm_key}",
                        chunks=True,
                        data=data,
                        maxshape=(None, data.shape[1]),
                    )
        else:
            with h5.File(output_adata_path, "a") as output_h5f:
                # If the dataset is added to an existing file that does not have the dataset
                if f"/obsm/{obsm_key}" not in output_h5f:
                    output_h5f.create_dataset(
                        f"/obsm/{obsm_key}",
                        chunks=True,
                        data=data,
                        maxshape=(None, data.shape[1]),
                    )
                else:
                    output_h5f[f"/obsm/{obsm_key}"].resize(
                        (output_h5f[f"/obsm/{obsm_key}"].shape[0] + data.shape[0]),
                        axis=0,
                    )
                    output_h5f[f"/obsm/{obsm_key}"][-data.shape[0] :] = data

    def get_gene_embedding(self, genes):
        protein_embeds = [
            self.protein_embeds[x] if x in self.protein_embeds else torch.zeros(5120)
            for x in genes
        ]
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        precision = get_precision_config(device_type=device_type)
        protein_embeds = torch.stack(protein_embeds).to(
            self.model.device, dtype=precision
        )
        return self.model.gene_embedding_layer(protein_embeds)

    def encode(self, dataloader, rda=None):
        with torch.no_grad():
            device_type = "cuda" if torch.cuda.is_available() else "cpu"
            precision = get_precision_config(device_type=device_type)
            with torch.autocast(device_type=device_type, dtype=precision):
                for i, batch in enumerate(dataloader):

                    _, _, _, emb, ds_emb = self.model._compute_embedding_for_batch(
                        batch
                    )
                    embeddings = emb.detach().cpu().float().numpy()

                    ds_emb = self.model.dataset_embedder(ds_emb)
                    ds_embeddings = ds_emb.detach().cpu().float().numpy()

                    yield embeddings, ds_embeddings

    # def encode_adata(
    #     self,
    #     input_adata_path: str,
    #     output_adata_path: str | None = None,
    #     emb_key: str = "X_emb",
    #     dataset_name: str | None = None,
    #     batch_size: int = 32,
    # ):
    #     shape_dict = self.__load_dataset_meta(input_adata_path)
    #     adata = anndata.read_h5ad(input_adata_path)

    #     # # # # # # # # # # 
    #     adata = adata[:10].copy()
    #     # # # # # # # # # # # # 


        
    #     if dataset_name is None:
    #         dataset_name = Path(input_adata_path).stem

    #     # Convert to CSR format if needed
    #     adata = self._convert_to_csr(adata)

    #     # Auto-detect the best gene column
    #     gene_column: Optional[str] = self._auto_detect_gene_column(adata)

    #     device_type = "cuda" if torch.cuda.is_available() else "cpu"
    #     precision = get_precision_config(device_type=device_type)
    #     dataloader = create_dataloader(
    #         self.model_conf,
    #         adata=adata,
    #         adata_name=dataset_name or "inference",
    #         shape_dict=shape_dict,
    #         data_dir=os.path.dirname(input_adata_path),
    #         shuffle=False,
    #         protein_embeds=self.protein_embeds,
    #         precision=precision,
    #         gene_column=gene_column,
    #     )

    #     all_embeddings = []
    #     all_ds_embeddings = []
    #     for embeddings, ds_embeddings in tqdm(
    #         self.encode(dataloader), total=len(dataloader), desc="Encoding"
    #     ):
    #         all_embeddings.append(embeddings)
    #         if ds_embeddings is not None:
    #             all_ds_embeddings.append(ds_embeddings)

    #     # attach this as a numpy array to the adata and write it out
    #     all_embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    #     if len(all_ds_embeddings) > 0:
    #         all_ds_embeddings = np.concatenate(all_ds_embeddings, axis=0).astype(
    #             np.float32
    #         )

    #         # concatenate along axis -1 with all embeddings
    #         all_embeddings = np.concatenate(
    #             [all_embeddings, all_ds_embeddings], axis=-1
    #         )

    #     # if output_adata_path is provided, write the adata to the file
    #     if output_adata_path is not None:
    #         adata.obsm[emb_key] = all_embeddings
    #         adata.write_h5ad(output_adata_path)
        
    #     return all_embeddings, adata

    def _convert_to_csr(self, adata):
        """Convert the adata.X matrix to CSR format if it's not already."""

        if issparse(adata.X) and not isinstance(adata.X, csr_matrix):
            LOGGER.info(f"Converting {type(adata.X).__name__} to csr_matrix format")
            adata.X = csr_matrix(adata.X)
        return adata

    def _auto_detect_gene_column(self, adata):
        """Auto-detect the gene column with highest overlap with protein embeddings."""
        if self.protein_embeds is None:
            LOGGER.warning(
                "No protein embeddings available for auto-detection, using index"
            )
            return None

        protein_genes = set(self.protein_embeds.keys())
        best_column = None
        best_overlap = 0
        best_overlap_pct = 0

        # Check index first
        if hasattr(adata.var, "index"):
            index_genes = set(adata.var.index)
            overlap = len(protein_genes.intersection(index_genes))
            overlap_pct = overlap / len(index_genes) if len(index_genes) > 0 else 0
            if overlap > best_overlap:
                best_overlap = overlap
                best_overlap_pct = overlap_pct
                best_column = None  # None means use index

        # Check all columns in var
        for col in adata.var.columns:
            if adata.var[col].dtype == "object" or adata.var[col].dtype.name.startswith(
                "str"
            ):
                col_genes = set(adata.var[col].dropna().astype(str))
                overlap = len(protein_genes.intersection(col_genes))
                overlap_pct = overlap / len(col_genes) if len(col_genes) > 0 else 0
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_overlap_pct = overlap_pct
                    best_column = col

        if best_column is None:
            LOGGER.info(
                f"Auto-detected gene column: var.index (overlap: {best_overlap}/{len(protein_genes)} protein embeddings, {best_overlap_pct:.1%} of genes)"
            )
        else:
            LOGGER.info(
                f"Auto-detected gene column: var.{best_column} (overlap: {best_overlap}/{len(protein_genes)} protein embeddings, {best_overlap_pct:.1%} of genes)"
            )

        return best_column

    def decode_from_file(
        self, adata_path, emb_key: str, read_depth=None, batch_size=64
    ):
        adata = anndata.read_h5ad(adata_path)
        genes = adata.var.index
        yield from self.decode_from_adata(adata, genes, emb_key, read_depth, batch_size)

    #  reverse engineering - it takes pre-computed cell embeddings and reconstructs the original gene expression from them. 
    @torch.no_grad()
    def decode_from_adata(
        self, adata, genes, emb_key: str, read_depth=None, batch_size=64
    ):
        try:
            cell_embs = adata.obsm[emb_key]
        except:
            cell_embs = adata.X

        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        precision = get_precision_config(device_type=device_type)
        cell_embs = torch.Tensor(cell_embs).to(self.model.device, dtype=precision)

        use_rda = getattr(self.model.cfg.model, "rda", False)
        if use_rda and read_depth is None:
            read_depth = 4.0

        gene_embeds = self.get_gene_embedding(genes)
        with torch.autocast(device_type=device_type, dtype=precision):
            for i in tqdm(
                range(0, cell_embs.size(0), batch_size),
                total=int(cell_embs.size(0) // batch_size),
            ):
                cell_embeds_batch = cell_embs[i : i + batch_size]
                task_counts = torch.full(
                    (cell_embeds_batch.shape[0],),
                    read_depth,
                    device=self.model.device,
                    dtype=precision,
                )

                ds_emb = cell_embeds_batch[
                    :, -self.model.z_dim_ds :
                ]  # last ten columns are the dataset embeddings
                merged_embs = StateEmbeddingModel.resize_batch(
                    cell_embeds_batch,
                    gene_embeds,
                    task_counts=task_counts,
                    ds_emb=ds_emb,
                )
                logprobs_batch = self.model.binary_decoder(merged_embs)
                logprobs_batch = logprobs_batch.detach().cpu().float().numpy()
                yield logprobs_batch.squeeze()

    @staticmethod
    def load_esm2_embeddings(cfg):
        # Load in ESM2 embeddings and special tokens
        all_pe = torch.load(get_embedding_cfg(cfg).all_embeddings, weights_only=False)
        if isinstance(all_pe, dict):
            all_pe = torch.vstack(list(all_pe.values()))

        all_pe = all_pe.cuda()
        return all_pe
