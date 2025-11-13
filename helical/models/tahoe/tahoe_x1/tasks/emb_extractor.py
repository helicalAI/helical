# Copyright (C) Tahoe Therapeutics 2025. All rights reserved.
import logging
from typing import List, Optional

import numpy as np
import torch
from anndata import AnnData
from omegaconf import DictConfig
from tqdm.auto import tqdm

from helical.models.tahoe.tahoe_x1.model import TXModel
from helical.models.tahoe.tahoe_x1.tokenizer import GeneVocab
from helical.models.tahoe.tahoe_x1.utils.util import loader_from_adata

log = logging.getLogger(__name__)


def get_batch_embeddings(
    adata: AnnData,
    model: TXModel,
    vocab: GeneVocab,
    model_cfg: DictConfig,
    collator_cfg: DictConfig,
    gene_ids: Optional[np.ndarray] = None,
    batch_size: int = 8,
    num_workers: int = 8,
    prefetch_factor: int = 48,
    max_length: Optional[int] = None,
    return_gene_embeddings: bool = False,
):
    """Get the cell embeddings for a batch of cells.

    Args:
        adata (AnnData): The AnnData object.
        model (TXModel): The model.
        vocab (GeneVocab): The gene-to-ID vocabulary
        model_cfg (DictConfig, optional): The model configuration dictionary.
        collator_cfg (DictConfig, optional): The collator configuration dictionary.
        gene_ids (np.ndarray, optional): The gene vocabulary ids.
            Defaults to None, in which case the gene IDs are taken from adata.var["id_in_vocab"].
        batch_size (int): The batch size for inference. Defaults to 8.
        num_workers (int): The number of workers for the data loader. Defaults to 8.
        max_length (int, optional): The maximum context length. Defaults to number of genes in the adata.
        return_gene_embeddings (bool): Whether to return the mean gene embeddings as well. Defaults to False.

    Returns:
        Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
            - If `return_gene_embeddings` is False, returns a NumPy array of cell embeddings.
            - If `return_gene_embeddings` is True, returns a tuple of cell embeddings and
              gene embeddings as NumPy arrays.
    """
    device = next(model.parameters()).device
    model.return_gene_embeddings = return_gene_embeddings

    print(f"Using device {device} for inference.")
    collator_cfg["do_mlm"] = False
    data_loader = loader_from_adata(
        adata=adata,
        collator_cfg=collator_cfg,
        vocab=vocab,
        batch_size=batch_size,
        max_length=max_length,
        gene_ids=gene_ids,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )

    cell_embs: List[torch.Tensor] = []

    if return_gene_embeddings:
        gene_array = torch.zeros(
            len(vocab),
            model_cfg["d_model"],
            dtype=torch.float32,
            device=device,
        )
        gene_array_counts = torch.zeros(
            len(vocab),
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
            dtype=dtype_from_string[model_cfg["precision"]],
            device_type=device.type,
        ),
    ):
        pbar = tqdm(total=len(data_loader), desc="Embedding cells")

        for data_dict in data_loader:
            input_gene_ids = data_dict["gene"].to(device)
            src_key_padding_mask = ~input_gene_ids.eq(collator_cfg["pad_token_id"])

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
            )

            cell_embs.append(output["cell_emb"].to("cpu").to(dtype=torch.float32))

            if return_gene_embeddings:
                gene_embs = output.get("gene_emb").to(torch.float32)
                flat_gene_ids = input_gene_ids.view(-1)
                flat_embeddings = gene_embs.view(-1, gene_embs.shape[-1])

                valid = flat_gene_ids != collator_cfg["pad_token_id"]
                flat_gene_ids = flat_gene_ids[valid]
                flat_embeddings = flat_embeddings[valid].to(gene_embs.dtype)

                gene_array.index_add_(0, flat_gene_ids, flat_embeddings)
                gene_array_counts.index_add_(
                    0,
                    flat_gene_ids,
                    torch.ones_like(flat_gene_ids, dtype=gene_embs.dtype),
                )

            pbar.update(len(input_gene_ids))

    cell_array = torch.cat(cell_embs, dim=0).numpy()
    cell_array = cell_array / np.linalg.norm(
        cell_array,
        axis=1,
        keepdims=True,
    )

    if return_gene_embeddings:
        gene_array = gene_array.to("cpu").to(torch.float32).numpy()
        gene_array_counts = gene_array_counts.to("cpu").to(torch.float32).numpy()
        gene_array_counts = np.expand_dims(gene_array_counts, axis=1)

        gene_array = np.divide(
            gene_array,
            gene_array_counts,
            out=np.ones_like(gene_array) * np.nan,
            where=gene_array_counts != 0,
        )

        gene2idx = vocab.get_stoi()
        all_gene_ids = np.array(list(gene2idx.values()))
        gene_array = gene_array[all_gene_ids, :]

    log.info(f"Extracted  cell embeddings of shape {cell_array.shape}.  ")

    if return_gene_embeddings:
        return cell_array, gene_array
    else:
        return cell_array
