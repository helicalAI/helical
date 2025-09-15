import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import typing as tp
import yaml

from .model_utils import get_loss_fn

LOGGER = logging.getLogger(__name__)


class LatentToGeneDecoder(nn.Module):
    """
    A decoder module to transform latent embeddings back to gene expression space.

    This takes concat([cell embedding]) as the input, and predicts
    counts over all genes as output.

    This decoder is trained separately from the main perturbation model.

    Args:
        latent_dim: Dimension of latent space
        gene_dim: Dimension of gene space (number of HVGs)
        hidden_dims: List of hidden layer dimensions
        dropout: Dropout rate
        residual_decoder: If True, adds residual connections between every other layer block
    """

    def __init__(
        self,
        latent_dim: int,
        gene_dim: int,
        hidden_dims: List[int] = [512, 1024],
        dropout: float = 0.1,
        residual_decoder=False,
    ):
        super().__init__()

        self.residual_decoder = residual_decoder

        if residual_decoder:
            # Build individual blocks for residual connections
            self.blocks = nn.ModuleList()
            input_dim = latent_dim

            for hidden_dim in hidden_dims:
                block = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                self.blocks.append(block)
                input_dim = hidden_dim

            # Final output layer
            self.final_layer = nn.Sequential(nn.Linear(input_dim, gene_dim), nn.ReLU())
        else:
            # Original implementation without residual connections
            layers = []
            input_dim = latent_dim

            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
                input_dim = hidden_dim

            # Final output layer
            layers.append(nn.Linear(input_dim, gene_dim))
            # Make sure outputs are non-negative
            layers.append(nn.ReLU())

            self.decoder = nn.Sequential(*layers)

    def gene_dim(self):
        # return the output dimension of the last layer
        if self.residual_decoder:
            return self.final_layer[0].out_features
        else:
            for module in reversed(self.decoder):
                if isinstance(module, nn.Linear):
                    return module.out_features
            return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the decoder.

        Args:
            x: Latent embeddings of shape [batch_size, latent_dim]

        Returns:
            Gene expression predictions of shape [batch_size, gene_dim]
        """
        if self.residual_decoder:
            # Apply blocks with residual connections between every other block
            block_outputs = []
            current = x

            for i, block in enumerate(self.blocks):
                output = block(current)

                # Add residual connection from every other previous block
                # Pattern: blocks 1, 3, 5, ... get residual from blocks 0, 2, 4, ...
                if i >= 1 and i % 2 == 1:  # Odd-indexed blocks (1, 3, 5, ...)
                    residual_idx = i - 1  # Previous even-indexed block
                    output = output + block_outputs[residual_idx]

                block_outputs.append(output)
                current = output

            return self.final_layer(current)
        else:
            return self.decoder(x)


class PerturbationModelWithoutLightning(ABC, nn.Module):
    """
    Base class for perturbation models that can operate on either raw counts or embeddings.

    Args:
        input_dim: Dimension of input features (genes or embeddings)
        hidden_dim: Hidden dimension for neural network layers
        output_dim: Dimension of output (always gene space)
        pert_dim: Dimension of perturbation embeddings
        dropout: Dropout rate
        lr: Learning rate for optimizer
        loss_fn: Loss function ('mse' or custom nn.Module)
        output_space: 'gene' or 'latent'
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        pert_dim: int,
        batch_dim: int = None,
        dropout: float = 0.1,
        lr: float = 3e-4,
        loss_fn: nn.Module = nn.MSELoss(),
        control_pert: str = "non-targeting",
        embed_key: Optional[str] = None,
        output_space: str = "gene",
        gene_names: Optional[List[str]] = None,
        batch_size: int = 64,
        gene_dim: int = 5000,
        hvg_dim: int = 2001,
        decoder_cfg: dict | None = None,
        **kwargs,
    ):
        super().__init__()
        self.decoder_cfg = decoder_cfg        
        self.gene_decoder_bool = kwargs.get("gene_decoder_bool", True)

        # Core architecture settings
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.pert_dim = pert_dim
        self.batch_dim = batch_dim
        self.gene_dim = gene_dim
        self.hvg_dim = hvg_dim

        if kwargs.get("batch_encoder", False):
            self.batch_dim = batch_dim
        else:
            self.batch_dim = None

        self.residual_decoder = kwargs.get("residual_decoder", False)

        self.embed_key = embed_key
        self.output_space = output_space
        self.batch_size = batch_size
        self.control_pert = control_pert

        # Training settings
        self.gene_names = gene_names  # store the gene names that this model output for gene expression space
        self.dropout = dropout
        self.lr = lr
        self.loss_fn = get_loss_fn(loss_fn)
        # self._build_decoder()
        self.on_load_checkpoint()

    def transfer_batch_to_device(self, batch, device, dataloader_idx: int):
        return {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }

    @abstractmethod
    def _build_networks(self):
        """Build the core neural network components."""
        pass

    # def _build_decoder(self):
    #     """Create self.gene_decoder from self.decoder_cfg (or leave None)."""
    #     if self.gene_decoder_bool == False:
    #         self.gene_decoder = None
    #         return
    #     if self.decoder_cfg is None:
    #         self.gene_decoder = None
    #         return
    #     self.gene_decoder = LatentToGeneDecoder(**self.decoder_cfg)

    def on_load_checkpoint(self) -> None:
        decoder_already_configured = (
            hasattr(self, "_decoder_externally_configured")
            and self._decoder_externally_configured
        )

        if self.gene_decoder_bool == False:
            self.gene_decoder = None
            return
        if (
            not decoder_already_configured
            and self.decoder_cfg
        ):
            self.gene_decoder = LatentToGeneDecoder(**self.decoder_cfg)
            LOGGER.info(
                f"Loaded decoder from checkpoint decoder_cfg: {self.decoder_cfg}"
            )
        elif not decoder_already_configured:
            # Only fall back to old logic if no decoder_cfg was saved and not externally configured
            self.decoder_cfg = None
            self._build_decoder()
            LOGGER.info(f"DEBUG: output_space: {self.output_space}")
            if self.gene_decoder is None:
                gene_dim = (
                    self.hvg_dim if self.output_space == "gene" else self.gene_dim
                )
                LOGGER.info(f"DEBUG: gene_dim: {gene_dim}")
                if (
                    self.embed_key
                    and self.embed_key != "X_hvg"
                    and self.output_space == "gene"
                ) or (
                    self.embed_key and self.output_space == "all"
                ):  # we should be able to decode from hvg to all
                    LOGGER.info(f"DEBUG: Creating gene_decoder, checking conditions...")
                    if gene_dim > 10000:
                        hidden_dims = [1024, 512, 256]
                    else:
                        if "DMSO_TF" in self.control_pert:
                            if self.residual_decoder:
                                hidden_dims = [2058, 2058, 2058, 2058, 2058]
                            else:
                                hidden_dims = [4096, 2048, 2048]
                        elif "PBS" in self.control_pert:
                            hidden_dims = [2048, 1024, 1024]
                        else:
                            hidden_dims = [1024, 1024, 512]  # make this config

                    self.gene_decoder = LatentToGeneDecoder(
                        latent_dim=self.output_dim,
                        gene_dim=gene_dim,
                        hidden_dims=hidden_dims,
                        dropout=self.dropout,
                        residual_decoder=self.residual_decoder,
                    )
                    LOGGER.info(
                        f"Initialized gene decoder for embedding {self.embed_key} to gene space"
                    )
        else:
            LOGGER.info(
                "Decoder was already configured externally, skipping checkpoint decoder configuration"
            )

    def predict_step(self, batch, batch_idx, **kwargs):
        """
        Typically used for final inference. We'll replicate old logic:
         returning 'preds', 'X', 'pert_name', etc.
        """
        latent_output = self.forward(batch)
        output_dict = {
            "preds": latent_output,
            "pert_cell_emb": batch.get("pert_cell_emb", None),
            "pert_cell_counts": batch.get("pert_cell_counts", None),
            "pert_name": batch.get("pert_name", None),
            "celltype_name": batch.get("cell_type", None),
            "batch": batch.get("batch", None),
            "ctrl_cell_emb": batch.get("ctrl_cell_emb", None),
        }

        if self.gene_decoder is not None:
            pert_cell_counts_preds = self.gene_decoder(latent_output)
            output_dict["pert_cell_counts_preds"] = pert_cell_counts_preds

        return output_dict

    def decode_to_gene_space(
        self, latent_embeds: torch.Tensor, basal_expr: None
    ) -> torch.Tensor:
        """
        Decode latent embeddings to gene expression space.

        Args:
            latent_embeds: Embeddings in latent space

        Returns:
            Gene expression predictions or None if decoder is not available
        """
        if self.gene_decoder is not None:
            pert_cell_counts_preds = self.gene_decoder(latent_embeds)
            if basal_expr is not None:
                # Add basal expression if provided
                pert_cell_counts_preds += basal_expr
            return pert_cell_counts_preds
        return None