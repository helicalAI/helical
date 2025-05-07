"""Transcriptformer model implementation."""

import logging
from typing import Literal
import torch
from torch import Tensor, nn
from torch.nn.attention.flex_attention import and_masks, create_block_mask
import pandas as pd

from helical.models.transcriptformer.data.dataclasses import (
    AuxVocab,
    BatchData,
    DataConfig,
    GeneVocab,
    InferenceConfig,
    LossConfig,
    ModelConfig,
)
from helical.models.transcriptformer.model_dir.layers import (
    MLP,
    CountDecoderHead,
    PretrainedEmbeddings,
    TranscriptEncoder,
    mean_embeddings,
)
from helical.models.transcriptformer.model_dir.losses import (
    ZTP_NLL,
    CrossEntropyLoss,
    logit_softcap,
)
from helical.models.transcriptformer.model_dir.masks import (
    causal_mask_factory,
    pad_mask_factory,
)

logger = logging.getLogger(__name__)

NON_GENE_TOKENS = ["unknown", "[PAD]", "[START]", "[END]", "[RD]", "[CELL]", "[MASK]"]


class Transcriptformer(nn.Module):
    """Autoregressive model that predicts the gene tokens and counts of a cell."""

    def __init__(
        self,
        data_config: DataConfig,
        model_config: ModelConfig,
        loss_config: LossConfig,
        inference_config: InferenceConfig = None,
        gene_vocab_dict: dict = None,
        aux_vocab_dict: dict = None,
        emb_matrix: Tensor = None,
        emb_mode: Literal["gene", "cell"] = "cell",
        **kwargs,
    ):
        super().__init__()

        # Store configurations as dataclass objects
        self.data_config = data_config
        self.model_config = model_config
        self.loss_config = loss_config
        self.inference_config = inference_config
        self.emb_mode = emb_mode
        # initialize vocab dicts
        self.gene_vocab_dict = gene_vocab_dict
        self.aux_vocab_dict = aux_vocab_dict

        # Load vocabularies and initialize model components
        self._build_vocabs()
        self._init_gene_embeddings(emb_matrix)
        self._init_model_components()

    def _build_vocabs(self):
        # Load the gene vocabulary
        self.gene_vocab = GeneVocab(
            vocab_dict=self.gene_vocab_dict,
            vocab_size=len(self.gene_vocab_dict),
            pad_idx=self.gene_vocab_dict["[PAD]"],
            start_idx=self.gene_vocab_dict.get("[START]"),
            end_idx=self.gene_vocab_dict.get("[END]"),
            cell_idx=self.gene_vocab_dict.get("[CELL]"),
            embedding_matrix=None,
        )
        # Create reverse dictionary for efficient token to gene name lookup
        self.token_to_gene_dict = {v: k for k, v in self.gene_vocab_dict.items()}

        # Load the auxiliary vocab if provided
        if self.aux_vocab_dict is not None:
            self.model_config.use_aux = True
            self.aux_vocab = AuxVocab(
                vocab_dict=self.aux_vocab_dict,
                vocab_size=sum([len(v) + 1 for v in self.aux_vocab_dict.values()]),
                pad_ids=[
                    self.aux_vocab_dict[k]["unknown"] for k in self.aux_vocab_dict
                ],
                aux_seq_len=len(self.aux_vocab_dict),
            )

    def _init_model_components(self):
        if hasattr(self.model_config, "use_aux"):
            self._init_aux_embeddings()

        # Encoder and heads
        self._init_encoder()
        self._init_count_heads()
        if self.loss_config.gene_id_loss_weight > 0:
            self._init_gene_id_head()

        # Loss initialization
        self._init_nll_loss()
        if self.loss_config.gene_id_loss_weight > 0:
            self._init_gene_id_loss()

        self._init_masks()

    def _init_masks(self):
        self.causal_mask = causal_mask_factory()

    def _init_gene_embeddings(self, emb_matrix: Tensor = None):
        self.gene_embeddings = PretrainedEmbeddings(
            embedding_matrix=emb_matrix,
            output_dim=self.model_config.embed_dim,
            freeze=True,
            normalize=True,
            dropout=self.model_config.dropout,
        )

    def _init_aux_embeddings(self):
        # Initialize auxiliary embeddings based on model_config and vocab
        aux_vocab_dict = self.aux_vocab.vocab_dict
        self.aux_embeddings = nn.ModuleDict(
            {
                key: nn.Embedding(len(vocab), self.model_config.embed_dim)
                for key, vocab in aux_vocab_dict.items()
            }
        )

    def _init_encoder(self):
        # Initialize the transformer encoder using model_config attributes
        self.transformer_encoder = TranscriptEncoder(
            embed_dim=self.model_config.embed_dim,
            num_head=self.model_config.num_heads,
            model_dim=self.model_config.model_dim,
            dropout=self.model_config.dropout,
            nlayers=self.model_config.num_layers,
            activation=self.model_config.activation,
            attn_bias=self.model_config.attn_bias,
            fw_bias=self.model_config.fw_bias,
        )

    def _init_count_heads(self):
        # Head layers
        self.mu = CountDecoderHead(
            model_dim=2
            * self.model_config.embed_dim,  # Double the embedding dim for the skip connection
            link_func=self.model_config.mu_link_fn,
            eps=self.model_config.log_counts_eps,
            dropout=self.model_config.dropout,
            gene_bias_size=self.gene_vocab.vocab_size,  # Gene specific bias
        )

    def _init_gene_id_head(self):
        # initialize gene id head
        self.gene_id_head = MLP(
            self.model_config.embed_dim,
            self.model_config.gene_head_hidden_dim,
            self.gene_vocab.vocab_size,
            dropout=self.model_config.dropout,
        )

    def _init_nll_loss(self):
        self.criterion = ZTP_NLL(
            eps=self.model_config.log_counts_eps,
            softplus_approx=self.loss_config.softplus_approx,
            max_counts=self.data_config.clip_counts,
        )

    def _init_gene_id_loss(self):
        self.gene_id_criterion = CrossEntropyLoss(
            shift_right=False,
            softcap=self.model_config.softcap,
        )

    def _pad_mask(self, gene_tokens, aux_tokens=None, dtype="float"):
        # Create the pad mask
        # False for masked positions
        pad_mask = gene_tokens == self.gene_vocab.pad_idx
        if aux_tokens is not None:
            aux_pad_mask = torch.stack(
                [aux_tokens == pad_id for pad_id in self.aux_vocab.pad_ids], dim=-1
            ).any(dim=-1)
            pad_mask = torch.cat([aux_pad_mask, pad_mask], dim=1)
        # convert mask to float
        if dtype == "float":
            pad_mask = pad_mask.float().masked_fill(pad_mask, float("-inf"))
        elif dtype == "bool":
            pad_mask = ~pad_mask
        pad_mask.requires_grad = False
        return pad_mask

    def get_gene_embeddings(self, batch: BatchData):
        right_shifted_gene_tokens = torch.cat(
            [
                self.gene_vocab.start_idx
                * torch.ones_like(batch.gene_token_indices[:, :1]),
                batch.gene_token_indices[:, :-1],
            ],
            dim=1,
        )

        return self.gene_embeddings(right_shifted_gene_tokens), self.gene_embeddings(
            batch.gene_token_indices
        )

    def get_aux_embeddings(self, aux_token_indices):
        aux_embeddings = []
        for i, aux_embedding in enumerate(self.aux_embeddings.values()):
            aux_embeddings.append(aux_embedding(aux_token_indices[:, i]))
        aux_embeddings = torch.stack(aux_embeddings, dim=1)
        return aux_embeddings

    def _score_mod_factory(self, log_counts, emb_mode=False):
        softcap = self.model_config.softcap

        def score_mod(score, b, h, q_idx, kv_idx):
            bias = log_counts[b, kv_idx] * ((q_idx > kv_idx) | emb_mode)
            score = score + bias
            if softcap > 0:
                score = logit_softcap(score, softcap)
            return score

        return score_mod

    def _get_gene_embeddings(self, transformer_output) -> list[pd.Series]:
        """
        Get the gene embeddings from the transformer output.

        Returns
        -------
            list[pd.Series]: A list of pandas Series, one for each cell, with the gene names as index and the embeddings as values.
        """
        # Create a list where each entry is a cell's gene embeddings Series
        gene_embeddings = []
        gene_output_cpu = transformer_output["gene_embeddings"].detach().cpu().numpy()
        gene_tokens = transformer_output["input_gene_token_indices"]

        for cell_idx in range(gene_output_cpu.shape[0]):
            # Get the actual gene tokens for this cell
            cell_gene_tokens = gene_tokens[cell_idx]
            gene_names = []
            embeddings = []
            for pos_idx in range(gene_output_cpu.shape[1]):
                if not transformer_output["mask"][
                    cell_idx, pos_idx
                ]:  # Only include non-padded positions
                    gene_token = cell_gene_tokens[pos_idx].item()
                    gene_name = self.token_to_gene_dict[gene_token]
                    if gene_name in NON_GENE_TOKENS:
                        continue
                    gene_names.append(gene_name)
                    embeddings.append(gene_output_cpu[cell_idx, pos_idx])
            # Create a pandas Series with gene names as index
            cell_gene_series = pd.Series(embeddings, index=gene_names)
            gene_embeddings.append(cell_gene_series)
        return gene_embeddings

    def forward(
        self,
        batch: BatchData,
        embed: bool = False,
        **kwargs,
    ) -> dict:
        """Forward pass of the model.

        Args:
            batch: BatchData containing gene counts, token indices, and optional auxiliary data
            embed: Whether to compute embeddings
            **kwargs: Additional arguments

        Returns
        -------
            dict: Model output containing:
                - mu: Predicted rate of count distribution
                - mask: Output mask
                - input_counts: Input count data
                - embeddings: Optional cell embeddings
        """
        device = next(self.parameters()).device
        aux_token_indices = batch.aux_token_indices.to(device)
        gene_token_indices = batch.gene_token_indices.to(device)
        gene_counts = batch.gene_counts.to(device)

        # append the start token to the gene_tokens
        # drop the last token from the gene_tokens to keep the same length as the counts
        right_shifted_gene_tokens = torch.cat(
            [
                self.gene_vocab.start_idx * torch.ones_like(gene_token_indices[:, :1]),
                gene_token_indices[:, :-1],
            ],
            dim=1,
        )
        right_shifted_counts = torch.cat(
            [torch.ones_like(gene_counts[:, :1]), gene_counts[:, :-1]], dim=1
        )

        # Embed the gene_tokens
        right_shifted_gene_embeddings, gene_embeddings = self.get_gene_embeddings(batch)

        # Add auxiliary tokens to the embeddings
        if aux_token_indices is not None:
            aux_embeddings = self.get_aux_embeddings(aux_token_indices)
            right_shifted_gene_embeddings = torch.cat(
                [aux_embeddings, right_shifted_gene_embeddings], dim=1
            )
            right_shifted_counts = torch.cat(
                [torch.ones_like(aux_token_indices), right_shifted_counts], dim=1
            )

        # Compile the block mask
        pad_mask = self._pad_mask(
            right_shifted_gene_tokens, aux_token_indices, dtype="bool"
        )
        mask_mod = and_masks(pad_mask_factory(pad_mask), self.causal_mask)

        block_mask = create_block_mask(
            mask_mod,
            right_shifted_gene_embeddings.shape[0],
            H=None,
            Q_LEN=right_shifted_gene_embeddings.shape[1],
            KV_LEN=right_shifted_gene_embeddings.shape[1],
            BLOCK_SIZE=self.model_config.block_len,
            _compile=True,
        )

        # Score mode
        log_counts = torch.log1p(
            right_shifted_counts + self.model_config.log_counts_eps
        )
        score_mod = self._score_mod_factory(log_counts, emb_mode=embed)

        # Apply the transformer encoder
        transformer_output = self.transformer_encoder(
            x=right_shifted_gene_embeddings,
            score_mod=score_mod,
            block_mask=block_mask,
        )

        # Extract the auxiliary features from the output
        if aux_token_indices is not None:
            # Determine the number of auxiliary tokens
            num_aux_tokens = aux_token_indices.shape[1]

            # Separate the auxiliary output and gene output from the concatenated embeddings
            gene_output = transformer_output[:, num_aux_tokens:, :]

            # Adjust the pad mask to exclude the auxiliary tokens
            pad_mask = pad_mask[:, num_aux_tokens:]
        else:
            gene_output = transformer_output

        result = {}

        if embed:
            result["cell_embeddings"] = (
                mean_embeddings(gene_output, pad_mask).detach().cpu()
            )
            result["gene_embeddings"] = gene_output.detach().cpu()

        # Append the end token to the gene_tokens
        result["input_gene_token_indices"] = torch.cat(
            [
                gene_token_indices[:, :-1],
                self.gene_vocab.end_idx * torch.ones_like(gene_token_indices[:, :1]),
            ],
            dim=1,
        )

        # set last count to zero
        result["input_counts"] = torch.cat(
            [gene_counts[:, :-1], torch.ones_like(gene_counts[:, :1])], dim=1
        )

        # concatenate gene_output with input embeddings
        # this allows the model to condition on the input gene tokens
        conditioned_output = torch.cat([gene_output, gene_embeddings], dim=-1)

        # Calculate the mu parameter
        result["mu"] = self.mu(
            gene_output=conditioned_output,
            gene_tokens=gene_token_indices,
            mask=self._pad_mask(gene_token_indices, dtype="float"),
        )
        if self.model_config.mu_link_fn == "softmax":
            # avoid in-place operations with tensor involved in the backward pass
            result["mu"] = result["mu"] * gene_counts.sum(dim=1, keepdim=True)

        # Mask the output
        # convention for loss functions is true = masked
        result["mask"] = ~self._pad_mask(gene_token_indices, dtype="bool")

        # Calculate the gene id loss
        if self.loss_config.gene_id_loss_weight > 0:
            gene_logit = self.gene_id_head(gene_output)
            result["gene_logit"] = gene_logit

        return result

    @torch.no_grad()
    def inference(self, batch: BatchData):
        """Run inference on a batch of data.

        Args:
            batch: BatchData containing input data

        Returns
        -------
            Dictionary containing inference results
        """
        # Flex Attn requires batch size to be a multiple of BLOCK_SIZE
        batch_size = batch.gene_counts.shape[0]

        if batch_size > self.inference_config.batch_size:
            raise ValueError(
                f"Batch size {batch_size} exceeds model batch size {self.inference_config.batch_size}"
            )

        pad_rows = self.inference_config.batch_size - batch_size

        resized_batch = BatchData(
            **{
                field_name: (
                    self._resize_data(field_value, self.inference_config.batch_size)
                )
                for field_name, field_value in batch.__dict__.items()
            }
        )

        transformer_output = self.forward(batch=resized_batch, embed=True)

        if pad_rows > 0:
            # Remove the last pad_rows rows from the batch
            for key, value in transformer_output.items():
                if isinstance(value, torch.Tensor):
                    transformer_output[key] = value[:-pad_rows]
                    assert len(transformer_output[key]) == batch_size

        if "llh" in self.inference_config.output_keys:
            llh = (
                self.criterion(**transformer_output, eval_mode=not self.training)
                .detach()
                .cpu()
            )
            transformer_output["llh"] = llh

        if "gene_llh" in self.inference_config.output_keys:
            self.gene_id_criterion.reduction = None
            gene_llh = (
                self.gene_id_criterion(
                    logits=transformer_output["gene_logit"],
                    input_ids=transformer_output["input_gene_token_indices"],
                    mask=transformer_output["mask"],
                )
                .detach()
                .cpu()
            )
            transformer_output["gene_llh"] = gene_llh

        results = {}
        results["obs"] = batch.obs
        results.update(
            {key: transformer_output[key] for key in self.inference_config.output_keys}
        )

        if self.emb_mode == "gene":
            results["gene_embeddings"] = self._get_gene_embeddings(transformer_output)
        elif self.emb_mode == "cell":
            results["cell_embeddings"] = (
                transformer_output["cell_embeddings"].detach().cpu()
            )
        else:
            raise ValueError(f"Invalid embedding mode: {self.emb_mode}")

        return results

    def _resize_data(self, data, config_batch_size):
        """
        Resize batch to match model batch_size.

        Necessary for inference when batch_size is not a multiple of model batch_size.
        """
        if not isinstance(data, torch.Tensor):
            return data

        batch_size = data.shape[0]
        if batch_size != config_batch_size:
            data = self._pad_rows(
                config_batch_size - batch_size,
                data,
                "int" if data.dtype == torch.int64 else "float",
            )
        return data

    def _pad_rows(self, pad_len, x, dtype="int"):
        # Create padding with same shape as x except for first dimension
        pad_shape = (pad_len,) + x.shape[1:]
        pad = torch.zeros(pad_shape, device=x.device)
        if dtype == "int":
            pad = pad.int()
        elif dtype == "float":
            pad = pad.float()
        else:
            raise ValueError(f"Invalid padding type: {dtype}")
        return torch.cat([x, pad], dim=0)

    def predict_step(self, batch, batch_idx):
        assert (
            self.inference_config.output_keys
        ), "output_keys must be set in inference_config"
        return self.inference(batch)
