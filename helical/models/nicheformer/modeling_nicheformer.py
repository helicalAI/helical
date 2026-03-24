import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import MaskedLMOutput
from .configuration_nicheformer import NicheformerConfig
from .masking import complete_masking
import math


class PositionalEncoding(nn.Module):
    """Positional encoding using sine and cosine functions."""

    def __init__(self, d_model: int, max_seq_len: int):
        super().__init__()
        encoding = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        encoding = encoding.unsqueeze(0)

        self.register_buffer("encoding", encoding, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor."""
        return x + self.encoding[:, : x.size(1)]


class NicheformerPreTrainedModel(PreTrainedModel):
    """Base class for Nicheformer models."""

    config_class = NicheformerConfig
    base_model_prefix = "nicheformer"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


class NicheformerModel(NicheformerPreTrainedModel):
    def __init__(self, config: NicheformerConfig):
        super().__init__(config)

        # Core transformer components
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.dim_model,
            nhead=config.nheads,
            dim_feedforward=config.dim_feedforward,
            batch_first=config.batch_first,
            dropout=config.dropout,
            layer_norm_eps=1e-12,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=config.nlayers,
            enable_nested_tensor=False,
        )

        # Embedding layers
        self.embeddings = nn.Embedding(
            num_embeddings=config.n_tokens + 5,
            embedding_dim=config.dim_model,
            padding_idx=1,
        )

        if config.learnable_pe:
            self.positional_embedding = nn.Embedding(
                num_embeddings=config.context_length, embedding_dim=config.dim_model
            )
            self.dropout = nn.Dropout(p=config.dropout)
            self.register_buffer(
                "pos", torch.arange(0, config.context_length, dtype=torch.long)
            )
        else:
            self.positional_embedding = PositionalEncoding(
                d_model=config.dim_model, max_seq_len=config.context_length
            )

        # Initialize weights
        self.post_init()

    def forward(self, input_ids, attention_mask=None):
        token_embedding = self.embeddings(input_ids)

        if self.config.learnable_pe:
            pos_embedding = self.positional_embedding(
                self.pos.to(token_embedding.device)
            )
            embeddings = self.dropout(token_embedding + pos_embedding)
        else:
            embeddings = self.positional_embedding(token_embedding)

        # Convert attention_mask to boolean and invert it for transformer's src_key_padding_mask
        # True indicates positions that will be masked
        if attention_mask is not None:
            attention_mask = ~attention_mask.bool()

        transformer_output = self.encoder(
            embeddings,
            src_key_padding_mask=attention_mask if attention_mask is not None else None,
            is_causal=False,
        )

        return transformer_output

    def get_embeddings(
        self,
        input_ids,
        attention_mask=None,
        layer: int = -1,
        with_context: bool = False,
    ) -> torch.Tensor:
        """Get embeddings from the model.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            layer: Which transformer layer to extract embeddings from (-1 means last layer)
            with_context: Whether to include context tokens in the embeddings

        Returns:
            torch.Tensor: Embeddings tensor
        """
        # Get token embeddings and positional encodings
        token_embedding = self.embeddings(input_ids)

        if self.config.learnable_pe:
            pos_embedding = self.positional_embedding(
                self.pos.to(token_embedding.device)
            )
            embeddings = self.dropout(token_embedding + pos_embedding)
        else:
            embeddings = self.positional_embedding(token_embedding)

        # Process through transformer layers up to desired layer
        if layer < 0:
            layer = self.config.nlayers + layer  # -1 means last layer

        # Convert attention_mask to boolean and invert it for transformer's src_key_padding_mask
        if attention_mask is not None:
            padding_mask = ~attention_mask.bool()
        else:
            padding_mask = None

        # Process through each layer up to the desired one
        for i in range(layer + 1):
            embeddings = self.encoder.layers[i](
                embeddings, src_key_padding_mask=padding_mask, is_causal=False
            )

        # Remove context tokens (first 3 tokens) if not needed
        if not with_context:
            embeddings = embeddings[:, 3:, :]

        # Mean pooling over sequence dimension
        embeddings = embeddings.mean(dim=1)

        return embeddings


class NicheformerForMaskedLM(NicheformerPreTrainedModel):
    def __init__(self, config: NicheformerConfig):
        super().__init__(config)

        self.nicheformer = NicheformerModel(config)
        self.classifier_head = nn.Linear(config.dim_model, config.n_tokens, bias=False)
        self.classifier_head.bias = nn.Parameter(torch.zeros(config.n_tokens))

        # Initialize weights
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        return_dict=None,
        apply_masking=False,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # Apply masking if requested (typically during training)
        if apply_masking:
            batch = {"input_ids": input_ids, "attention_mask": attention_mask}
            masked_batch = complete_masking(
                batch, self.config.masking_p, self.config.n_tokens
            )
            input_ids = masked_batch["masked_indices"]
            labels = masked_batch["input_ids"]  # Original tokens become labels
            mask = masked_batch["mask"]
            # Only compute loss on masked tokens and ensure labels are long
            labels = torch.where(
                mask, labels, torch.tensor(-100, device=labels.device)
            ).long()

        transformer_output = self.nicheformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        prediction_scores = self.classifier_head(transformer_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.n_tokens), labels.view(-1)
            )

        if not return_dict:
            output = (prediction_scores,) + (transformer_output,)
            return (
                ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
            )

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=transformer_output,
        )

    def get_embeddings(
        self,
        input_ids,
        attention_mask=None,
        layer: int = -1,
        with_context: bool = False,
    ) -> torch.Tensor:
        """Get embeddings from the model.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            layer: Which transformer layer to extract embeddings from (-1 means last layer)
            with_context: Whether to include context tokens in the embeddings

        Returns:
            torch.Tensor: Embeddings tensor
        """
        return self.nicheformer.get_embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
            layer=layer,
            with_context=with_context,
        )
