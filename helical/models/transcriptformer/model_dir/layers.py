import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.attention.flex_attention import flex_attention


def clones(module, N):
    """Produce N identical layers."""
    return torch.nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def mean_embeddings(embeddings, mask=None, include_masked=True):
    if mask is None:
        return embeddings.mean(dim=1)
    elif include_masked:
        return (embeddings * mask.unsqueeze(-1)).mean(dim=1)
    else:
        return (embeddings * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1)


class MultiHeadSelfFlexAttn(nn.Module):
    """
    Multi-Head Self-Attention with Flex Attention mechanism.

    This module implements a multi-head self-attention layer using flex attention,
    which allows for custom score modifications and block masking.

    Args:
        d_model (int): The dimension of the model (input and output)
        nheads (int): The number of attention heads
        bias (bool, optional): Whether to include bias in linear layers. Defaults to False.

    Attributes
    ----------
        d_k (int): The dimension of keys/queries in each head
        h (int): The number of attention heads
        linears (nn.ModuleList): List of linear transformations for Q, K, V, and output
        attn (None): Placeholder for attention scores (not used in this implementation)

    """

    def __init__(self, d_model: int, nheads: int, bias: bool = False):
        super().__init__()
        assert d_model % nheads == 0, "d_model must be divisible by number of heads"

        # We assume d_v always equals d_k
        self.d_k = d_model // nheads
        self.h = nheads
        self.linears = clones(nn.Linear(d_model, d_model, bias=bias), 4)
        self.self_attn = flex_attention

    def forward(self, inp, score_mod=None, block_mask=None, **kwargs):
        """Forward pass of the MultiHeadSelfFlexAttn."""
        batch_size = inp.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        q, k, v = (
            layer(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            for layer, x in zip(self.linears, [inp] * 3, strict=False)
        )

        # 2) Apply attention on all the projected vectors in batch.
        o = self.self_attn(
            q,
            k,
            v,
            score_mod=score_mod,
            block_mask=block_mask,
            kernel_options={
                "BLOCK_M": 32,
                "BLOCK_N": 32,
                "BLOCK_M1": 32,
                "BLOCK_N1": 32,
                "BLOCK_M2": 32,
                "BLOCK_N2": 32,
            },
        )

        # 3) "Concat" using a view and apply a final linear.
        o = o.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.linears[-1](o)


class FlexAttnTransformerLayer(nn.Module):
    """
    Flex Attention Transformer Layer

    This layer implements a transformer block with flex attention mechanism.
    It includes self-attention with customizable score modifications and block masking,
    followed by a feed-forward network.

    Args:
        d_model (int): The dimension of the model (input and output)
        nhead (int): The number of attention heads
        dim_fw (int, optional): The dimension of the feed-forward network. Defaults to 2048.
        dropout (float, optional): Dropout rate. Defaults to 0.0.
        fw_bias (bool, optional): Whether to include bias in feed-forward layers. Defaults to False.
        attn_bias (bool, optional): Whether to include bias in attention layers. Defaults to False.
        activation (str, optional): Activation function to use. Can be 'relu', 'gelu', or 'silu'. Defaults to 'gelu'.

    Attributes
    ----------
        linear1 (nn.Linear): First linear layer of the feed-forward network
        linear2 (nn.Linear): Second linear layer of the feed-forward network
        self_attn (MultiHeadSelfFlexAttn): Multi-head self-attention layer with flex attention
        norm1 (nn.LayerNorm): Layer normalization for attention output
        norm2 (nn.LayerNorm): Layer normalization for feed-forward output
        dropout (nn.Dropout): Dropout layer
        activation (nn.Module): Activation function

    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_fw: int = 2048,
        dropout: float = 0.0,
        fw_bias: bool = False,
        attn_bias: bool = False,
        activation: str = "gelu",
    ):
        super().__init__()

        self.linear1 = nn.Linear(d_model, dim_fw, bias=fw_bias)
        self.linear2 = nn.Linear(dim_fw, d_model, bias=fw_bias)
        self.self_attn = MultiHeadSelfFlexAttn(d_model, nhead, bias=attn_bias)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Activation {activation} not supported")

    def forward(self, x, score_mod=None, block_mask=None):
        """Forward pass of the FlexAttnTransformerLayer."""
        x = self.norm1(x)  # pre-norm is the new norm
        x = self.self_attn(x, score_mod=score_mod, block_mask=block_mask) + x
        x = self.norm2(x)
        x = self.linear2(self.dropout(self.activation(self.linear1(x)))) + x
        return x


class TranscriptEncoder(nn.Module):
    """
    TranscriptEncoder module for encoding transcript sequences.

    This module applies a series of FlexAttnTransformerLayers to the input sequence,
    incorporating count-based attention bias and optional masking.

    Args:
        embed_dim (int): The dimension of the input embeddings.
        num_head (int): The number of attention heads in each transformer layer.
        nlayers (int): The number of transformer layers.
        model_dim (int, optional): The dimension of the feed-forward network in each layer. Defaults to 2048.
        dropout (float, optional): Dropout rate. Defaults to 0.0.
        activation (str, optional): Activation function to use. Can be 'relu', 'gelu', or 'silu'. Defaults to 'gelu'.
        attn_bias (bool, optional): Whether to include bias in attention layers. Defaults to False.
        fw_bias (bool, optional): Whether to include bias in feed-forward layers. Defaults to False.
        eps (float, optional): Small constant for numerical stability. Defaults to 1e-6.
        softcap (int, optional): If > 0, applies a soft cap to attention scores. Defaults to 0.

    Attributes
    ----------
        nlayers (int): The number of transformer layers.
        nheads (int): The number of attention heads in each layer.
        eps (float): Small constant for numerical stability.
        score_mod_factory (function): Factory function for creating score modifiers.
        mask_mod_factory (function): Factory function for creating mask modifiers.
        encoder_layers (nn.ModuleList): List of FlexAttnTransformerLayer modules.

    """

    def __init__(
        self,
        embed_dim: int,
        num_head: int,
        nlayers: int,
        model_dim: int = 2048,
        dropout: float = 0.0,
        activation: str = "gelu",
        attn_bias: bool = False,
        fw_bias: bool = False,
    ):
        super().__init__()
        self.nlayers = nlayers
        self.nheads = num_head

        self.encoder_layers = nn.ModuleList(
            [
                FlexAttnTransformerLayer(
                    d_model=embed_dim,
                    nhead=num_head,
                    dim_fw=model_dim,
                    dropout=dropout,
                    fw_bias=fw_bias,
                    attn_bias=attn_bias,
                    activation=activation,
                )
                for _ in range(nlayers)
            ]
        )

    def forward(self, x, score_mod, block_mask):
        """
        Forward pass of the TranscriptEncoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).
            score_mod (torch.Tensor): Score modifier tensor for attention.
            block_mask (torch.Tensor): Block mask tensor for attention.

        Returns
        -------
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embed_dim).
        """
        for layer in self.encoder_layers:
            x = layer(x, score_mod=score_mod, block_mask=block_mask)
        return x


class MaskedSoftmax(nn.Module):
    def __init__(self, dim: int, softcap: int = 0):
        super().__init__()
        self.dim = dim
        self.softcap = softcap

    def forward(self, x: Tensor, m: Tensor) -> Tensor:
        """
        Forward pass.

        x: (batch, seq_len, d_model)
        m: (batch, seq_len, d_model)

        m is a additive matrix that is added to x before applying softmax
        masked values are set to -inf
        """
        if self.softcap > 0:
            x = x / self.softcap
            x = torch.tanh(x)
            x = x * self.softcap
        return F.softmax(x + m, dim=self.dim)


class CountDecoderHead(nn.Module):
    """
    A decoder head that predicts gene expression counts.

    This module takes gene embeddings and optionally gene token indices as input,
    and outputs predicted expression values using a specified link function.
    """

    def __init__(
        self,
        model_dim: int,  # Dimension of the input embeddings
        link_func: str,  # Type of activation function to use
        eps: float = 0,  # Small epsilon value for numerical stability
        dropout: float = 0.1,  # Dropout rate for regularization
        use_layer_norm: bool = True,  # Whether to use layer normalization
        gene_bias_size: int = 0,  # Size of gene-specific bias embedding
        softcap: int = 0,  # Cap value for softmax activation
    ):
        super().__init__()
        self.seq_dim = model_dim
        self.link_func = link_func  # Store the name for reference
        self.eps = eps
        self.gene_bias_size = gene_bias_size
        self.softcap = softcap

        # Optional gene-specific bias term
        if gene_bias_size > 0:
            self.gene_embedding = nn.Embedding(gene_bias_size, 1)

        # Main MLP for processing gene embeddings
        self.mlp = MLP(
            model_dim,  # Input dimension
            model_dim // 4,  # Hidden dimension (1/4 of input)
            1,  # Output dimension (scalar count)
            num_layers=2,  # Two-layer MLP
            dropout=dropout,
            use_layer_norm=use_layer_norm,
        )

        # Initialize the appropriate activation function based on link_func
        if link_func == "sigmoid":
            self.link_func = nn.Sigmoid()
        elif link_func == "softplus":
            self.link_func = nn.Softplus()
        elif link_func == "relu":
            self.link_func = nn.ReLU()
        elif link_func == "linear":
            self.link_func = nn.Identity()
        elif link_func == "exp":
            self.link_func = torch.exp
        elif link_func == "log":
            self.link_func = torch.log
        elif link_func == "tanh":
            self.link_func = nn.Tanh()
        elif link_func == "softmax":
            # Special case: masked softmax with optional softcap
            self.link_func = MaskedSoftmax(dim=1, softcap=softcap)
        else:
            raise ValueError(f"Invalid link function: {link_func}")

    def forward(
        self,
        gene_output: Tensor = None,  # Gene embeddings from transformer
        gene_tokens: Tensor = None,  # Gene token indices (optional)
        mask: Tensor = None,  # Mask for softmax (optional)
    ) -> Tensor:
        """
        Forward pass to predict gene expression counts.

        Args:
            gene_output: Embeddings from transformer, shape [batch, seq_len, model_dim]
            gene_tokens: Gene token indices, shape [batch, seq_len]
            mask: Mask tensor for softmax, shape [batch, seq_len]

        Returns
        -------
            Predicted gene expression values, shape [batch, seq_len]
        """
        # Initialize logits
        logit = 0

        # Add gene-specific bias if enabled
        if self.gene_bias_size > 0:
            gene_emb = self.gene_embedding(gene_tokens)  # [batch, seq_len, 1]
            logit = logit + gene_emb

        # Process gene embeddings through MLP
        logit = logit + self.mlp(gene_output)  # [batch, seq_len, 1]

        # Remove singleton dimension if needed
        logit = logit.squeeze(-1) if logit.dim() > 2 else logit  # [batch, seq_len]

        # Apply the link function (with mask if using MaskedSoftmax)
        if mask is not None and isinstance(self.link_func, MaskedSoftmax):
            output = self.link_func(logit, mask)
        else:
            output = self.link_func(logit)

        return output

    def init_weights(self):
        """Initialize the gene embedding weights to zero if used."""
        if self.gene_bias_size > 0:
            self.gene_embedding.weight.data.zero_()


class PretrainedEmbeddings(nn.Module):
    def __init__(
        self,
        embedding_matrix,
        output_dim,
        freeze=True,
        normalize=True,
        mlp_hidden_dim=None,
        dropout=0.1,
    ):
        super().__init__()
        self.embedding_dim = embedding_matrix.shape[1]

        # Check embedding_matrix for NaNs
        if torch.isnan(embedding_matrix).any():
            raise ValueError("The embedding matrix contains NaN values.")

        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix.type(torch.float32), freeze=freeze
        )
        if freeze:
            self.embedding.weight.requires_grad = False

        # Add MLP
        if mlp_hidden_dim is None:
            mlp_hidden_dim = output_dim

        self.mlp = MLP(
            self.embedding_dim,
            mlp_hidden_dim,
            output_dim,
            num_layers=2,
            dropout=dropout,
            use_layer_norm=normalize,
        )

    def forward(self, x, embed_only=False):
        embeddings = self.embedding(x.to(next(self.parameters()).device))
        if embed_only:
            return embeddings
        x = self.mlp(embeddings)
        return x


class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers=2,
        dropout=0.1,
        use_layer_norm=True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_layer_norm = use_layer_norm

        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
            elif i == num_layers - 1:
                layers.append(nn.Linear(hidden_dim, output_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))

            if (
                i < num_layers - 1
            ):  # Don't add activation and normalization after the last layer
                layers.append(nn.ReLU())
                if use_layer_norm:
                    layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.Dropout(dropout))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
