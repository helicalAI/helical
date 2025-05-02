"""Dataclasses for the Transcripformer model."""

from dataclasses import dataclass, field
import numpy as np
import torch

# Set float32 matmul precision for better performance with Tensor Cores
torch.set_float32_matmul_precision("high")


# Parameters that control the model architecture
@dataclass
class ModelConfig:
    """Configuration for model architecture.

    Parameters
    ----------
        log_counts_eps (float): Epsilon for log counts (default: 1e-6)
        num_heads (int): Number of attention heads (default: 16)
        num_layers (int): Number of transformer layers (default: 12)
        model_dim (int): Model dimension (default: 2048)
        embed_dim (int): Embedding dimension (default: 2048)
        dropout (float): Dropout rate (default: 0.1)
        activation (str): Activation function (default: "gelu")
        attn_bias (bool): Attention bias flag (default: false)
        fw_bias (bool): Forward bias flag (default: false)
        mu_link_fn (str): Mu link function (default: "softmax")
        softcap (int): Soft cap value (default: 10)
        seq_len (int): Gene sequence length (default: 2047)
        aux_len (int): Auxiliary sequence length (default: 1)
        block_len (int): Block length for Flex attention (default: 128)
        use_aux (bool): Use auxiliary inputs flag (default: false)
        gene_head_hidden_dim (int): Gene head hidden dimension (default: 2048)
    """

    log_counts_eps: float
    num_heads: int
    num_layers: int
    model_dim: int
    embed_dim: int
    dropout: float
    activation: str
    attn_bias: bool
    fw_bias: bool
    mu_link_fn: str
    softcap: int
    seq_len: int
    aux_len: int
    block_len: int

    # Optional fields
    gene_head_hidden_dim: int = 2048

    def __post_init__(self):
        if (self.seq_len + self.aux_len) % self.block_len != 0:
            raise ValueError(
                "Sum of sequence length and auxiliary length must be divisible by block length"
            )
        if self.mu_link_fn not in {
            "exp",
            "log",
            "relu",
            "sigmoid",
            "softplus",
            "linear",
            "softmax",
        }:
            raise ValueError(
                "Mu link function must be one of ['exp', 'log', 'relu', 'sigmoid', 'softplus', 'linear', 'softmax']"
            )


# Parameters that control how the input data is processed
@dataclass
class DatasetConfig:
    """Configuration for individual datasets.

    Parameters
    ----------
        files (str): Path to dataset files
        weight (float): Dataset weight in training (default: 1.0)
        name (str): Dataset name
    """

    files: str
    weight: float = 1.0
    name: str = None


@dataclass
class DataConfig:
    """Configuration for data processing and loading.

    Parameters
    ----------
        aux_vocab_path (str): Path to auxiliary vocabulary file
        pretrained_emb_path (str): Path to pretrained embeddings
        pin_memory (bool): Enable pinned memory for data loading
        aux_cols (list): Auxiliary columns to use (default: "assay")
        gene_col_name (str): Column name for gene IDs (default: "ensembl_id")
        clip_counts (bool): Maximum count value clipping
        filter_to_vocabs (bool): Filter genes to only those in vocabulary
        filter_outliers (bool): Gene outlier filtering threshold
        pad_zeros (bool): Enable padding of gene zero counts
        normalize_to_scale (bool): Normalization counts to scale
        val_size (int): Validation set size
        n_data_workers (int): Number of data loading workers (default: 8)
        sort_genes (bool): Sort genes by expression values
        randomize_genes (bool): Randomize gene order in each cell
        min_expressed_genes (bool): Minimum number of expressed genes in a cell
        gene_pad_token (str): Padding token for genes
        aux_pad_token (str): Auxiliary padding token
        add_cell_token (bool): Add cell token (default: false)
        train_datasets (List[DatasetConfig]): List of training datasets
        val_datasets (List[DatasetConfig]): List of validation datasets
        esm2_mappings (List[str]): ESM2 mapping paths
        special_tokens (List[str]): Special tokens
        esm2_mappings_path (str): Path to ESM2 mappings
        mproc_context (str): Multiprocessing context
    """

    # Required fields
    aux_vocab_path: str
    pin_memory: bool
    aux_cols: list
    gene_col_name: str
    clip_counts: bool
    filter_to_vocabs: bool
    filter_outliers: bool
    pad_zeros: bool
    normalize_to_scale: bool
    n_data_workers: int
    sort_genes: bool
    randomize_genes: bool
    min_expressed_genes: bool
    gene_pad_token: str
    aux_pad_token: str

    # Optional fields with None defaults
    esm2_mappings: list[str] | None = None
    special_tokens: list[str] | None = None
    esm2_mappings_path: str | None = None


# Parameters that control the loss function
@dataclass
class LossConfig:
    """Configuration for loss functions.

    Parameters
    ----------
        gene_id_loss_weight (float): Gene ID prediction loss weight (default: 1.0)
    """

    gene_id_loss_weight: float
    softplus_approx: bool = True


# Parameters that control the inference mode
@dataclass
class InferenceConfig:
    """Configuration for inference mode.

    Parameters
    ----------
        pred_mode (str): Prediction mode
        output_keys (list): Output keys to save
        batch_size (int): Batch size for inference
        obs_keys (list): Observation keys to pass through
        data_files (list): Data files for inference
        num_nodes (int): Number of nodes
        load_checkpoint (str): Path to checkpoint to load
        output_path (str): Path to save outputs
        num_gpus_per_node (int): GPUs per node (default: 1)
        special_tokens (list): Special tokens to use
    """

    output_keys: list
    batch_size: int
    obs_keys: list
    data_files: list | None
    load_checkpoint: str | None
    output_path: str | None
    num_gpus_per_node: int = 1
    num_nodes: int = 1
    precision: str = "16-mixed"
    special_tokens: list = field(default_factory=list)
    pretrained_embedding: list = field(default_factory=list)


@dataclass
class GeneVocab:
    vocab_dict: dict  # Path to dictionary of gene token mappings
    vocab_size: int
    pad_idx: int
    start_idx: int
    end_idx: int
    cell_idx: int
    embedding_matrix: np.ndarray = field(default=None)


@dataclass
class AuxVocab:
    vocab_dict: dict
    vocab_size: int
    pad_ids: list
    aux_seq_len: int


@dataclass
class BatchData:
    gene_counts: torch.Tensor
    gene_token_indices: torch.Tensor
    aux_token_indices: torch.Tensor | None = None
    file_path: str | None = None
    obs: dict[str, np.ndarray] | None = None
