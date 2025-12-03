from typing import Literal, Optional
from helical.constants.paths import CACHE_DIR_HELICAL
from pathlib import Path
import logging

LOGGER = logging.getLogger(__name__)

class TahoeConfig:
    """Configuration class to use the Tahoe-1x Model.

    Parameters
    ----------
    model_size : Literal["70m", "1b", "3b"], default="70m"
        The size of the model to use. Options are:
        - "70m": 12-layer transformer with 512 embedding dimensions
        - "1b": Larger model variant (1 billion parameters)
        - "3b": Largest model variant (3 billion parameters)
    batch_size : int, optional, default=8
        The batch size for inference.
    emb_mode : Literal["cell", "gene"], optional, default="cell"
        The embedding mode to use:
        - "cell": Returns cell-level embeddings (mean-pooled across genes)
        - "gene": Returns gene-level embeddings for each gene token
    device : Literal["cpu", "cuda"], optional, default="cpu"
        The device to use. Either use "cuda" or "cpu".
    attn_impl : Literal["flash", "torch"], optional, default="flash"
        The attention implementation to use:
        - "flash": Uses Flash Attention for speed and memory efficiency (doesn't support attention output)
        - "torch": Uses standard PyTorch attention (supports attention output but slower)
    max_length : int, optional, default=2048
        The maximum sequence length for tokenization.
    num_workers : int, optional, default=8
        Number of workers for data loading.
    prefetch_factor : int, optional, default=48
        Number of batches to prefetch per worker.
    hf_repo_id : str, optional, default="tahoebio/Tahoe-x1"
        The Hugging Face repository ID to load the model from.

    Returns
    -------
    TahoeConfig
        The Tahoe configuration object

    Notes
    -----
    The Tahoe-1x model is a foundation model for single-cell RNA-seq data that uses
    a transformer architecture to learn representations of cellular states. The model
    accepts raw count data and produces embeddings for cells and optionally genes.
    """

    def __init__(
        self,
        model_size: Literal["70m", "1b", "3b"] = "70m",
        batch_size: int = 8,
        emb_mode: Literal["cell", "gene"] = "cell",
        device: Literal["cpu", "cuda"] = "cpu",
        attn_impl: Literal["flash", "torch"] = "flash",
        max_length: int = 2048,
        num_workers: int = 8,
        prefetch_factor: int = 48,
        hf_repo_id: str = "tahoebio/Tahoe-x1",
    ):
        # Model size specifications
        self.model_map = {
            "70m": {
                "n_layers": 12,
                "d_model": 512,
            },
            "1b": {
                "n_layers": 24,
                "d_model": 1024,
            },
            "3b": {
                "n_layers": 36,
                "d_model": 1536,
            },
        }

        if model_size not in self.model_map:
            raise ValueError(
                f"Model size {model_size} not found in available models: {list(self.model_map.keys())}"
            )

        self.model_dir = Path(CACHE_DIR_HELICAL, "tahoe", model_size)

        self.config = {
            "model_size": model_size,
            "batch_size": batch_size,
            "emb_mode": emb_mode,
            "device": device,
            "attn_impl": attn_impl,
            "max_length": max_length,
            "num_workers": num_workers,
            "prefetch_factor": prefetch_factor,
            "hf_repo_id": hf_repo_id,
            "d_model": self.model_map[model_size]["d_model"],
            "n_layers": self.model_map[model_size]["n_layers"],
        }
