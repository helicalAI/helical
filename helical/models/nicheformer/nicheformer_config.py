from helical.constants.paths import CACHE_DIR_HELICAL
from pathlib import Path
from typing import Literal, Optional, Union
import numpy as np
import logging

LOGGER = logging.getLogger(__name__)

_DEFAULT_MODEL_NAME = "theislab/Nicheformer"

# All files required to load the model and tokenizer from a local directory.
_MODEL_FILES = [
    "config.json",
    "vocab.json",
    "model.safetensors",
    "model.h5ad",
    "modeling_nicheformer.py",
    "tokenization_nicheformer.py",
    "configuration_nicheformer.py",
    "masking.py",
    "__init__.py",
]


class NicheformerConfig:
    """Configuration class to use the Nicheformer Model.

    Parameters
    ----------
    model_name : str, optional, default="theislab/Nicheformer"
        The HuggingFace repository ID to load the model and tokenizer from.
    batch_size : int, optional, default=32
        The batch size used during embedding extraction.
    device : Literal["cpu", "cuda"], optional, default="cpu"
        The device to use for inference.
    layer : int, optional, default=-1
        The transformer layer from which to extract embeddings. ``-1`` selects
        the last layer.
    with_context : bool, optional, default=False
        Whether to include context tokens (species, technology, modality) when
        computing the mean-pooled cell embedding.
    technology_mean : str or np.ndarray or None, optional, default=None
        Per-gene technology mean used for additional normalisation inside the
        tokenizer. Accepts either a path to a ``.npy`` file or a NumPy array
        of shape ``(n_genes,)``. When ``None`` the step is skipped.

    Returns
    -------
    NicheformerConfig
        The Nicheformer configuration object.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL_NAME,
        batch_size: int = 32,
        device: Literal["cpu", "cuda"] = "cpu",
        layer: int = -1,
        with_context: bool = False,
        technology_mean: Optional[Union[str, np.ndarray]] = None,
    ):
        hf_base_url = f"https://huggingface.co/{model_name}/resolve/main"
        self.model_dir = Path(CACHE_DIR_HELICAL, "nicheformer")

        self.list_of_files_to_download = [
            (self.model_dir / filename, f"{hf_base_url}/{filename}")
            for filename in _MODEL_FILES
        ]

        self.files_config = {
            "model_files_dir": self.model_dir,
        }

        self.config = {
            "model_name": model_name,
            "batch_size": batch_size,
            "device": device,
            "layer": layer,
            "with_context": with_context,
            "technology_mean": technology_mean,
        }
