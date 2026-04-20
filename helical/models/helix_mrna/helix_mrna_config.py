from pathlib import Path
from helical.constants.paths import CACHE_DIR_HELICAL
from typing import Literal


class HelixmRNAConfig:
    """HelixmRNAConfig class to store the configuration of the Helix-mRNA model.

    Parameters
    ----------
    batch_size : int, optional, default=10
        The batch size
    device : str, optional, default="cpu"
        The device to use. Accepts any string torch.device accepts, e.g. "cpu",
        "cuda", "cuda:0".
    max_length : int, optional, default=12288
        The maximum length of the input sequence.
    nproc: int, optional, default=1
        Number of processes to use for data processing.
    output_attentions : bool, optional, default=False
        Whether to return attention weights from get_embeddings. Must be set at
        construction time: True forces eager attention (required for attention
        output), False uses flash_attention_2 when available, else sdpa. Note:
        eager attention materialises the full O(seq²) matrix and may OOM on long
        sequences or large batches.
    """

    def __init__(
        self,
        batch_size: int = 10,
        device: str = "cpu",
        max_length: int = 12288,
        nproc: int = 1,
        output_attentions: bool = False,
    ):

        model_name: Literal["helical-ai/Helix-mRNA"] = "helical-ai/Helix-mRNA"

        self.config = {
            "model_name": model_name,
            "input_size": max_length,
            "batch_size": batch_size,
            "device": device,
            "nproc": nproc,
            "output_attentions": output_attentions,
        }
