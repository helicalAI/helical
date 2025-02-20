from pathlib import Path
from helical.constants.paths import CACHE_DIR_HELICAL
from typing import Literal


class HelixmRNAConfig:
    """HelixmRNAConfig class to store the configuration of the Helix-mRNA model.

    Parameters
    ----------
    batch_size : int, optional, default=10
        The batch size
    device : Literal["cpu", "cuda"], optional, default="cpu"
        The device to use. Either use "cuda" or "cpu".
    max_length : int, optional, default=12288
        The maximum length of the input sequence.
    nproc: int, optional, default=1
        Number of processes to use for data processing.
    """

    def __init__(
        self,
        batch_size: int = 10,
        device: Literal["cpu", "cuda"] = "cpu",
        max_length: int = 12288,
        nproc: int = 1,
    ):

        model_name: Literal["helical-ai/Helix-mRNA"] = "helical-ai/Helix-mRNA"

        self.config = {
            "model_name": model_name,
            "input_size": max_length,
            "batch_size": batch_size,
            "device": device,
            "nproc": nproc,
        }
