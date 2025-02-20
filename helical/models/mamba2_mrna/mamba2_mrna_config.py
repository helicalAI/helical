from typing import Literal
import logging

LOGGER = logging.getLogger(__name__)


class Mamba2mRNAConfig:
    """Mamba2-mRNA Config class to store the configuration of the Mamba2-mRNA model.

    Parameters
    ----------
    batch_size : int, optional, default=10
        The batch size
    device : Literal["cpu", "cuda"], optional, default="cpu"
        The device to use. Either use "cuda" or "cpu".
    max_length : int, optional, default=8192
        The maximum length of the input sequence. It is recommeded to set this to the longest sequence in the dataset.
    nproc: int, optional, default=1
        Number of processes to use for data processing.
    """

    def __init__(
        self,
        batch_size: int = 10,
        device: Literal["cpu", "cuda"] = "cpu",
        max_length: int = 8192,
        nproc: int = 1,
    ):

        model_name = "helical-ai/mamba2-mRNA"

        self.config = {
            "model_name": model_name,
            "batch_size": batch_size,
            "input_size": max_length,
            "device": device,
            "nproc": nproc,
        }
