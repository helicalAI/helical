import logging
from helical.models.hyena_dna.hyena_dna_config import HyenaDNAConfig
from helical.models.base_models import HelicalDNAModel
from tqdm import tqdm
from datasets import Dataset
from helical.models.hyena_dna.pretrained_model import HyenaDNAPreTrainedModel
import torch
from .standalone_hyenadna import CharacterTokenizer
from helical.utils.downloader import Downloader
from torch.utils.data import DataLoader
import numpy as np
from typing import Union
from pandas import DataFrame

LOGGER = logging.getLogger(__name__)


class HyenaDNA(HelicalDNAModel):
    """HyenaDNA model.
    This class represents the HyenaDNA model, which is a long-range genomic foundation model pretrained on context lengths of up to 1 million tokens at single nucleotide resolution.

    Example
    -------
    ```python
    from helical.models.hyena_dna import HyenaDNA, HyenaDNAConfig

    hyena_config = HyenaDNAConfig(model_name = "hyenadna-tiny-1k-seqlen-d256")
    model = HyenaDNA(configurer = hyena_config)

    sequence = 'ACTG' * int(1024/4)

    tokenized_sequence = model.process_data(sequence)
    embeddings = model.get_embeddings(tokenized_sequence)

    print(embeddings.shape)
    ```

    Parameters
    ----------
    configurer : HyenaDNAConfig, optional, default=default_configurer
        The model configuration.

    Notes
    -----
    The link to the paper can be found [here](https://arxiv.org/abs/2306.15794.
    We use the implementation from the [HyenaDNA](https://github.com/HazyResearch/hyena-dna) repository.
    """

    default_configurer = HyenaDNAConfig()

    def __init__(self, configurer: HyenaDNAConfig = default_configurer) -> None:
        super().__init__()
        self.config = configurer.config

        downloader = Downloader()
        for file in self.config["list_of_files_to_download"]:
            downloader.download_via_name(file)

        self.model = HyenaDNAPreTrainedModel().from_pretrained(self.config)

        # create tokenizer
        self.tokenizer = CharacterTokenizer(
            characters=["A", "C", "G", "T", "N"],  # add DNA characters, N is uncertain
            model_max_length=self.config["max_length"]
            + 2,  # to account for special tokens, like EOS
            # add_special_tokens=False,  # we handle special tokens elsewhere
            padding_side="left",  # since HyenaDNA is causal, we pad on the left
        )

        # prep model and forward
        self.device = self.config["device"]
        self.model.to(self.device)
        self.model.eval()

        LOGGER.info(f"Model finished initializing.")
        mode = "training" if self.model.training else "eval"
        LOGGER.info(
            f"'{self.config['model_name']}' model is in '{mode}' mode, on device '{next(self.model.parameters()).device.type}'."
        )

    def process_data(
        self,
        sequences: Union[list[str], DataFrame],
        return_tensors: str = "pt",
        padding: str = "max_length",
        truncation: bool = True,
    ) -> Dataset:
        """Process the input DNA sequence.

        Parameters
        ----------
        sequences : list[str] or DataFrame
            The input DNA sequences to be processed. If a DataFrame is provided, it should have a column named 'Sequence'.
        return_tensors : str, optional, default="pt"
            The return type of the processed data.
        padding : str, optional, default="max_length"
            The padding strategy to be used.
        truncation : bool, optional, default=True
            Whether to truncate the sequences or not.

        Returns
        -------
        Dataset
            Containing processed DNA sequences.
        """
        LOGGER.info("Processing data for HyenaDNA.")
        sequences = self.get_valid_dna_sequence(sequences)

        max_length = (
            len(max(sequences, key=len)) + 2
        )  # +2 for special tokens at the beginning and end of sequences

        tokenized_sequences = self.tokenizer(
            sequences,
            return_tensors=return_tensors,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
        )

        dataset = Dataset.from_dict(tokenized_sequences)
        LOGGER.info(f"Succesfully prepared the HyenaDNA Dataset.")
        return dataset

    def get_embeddings(self, dataset: Dataset) -> torch.Tensor:
        """Get the embeddings for the tokenized sequence.

        Parameters
        ----------
        dataset : Dataset
            The output dataset from `process_data`.

        Returns
        ----------
        np.ndarray
            The embeddings for the tokenized sequence in the form of a numpy array.

        """
        LOGGER.info(f"Started getting embeddings:")

        train_data_loader = DataLoader(
            dataset, collate_fn=self._collate_fn, batch_size=self.config["batch_size"]
        )
        with torch.inference_mode():
            embeddings = []
            for batch in tqdm(train_data_loader, desc="Getting embeddings"):
                input_data = batch["input_ids"].to(self.device)
                embeddings.append(self.model(input_data).detach().cpu().numpy())

        LOGGER.info(f"Finished getting embeddings.")
        return np.vstack(embeddings)

    def _collate_fn(self, batch):
        input_ids = torch.tensor([item["input_ids"] for item in batch])
        batch_dict = {
            "input_ids": input_ids,
        }

        if "labels" in batch[0]:
            batch_dict["labels"] = torch.tensor([item["labels"] for item in batch])

        return batch_dict
