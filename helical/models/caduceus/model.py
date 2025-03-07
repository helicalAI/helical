from typing import List, Union
from helical.models.base_models import HelicalDNAModel
from helical.utils.downloader import Downloader
from .caduceus_config import CaduceusConfig
from .pretrained_config import CaduceusPretrainedConfig
from .modeling_caduceus import CaduceusModel
from .caduceus_tokenizer import CaduceusTokenizer
from datasets import Dataset
import torch
from torch.utils.data import DataLoader
import numpy as np
import logging
from pandas import DataFrame

LOGGER = logging.getLogger(__name__)

# default configuration if not specified
configurer = CaduceusConfig()


class Caduceus(HelicalDNAModel):
    """Caduceus model.

    This class represents the Caduceus model, a DNA model using bi-directional and reverse complement DNA properties for better genomic analysis.

    Example
    ----------
    ```python
    from helical.models.caduceus import Caduceus, CaduceusConfig

    caduceus_config = CaduceusConfig(model_name="caduceus-ph-4L-seqlen-1k-d118", batch_size=5)
    caduceus = Caduceus(configurer = caduceus_config)

    sequence = ['ACTG' * int(1024/4)]
    processed_data = caduceus.process_data(sequence)

    embeddings = caduceus.get_embeddings(processed_data)
    print(embeddings.shape)
    ```

    Parameters
    ----------
    configurer : CaduceusConfig, optional, default=configurer
        The model configuration.

    Notes
    ----------
    This model has dependencies which only allow it to be run on CUDA devices.
    The link to the paper can be found [here](https://arxiv.org/abs/2403.03234).
    We make use of the implementation from the [Caduceus](https://github.com/kuleshov-group/caduceus) repository.
    """

    def __init__(self, configurer: CaduceusConfig = configurer):
        super().__init__()

        if torch.cuda.is_available() == False:
            message = "Caduceus requires a CUDA device to run and CUDA is not available"
            LOGGER.error(message)
            raise RuntimeError(message)

        self.configurer = configurer
        self.config = configurer.config
        self.files_config = configurer.files_config
        self.device = self.config["device"]

        downloader = Downloader()
        for file in self.configurer.list_of_files_to_download:
            downloader.download_via_name(file)

        self.pretrained_config = CaduceusPretrainedConfig.from_pretrained(
            self.files_config["model_files_dir"]
        )
        self.model = CaduceusModel.from_pretrained(
            self.files_config["model_files_dir"], config=self.pretrained_config
        )

        self.model.eval()

        self.tokenizer = CaduceusTokenizer(model_max_length=self.config["input_size"])

        LOGGER.info("Caduceus model initialized")

    def _collate_fn(self, batch):
        input_ids = torch.tensor([item["input_ids"] for item in batch])
        batch_dict = {
            "input_ids": input_ids,
        }

        if "labels" in batch[0]:
            batch_dict["labels"] = torch.tensor([item["labels"] for item in batch])

        return batch_dict

    def _pool_hidden_states(self, hidden_states, sequence_length_dim=1):
        """Pools hidden states along sequence length dimension."""
        if (
            self.config["pooling_strategy"] == "mean"
        ):  # Mean pooling along sequence length dimension
            return hidden_states.mean(dim=sequence_length_dim)
        if (
            self.config["pooling_strategy"] == "max"
        ):  # Max pooling along sequence length dimension
            return hidden_states.max(dim=sequence_length_dim).values
        if (
            self.config["pooling_strategy"] == "last"
        ):  # Use embedding of last token in the sequence
            return hidden_states.moveaxis(hidden_states, sequence_length_dim, 0)[
                -1, ...
            ]
        if (
            self.config["pooling_strategy"] == "first"
        ):  # Use embedding of first token in the sequence
            return hidden_states.moveaxis(hidden_states, sequence_length_dim, 0)[0, ...]

    def process_data(
        self,
        sequences: Union[List[str], DataFrame],
        return_tensors: str = "pt",
        padding: str = "max_length",
        truncation: bool = True,
    ) -> Dataset:
        """Process the input DNA sequences.

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
        ----------
        Dataset
            Containing processed DNA sequences.

        """
        LOGGER.info("Processing data for Caduceus.")
        sequences = self.get_valid_dna_sequence(sequences)

        max_length = min(len(max(sequences, key=len)), self.config["input_size"]) + 1

        # tokenized_sequences = []
        # for seq in sequences:
        tokenized_sequences = self.tokenizer(
            sequences,
            return_tensors=return_tensors,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
        )
        # tokenized_sequences.append(tokenized_seq)

        dataset = Dataset.from_dict(tokenized_sequences)
        LOGGER.info("Successfully processed the data for Caduceus.")
        return dataset

    def get_embeddings(self, dataset: Dataset) -> np.ndarray:
        """Get the embeddings for the tokenized sequence.

        Parameters
        ----------
        dataset : Dataset
            The output dataset from `process_data`.

        Returns
        ----------
        np.ndarray
            The embeddings for the tokenized sequence in the form of a numpy array.
            NOTE: This method returns the embeddings using the pooling strategy specified in the config.
        """
        LOGGER.info("Started getting embeddings:")
        dataloader = DataLoader(
            dataset,
            collate_fn=self._collate_fn,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config["nproc"],
        )

        embeddings = []
        self.model.to(self.device)
        self.model.eval()
        for batch in dataloader:
            with torch.no_grad():
                outputs = self.model(input_ids=batch["input_ids"].to(self.device))
                embeddings.append(
                    self._pool_hidden_states(outputs.last_hidden_state).cpu().numpy()
                )

                del batch
                del outputs

        LOGGER.info(f"Finished getting embeddings.")
        return np.vstack(embeddings)
