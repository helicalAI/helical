from helical.models.base_models import HelicalRNAModel
from helical.models.mamba2_mrna.mamba2_mrna_config import Mamba2mRNAConfig
from .mamba2_mrna_tokenizer import CharTokenizer
from datasets import Dataset
from transformers import Mamba2Model, Mamba2Config, BatchEncoding
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from typing import Union
from pandas import DataFrame

import logging

LOGGER = logging.getLogger(__name__)


class Mamba2mRNA(HelicalRNAModel):
    """Mamba2-mRNA Model.

    The Mamba2-mRNA Model is a transformer-based model that can be used to extract mRNA embeddings from mRNA sequences.
    The model is based on the Mamba2 model, which is a transformer-based model trained on mRNA sequences. The model is available through this interface.

    Example
    ----------
    ```python
    from helical.models.mamba2_mrna import Mamba2mRNA, Mamba2mRNAConfig
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_sequences = ["ACUG"*20, "AUGC"*20, "AUGC"*20, "ACUG"*20, "AUUG"*20]

    mamba2_mrna_config = Mamba2mRNAConfig(batch_size=5, device=device)
    mamba2_mrna = Mamba2mRNA(configurer=mamba2_mrna_config)

    processed_input_data = mamba2_mrna.process_data(input_sequences)

    embeddings = mamba2_mrna.get_embeddings(processed_input_data)
    print(embeddings.shape)
    ```


    Parameters
    ----------
    configurer : Mamba2mRNAConfig
        The configuration object for the Mamba2-mRNA model.
    """

    default_configurer = Mamba2mRNAConfig()

    def __init__(self, configurer: Mamba2mRNAConfig = default_configurer):
        super().__init__()
        self.configurer = configurer
        self.config = configurer.config

        self.model = Mamba2Model.from_pretrained(self.config["model_name"])
        self.pretrained_config = Mamba2Config.from_pretrained(
            self.config["model_name"], trust_remote=True
        )
        self.tokenizer = CharTokenizer.from_pretrained(
            self.config["model_name"], trust_remote=True
        )
        self.model.to(self.config["device"])
        self.model.post_init()

        LOGGER.info("Mamba2-mRNA initialized successfully.")
        mode = "training" if self.model.training else "eval"
        LOGGER.info(
            f"'{self.config['model_name']}' model is in '{mode}' mode, on device '{next(self.model.parameters()).device.type}'."
        )

    def process_data(self, sequences: Union[list[str], DataFrame]) -> Dataset:
        """Process the mRNA sequences and return a Dataset object.

        Parameters
        ----------
        sequences : str or DataFrame
            The mRNA sequences. If a DataFrame is provided, it should have a column named "Sequence".

        Returns
        -------
        Dataset
            The dataset object.
        """
        LOGGER.info(f"Processing data for Mamba2-mRNA.")
        sequences = self.get_valid_rna_sequence(sequences)

        tokenized_sequences = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.config["input_size"],
            return_special_tokens_mask=True,
        )

        dataset = Dataset.from_dict(tokenized_sequences)

        LOGGER.info(f"Successfully preprocessed the data for Mamba2-mRNA.")
        return dataset

    def get_embeddings(self, dataset: Dataset) -> np.ndarray:
        """Get the embeddings for the mRNA sequences.

        Parameters
        ----------
        dataset : Dataset
            The dataset object.

        Returns
        -------
        np.ndarray
            The embeddings array.
        """
        LOGGER.info(f"Started getting embeddings:")
        dataloader = DataLoader(
            dataset,
            collate_fn=self._collate_fn,
            batch_size=self.config["batch_size"],
            shuffle=False,
        )
        embeddings = []

        progress_bar = tqdm(dataloader, desc="Getting embeddings")
        with torch.no_grad():
            for batch in progress_bar:
                input_ids = batch["input_ids"].to(self.config["device"])
                special_tokens_mask = batch["special_tokens_mask"].to(
                    self.config["device"]
                )
                attention_mask = batch["attention_mask"].to(self.config["device"])

                output = self.model(
                    input_ids,
                    special_tokens_mask=special_tokens_mask,
                    attention_mask=attention_mask,
                )

                last_hidden_states = output[0]

                if input_ids is not None:
                    batch_size, _ = input_ids.shape[:2]

                if self.pretrained_config.pad_token_id is None and batch_size > 1:
                    message = (
                        "Cannot handle batch sizes > 1 if no padding token is defined."
                    )
                    LOGGER.error(message)
                    raise ValueError(message)

                if self.pretrained_config.pad_token_id is None:
                    sequence_lengths = -1
                else:
                    if input_ids is not None:
                        sequence_lengths = (
                            torch.eq(input_ids, self.pretrained_config.pad_token_id)
                            .int()
                            .argmax(-1)
                            - 1
                        )
                        sequence_lengths = sequence_lengths % input_ids.shape[-1]
                        sequence_lengths = sequence_lengths.to(
                            last_hidden_states.device
                        )
                    else:
                        sequence_lengths = -1

                pooled_last_hidden_states = last_hidden_states[
                    torch.arange(batch_size, device=last_hidden_states.device),
                    sequence_lengths,
                ]

                # Take second last element from the output as last element is a special token
                embeddings.append(pooled_last_hidden_states.cpu().numpy())

                del batch
                del output

        LOGGER.info(f"Finished getting embeddings.")
        return np.concatenate(embeddings)

    def _collate_fn(self, batch):
        input_ids = torch.tensor([item["input_ids"] for item in batch])
        special_tokens_mask = torch.tensor(
            [item["special_tokens_mask"] for item in batch]
        )

        batch_dict = {
            "input_ids": input_ids,
            "special_tokens_mask": special_tokens_mask,
            "attention_mask": 1 - special_tokens_mask,
        }

        if "labels" in batch[0]:
            batch_dict["labels"] = torch.tensor([item["labels"] for item in batch])

        return BatchEncoding(batch_dict)
