from helical.models.base_models import HelicalRNAModel
from helical.models.helix_mrna.helix_mrna_config import HelixmRNAConfig
from datasets import Dataset
from transformers import BatchEncoding
from .modeling_helix_mrna import HelixmRNAPretrainedModel
from .helix_mrna_tokenizer import CharTokenizer
from .helix_mrna_pretrained_config import HelixmRNAPretrainedConfig
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from typing import Union
from pandas import DataFrame
from helical.utils.attn_backend import select_attn_backend

import logging

LOGGER = logging.getLogger(__name__)


class HelixmRNA(HelicalRNAModel):
    """Helix-mRNA Model.

    The Helix-mRNA Model is a transformer-based model that can be used to extract mRNA embeddings from mRNA sequences.
    The model is based on the Mamba2 model, which is a transformer-based model trained on mRNA sequences. The model is available through this interface.

    Example
    -------
    ```python
    from helical.models.helix_mrna import HelixmRNA, HelixmRNAConfig
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    helix_mrna_config = HelixmRNAConfig(batch_size=5, max_length=100, device=device)
    helix_mrna = HelixmRNA(configurer=helix_mrna_config)

    rna_sequences = ["EACUEGGG", "EACUEGGG", "EACUEGGG", "EACUEGGG", "EACUEGGG"]
    dataset = helix_mrna.process_data(rna_sequences)

    rna_embeddings = helix_mrna.get_embeddings(dataset)
    print("Helix_mRNA embeddings shape: ", rna_embeddings.shape)
    ```

    Parameters
    ----------
    configurer : HelixmRNAConfig
        The configuration object for the Helix-mRNA model.

    Notes
    ----------
    Helix_mRNA was trained using a character in between each codon of the mRNA sequence.
    This is done to ensure that the model can learn the structure of the mRNA sequence.
    Although it can take a standard RNA sequence as input, it is recommended to add the letter E between each codon of the mRNA sequence to get better embeddings.
    """

    default_configurer = HelixmRNAConfig()

    def __init__(self, configurer: HelixmRNAConfig = default_configurer):
        super().__init__()
        self.configurer = configurer
        self.config = configurer.config

        attn_impl, model_dtype = select_attn_backend(
            self.config["device"],
            output_attentions=self.config.get("output_attentions", False),
        )
        if attn_impl == "flash_attention_2":
            LOGGER.warning(
                "Loading Helix-mRNA in bfloat16 for flash_attention_2 compatibility."
            )
        self.model = HelixmRNAPretrainedModel.from_pretrained(
            self.config["model_name"],
            attn_implementation=attn_impl,
            torch_dtype=model_dtype,
        )
        self.pretrained_config = HelixmRNAPretrainedConfig.from_pretrained(
            self.config["model_name"], trust_remote=True
        )
        self.tokenizer = CharTokenizer.from_pretrained(
            self.config["model_name"], trust_remote=True
        )
        self.model.to(self.config["device"])

        LOGGER.info("Helix-mRNA initialized successfully.")
        mode = "training" if self.model.training else "eval"
        LOGGER.info(
            f"'{self.config['model_name']}' model is in '{mode}' mode, on device '{next(self.model.parameters()).device.type}'."
        )

    def process_data(self, sequences: Union[list[str], DataFrame]) -> Dataset:
        """Process the mRNA sequences and return a Dataset object.

        Parameters
        ----------
        sequences : list[str] or DataFrame
            The mRNA sequences. If a DataFrame is provided, it should have a column named 'Sequence'.

        Returns
        -------
        Dataset
            The dataset object.
        """
        LOGGER.info(f"Processing data for Helix-mRNA.")
        sequences = self.get_valid_rna_sequence(sequences)

        tokenized_sequences = self.tokenizer(
            sequences,
            return_tensors="pt",
            truncation=True,
            padding="longest",
            max_length=self.config["input_size"],
            return_special_tokens_mask=True,
        )

        dataset = Dataset.from_dict(tokenized_sequences)

        LOGGER.info("Successfully processed the data for Helix-mRNA.")
        return dataset

    def get_embeddings(
        self, dataset: Dataset, output_attentions: bool = False
    ) -> np.ndarray:
        """Get the embeddings for the mRNA sequences.

        Parameters
        ----------
        dataset : HelixmRNADataset
            The dataset object.
        output_attentions : bool, optional, default=False
            Whether to also return the last-layer attention weights. Requires the
            model to have been loaded with eager attention (set
            ``output_attentions=True`` in ``HelixmRNAConfig``).

        Returns
        -------
        np.ndarray
            The embeddings array.
        list, optional
            Per-batch last-layer attention tensors of shape
            ``(batch_size, num_heads, seq_length, seq_length)``. Only returned
            when ``output_attentions=True``.
        """
        if output_attentions and not self.config.get("output_attentions", False):
            raise ValueError(
                "output_attentions=True requires the model to be loaded with eager attention. "
                "Set output_attentions=True in HelixmRNAConfig before instantiating the model."
            )
        LOGGER.info("Started getting embeddings:")
        dataloader = DataLoader(
            dataset,
            collate_fn=self._collate_fn,
            batch_size=self.config["batch_size"],
            shuffle=False,
        )
        embeddings = []
        attentions = [] if output_attentions else None

        progress_bar = tqdm(dataloader, desc="Getting embeddings")
        with torch.no_grad():
            for batch in progress_bar:
                input_ids = batch["input_ids"].to(self.config["device"])
                special_tokens_mask = batch["special_tokens_mask"].to(
                    self.config["device"]
                )
                attention_mask = 1 - special_tokens_mask

                output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=output_attentions,
                )

                last_hidden_states = output[0]
                embeddings.append(last_hidden_states.cpu().float().numpy())

                if output_attentions:
                    output_attn = getattr(output, "attentions", None)
                    if output_attn:
                        attentions.append(output_attn[-1].cpu().float().numpy())

                del batch
                del output

        LOGGER.info(f"Finished getting embeddings.")
        if output_attentions:
            return np.concatenate(embeddings), (attentions if attentions else None)
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
