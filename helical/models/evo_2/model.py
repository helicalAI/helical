from functools import partial
from helical.models.base_models import HelicalDNAModel
from helical.models.evo_2 import Evo2Config
from datasets import Dataset
from .evo2_tokenizer import CharLevelTokenizer

import huggingface_hub
from huggingface_hub import hf_hub_download, constants
import os

import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple, Union
from pandas import DataFrame

from .scoring import score_sequences, score_sequences_rc

from vortex.model.generation import generate as vortex_generate
from vortex.model.model import StripedHyena
from vortex.model.utils import dotdict, load_checkpoint

import logging

LOGGER = logging.getLogger(__name__)


class Evo2(HelicalDNAModel):
    """Evo 2 Model.

    Example
    -------
    ```python
    from helical import Evo2, Evo2Config

    evo2_config = Evo2Config(batch_size=1)

    evo2 = Evo2(configurer=evo2_config)

    sequences = ["ACGT" * 1000]

    dataset = evo2.process_data(sequences)

    embeddings = evo2.get_embeddings(dataset)

    generate = evo2.generate(sequences)

    print(generate)
    ```

    Parameters
    ----------
    configurer : Evo2Config
        The configuration object for the Evo 2 model.
    """

    default_configurer = Evo2Config()

    def __init__(self, configurer: Evo2Config = default_configurer):
        super().__init__()
        self.configurer = configurer
        self.config = configurer.config

        self.tokenizer = CharLevelTokenizer(self.config["model_map"]["vocab_size"])

        self.model = self._load_evo2_model("evo2_7b")

        LOGGER.info("Evo 2 initialized successfully.")
        mode = "training" if self.model.training else "eval"

        LOGGER.info(
            f"'{self.config['model_map']['model_name']}' model is in '{mode}' mode, on device '{next(self.model.parameters()).device.type}'."
        )

    def process_data(self, sequences: Union[list[str], DataFrame]) -> Dataset:
        """Process the mRNA sequences and return a Dataset object.

        Parameters
        ----------
        sequences : list[str] or DataFrame
            The DNA sequences. If a DataFrame is provided, it should have a column named 'Sequence'.

        Returns
        -------
        Dataset
            The dataset object.
        """
        LOGGER.info(f"Processing data for Evo 2.")
        sequences = self.get_valid_dna_sequence(sequences, enforce_characters=False)

        tokenized_sequences = {}

        tokenized_sequences["input_ids"] = sequences

        dataset = Dataset.from_dict(tokenized_sequences)

        LOGGER.info("Successfully processed the data for Evo 2.")
        return dataset

    def get_embeddings(
        self, dataset: Dataset, embedding_layer: str = None
    ) -> np.ndarray:
        """Get the embeddings for the mRNA sequences.

        Parameters
        ----------
        dataset : Evo2Dataset
            The dataset object.

        Returns
        -------
        np.ndarray
            The embeddings array.
        """
        LOGGER.info("Started getting embeddings:")
        dataloader = DataLoader(
            dataset,
            collate_fn=self._collate_fn,
            batch_size=self.config["batch_size"],
            shuffle=False,
        )
        embeddings = []

        if embedding_layer is None:
            embedding_layer = self.config["model_map"]["default_embedding_layer"]

        progress_bar = tqdm(dataloader, desc="Getting embeddings")
        with torch.no_grad():
            for batch in progress_bar:
                input_ids = batch["input_ids"].to(self.config["device"])

                _, layer_embedding = self(
                    input_ids, return_embeddings=True, layer_names=[embedding_layer]
                )

                embeddings.append(
                    layer_embedding[embedding_layer].float().cpu().numpy()
                )

                del batch
                del layer_embedding

        LOGGER.info(f"Finished getting embeddings.")
        return np.concatenate(embeddings)

    def _collate_fn(self, batch):

        input_ids = [item["input_ids"] for item in batch]

        max_len = max(len(ids) for ids in input_ids)

        input_ids = torch.tensor(
            [
                self.tokenizer.tokenize(ids)
                + [self.tokenizer.pad_id] * (max_len - len(ids))
                for ids in input_ids
            ],
            dtype=torch.int,
        )

        batch_dict = {
            "input_ids": input_ids,
        }

        if "labels" in batch[0]:
            batch_dict["labels"] = torch.tensor([item["labels"] for item in batch])

        return batch_dict

    def forward(
        self,
        input_ids: torch.Tensor,
        return_embeddings: bool = False,
        layer_names=None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with optional embedding extraction.

        Args:
            input_ids: Input token IDs
            return_embeddings: If True, returns embeddings from specified layers
            layer_names: List of layer names to extract embeddings from. Required if
                return_embeddings=True

        Returns:
            Tuple of (logits, embeddings_dict) if return_embeddings=True
            Tuple of (logits, None) otherwise
        """
        embeddings = {}
        handles = []

        if return_embeddings:
            if layer_names is None:
                raise ValueError(
                    "layer_names must be specified when return_embeddings=True. Look at "
                    "evo2_model.model.state_dict().keys() to see available layers."
                )

            def hook_fn(layer_name):
                def hook(_, __, output):
                    if isinstance(output, tuple):
                        output = output[0]
                    embeddings[layer_name] = output.detach()

                return hook

            # Register hooks for requested layers
            for name in layer_names:
                layer = self.model.get_submodule(name)
                handles.append(layer.register_forward_hook(hook_fn(name)))

        try:
            # Original forward pass
            with torch.no_grad():
                logits = self.model.forward(input_ids)

            if return_embeddings:
                return logits, embeddings
            return logits, None

        finally:
            for handle in handles:
                handle.remove()

    def __call__(self, input_ids, return_embeddings=False, layer_names=None):
        return self.forward(input_ids, return_embeddings, layer_names)

    def score_sequences(
        self,
        seqs: List[str],
        batch_size: int = 1,
        prepend_bos: bool = False,
        reduce_method: str = "mean",
        average_reverse_complement: bool = False,
    ) -> List[float]:
        scoring_func = partial(
            score_sequences_rc if average_reverse_complement else score_sequences,
            model=self.model,
            tokenizer=self.tokenizer,
            batch_size=batch_size,
            prepend_bos=prepend_bos,
            reduce_method=reduce_method,
        )

        with torch.no_grad():
            try:
                scores = scoring_func(seqs)
            except Exception as e:
                raise RuntimeError(f"Error during sequence scoring: {str(e)}") from e

        return scores

    def generate(
        self,
        prompt_seqs: List[str],
        n_tokens: int = 500,
        temperature: float = 1.0,
        top_k: int = 4,
        top_p: float = 1.0,
        batched: bool = True,
        cached_generation: bool = True,
        verbose: int = 1,
        force_prompt_threshold: int = None,
    ) -> Tuple[List[str], List[float]]:
        """
        Generate sequences from a list of prompts.

        force_prompt_threshold: If specified, avoids OOM errors through teacher forcing if the prompt is longer than this threshold.

        If force_prompt_threshold is none, sets default assuming 1xH100 (evo2_7b) and 2xH100 (evo2_40b) to help avoid OOM errors.
        """

        with torch.no_grad():
            output = vortex_generate(
                prompt_seqs=prompt_seqs,
                model=self.model,
                tokenizer=self.tokenizer,
                n_tokens=n_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                batched=batched,
                cached_generation=cached_generation,
                verbose=verbose,
                force_prompt_threshold=force_prompt_threshold,
            )
            return output

    def _load_evo2_model(self, model_name: str = None):
        """
        Load HuggingFace checkpoint using StripedHyena 2.
        """

        hf_model_name = self.config["model_map"]["model_hf_name"]
        filename = f"{model_name}.pt"

        # First try normal download
        try:
            weights_path = hf_hub_download(
                repo_id=hf_model_name,
                filename=filename,
            )
        # If file is split, download and join parts
        except:
            print(f"Loading checkpoint shards for {filename}")
            # If file is split, get the first part's directory to use the same cache location
            weights_path = os.path.join(
                os.path.dirname(constants.HF_HUB_CACHE), filename
            )
            if os.path.exists(weights_path):
                print(f"Found {filename}")
            else:
                # Download and join parts
                parts = []
                part_num = 0
                while True:
                    try:
                        part_path = hf_hub_download(
                            repo_id=hf_model_name, filename=f"{filename}.part{part_num}"
                        )
                        parts.append(part_path)
                        part_num += 1
                    except huggingface_hub.errors.EntryNotFoundError:
                        break

                # Join in the same directory
                with open(weights_path, "wb") as outfile:
                    for part in parts:
                        with open(part, "rb") as infile:
                            while True:
                                chunk = infile.read(8192 * 1024)
                                if not chunk:
                                    break
                                outfile.write(chunk)

                # Cleaning up the parts
                for part in parts:
                    try:
                        os.remove(part)
                    except OSError as e:
                        print(f"Error removing {part}: {e}")
                    print("Cleaned up shards, final checkpoint saved to", weights_path)

        global_config = dotdict(self.config["model_map"])
        model = StripedHyena(global_config)
        load_checkpoint(model, weights_path)

        return model
