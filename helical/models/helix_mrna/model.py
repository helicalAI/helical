from helical.models.base_models import HelicalRNAModel
from helical.models.helix_mrna.helix_mrna_config import HelixmRNAConfig
from helical.models.helix_mrna.helix_mrna_utils import HelixmRNADataset
from transformers import AutoModel, AutoConfig, AutoTokenizer
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

import logging

logger = logging.getLogger(__name__)

class HelixmRNA(HelicalRNAModel):
    """Helix-mRNA Model.
    
    The Helix-mRNA Model is a transformer-based model that can be used to extract RNA embeddings from RNA sequences. 
    The model is based on the Mamba2 model, which is a transformer-based model trained on RNA sequences. The model is available through this interface.
    
    Example
    -------
    >>> from helical.models import Helix_mRNA, HelixRConfig
    >>> import pandas as pd
    >>>
    >>> helixr_config = HelimRNAConfig(batch_size=5)
    >>> helixr = HelixmRNA(configurer=helixr_config)
    >>>
    >>> rna_sequences = pd.read_csv("rna_sequences.csv")["Sequence"]
    >>> helixr_dataset = helixr.process_data(rna_sequences)
    >>> rna_embeddings = helixr.get_embeddings(helixr_dataset)
    >>>
    >>> print("Helix_mRNA embeddings shape: ", rna_embeddings.shape)
    
    Parameters
    ----------
    configurer : HelixmRNAConfig
        The configuration object for the Helix-mRNA model.

    Notes
    ----------
    Helix_mRNA notes
    """
    default_configurer = HelixmRNAConfig()
    def __init__(self, configurer: HelixmRNAConfig = default_configurer):
        super().__init__()
        self.configurer = configurer
        self.config = configurer.config

        # downloader = Downloader()
        # for file in self.configurer.list_of_files_to_download:
        #         downloader.download_via_name(file)

        self.model = AutoModel.from_pretrained('helical-ai/Helix-mRNA')

        self.pretrained_config = AutoConfig.from_pretrained('helical-ai/Helix-mRNA', trust_remote=True)

        logger.info("Helix-mRNA initialized successfully.")

    def process_data(self, sequences: str) -> HelixmRNADataset:
        """Process the RNA sequences and return a Dataset object.

        Parameters
        ----------
        sequences : str
            The RNA sequences.

        Returns
        -------
        HelixmRNADataset
            The dataset object.
        """
        self.ensure_rna_sequence_validity(sequences)

        arr_sequences = []
        for sequence in tqdm(sequences, "Processing sequences"):
            arr_sequences.append(sequence)

        tokenizer = AutoTokenizer.from_pretrained(self.config["model_name"], trust_remote_code=True)

        return HelixmRNADataset(arr_sequences, tokenizer)

    def get_embeddings(self, dataset: HelixmRNADataset) -> np.ndarray:
        """Get the embeddings for the RNA sequences.
        
        Parameters
        ----------
        dataset : HelixmRNADataset
            The dataset object.
        
        Returns
        -------
        np.ndarray
            The embeddings array.
        """
        dataloader = DataLoader(dataset, batch_size=self.config["batch_size"], shuffle=False)
        embeddings = []

        self.model.to(self.config["device"])

        progress_bar = tqdm(dataloader, desc="Getting embeddings")
        with torch.no_grad():
            for batch in progress_bar:
                input_ids = batch["input_ids"].to(self.config["device"])
                special_tokens_mask = batch["special_tokens_mask"].to(self.config["device"])

                output = self.model(input_ids, special_tokens_mask=special_tokens_mask)

                last_hidden_states = output[0]

                if input_ids is not None:
                    batch_size, _ = input_ids.shape[:2]

                if self.pretrained_config.pad_token_id is None and batch_size > 1:
                    message = "Cannot handle batch sizes > 1 if no padding token is defined."
                    logger.error(message)
                    raise ValueError(message)

                if self.pretrained_config.pad_token_id is None:
                    sequence_lengths = -1
                else:
                    if input_ids is not None:
                        sequence_lengths = torch.eq(input_ids, self.pretrained_config.pad_token_id).int().argmax(-1) - 1
                        sequence_lengths = sequence_lengths % input_ids.shape[-1]
                        sequence_lengths = sequence_lengths.to(last_hidden_states.device)
                    else:
                        sequence_lengths = -1

                pooled_last_hidden_states = last_hidden_states[
                    torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
                ]
                
                # Take second last element from the output as last element is a special token
                embeddings.append(pooled_last_hidden_states.cpu().numpy())

                del batch
                del output

        return np.concatenate(embeddings)
    