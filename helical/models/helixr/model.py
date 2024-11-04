from helical.models.base_models import HelicalRNAModel
from helical.models.helixr.helixr_config import HelixRConfig
from helical.models.helixr.hg38_char_tokenizer import CharTokenizer
from helical.models.helixr.dataset import HelixRDataset
from transformers import Mamba2Model, AutoConfig
from helical.utils.downloader import Downloader
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

import logging

logger = logging.getLogger(__name__)

class HelixR(HelicalRNAModel):
    """HelixR Model.
    
    The HelixR Model is a transformer-based model that can be used to extract RNA embeddings from RNA sequences. 
    The model is based on the Mamba2 model, which is a transformer-based model trained on RNA sequences. The model is available through this interface.
    
    Example
    -------
    >>> from helical.models import HelixR, HelixRConfig
    >>> import pandas as pd
    >>>
    >>> helixr_config = HelixRConfig(batch_size=5)
    >>> helixr = HelixR(configurer=helixr_config)
    >>>
    >>> rna_sequences = pd.read_csv("rna_sequences.csv")["Sequence"]
    >>> helixr_dataset = helixr.process_data(rna_sequences)
    >>> rna_embeddings = helixr.get_embeddings(helixr_dataset)
    >>>
    >>> print("HelixR embeddings shape: ", rna_embeddings.shape)
    
    Parameters
    ----------
    configurer : HelixRConfig
        The configuration object for the HelixR model.

    Notes
    ----------
    HelixR notes
    """
    default_configurer = HelixRConfig()
    def __init__(self, configurer: HelixRConfig = default_configurer):
        super().__init__()
        self.configurer = configurer
        self.config = configurer.config

        downloader = Downloader()
        for file in self.configurer.list_of_files_to_download:
                downloader.download_via_name(file)

        self.model = Mamba2Model.from_pretrained(self.config["model_dir"])

        self.pretrained_config = AutoConfig.from_pretrained(self.config["model_dir"])

        logger.info("HelixR initialized successfully.")

    def process_data(self, sequences: str) -> HelixRDataset:
        """Process the RNA sequences and return a Dataset object.

        Parameters
        ----------
        sequences : str
            The RNA sequences.

        Returns
        -------
        HelixRDataset
            The dataset object.
        """
        self.ensure_rna_sequence_validity(sequences)

        tokenizer = CharTokenizer(
            characters=["A", "C", "G", "U", "N"]
        )

        return HelixRDataset(sequences, tokenizer)

    def get_embeddings(self, dataset: HelixRDataset) -> np.ndarray:
        """Get the embeddings for the RNA sequences.
        
        Parameters
        ----------
        dataset : HelixRDataset
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

                # Take second last element from the output as last element is a special token
                embeddings.append(output.last_hidden_state[-2].cpu().numpy())

                del batch
                del output

        return np.concatenate(embeddings)
    