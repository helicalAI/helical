from helical.models.base_models import HelicalRNAModel
from helical.models.mamba2_mrna.mamba2_mrna_config import Mamba2mRNAConfig
from datasets import Dataset
from transformers import Mamba2Model, Mamba2Config, AutoTokenizer
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

import logging

logger = logging.getLogger(__name__)

class Mamba2mRNA(HelicalRNAModel):
    """Mamba2-mRNA Model.
    
    The Mamba2-mRNA Model is a transformer-based model that can be used to extract RNA embeddings from RNA sequences. 
    The model is based on the Mamba2 model, which is a transformer-based model trained on RNA sequences. The model is available through this interface.
    
    Example
    ----------
    ```python
    from helical import Mamba2mRNA, Mamba2mRNAConfig
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_sequences = ["ACUG"*20, "AUGC"*20, "AUGC"*20, "ACUG"*20, "AUUG"*20]

    mamba2_mrna_config = Mamba2mRNAConfig(model_name="helix-mRNA", batch_size=5, device=device)
    mamba2_mrna = Mamba2mRNA(configurer=mamba2_mrna_config)

    # prepare data for input to the model
    processed_input_data = mamba2_mrna.process_data(input_sequences)

    # generate the embeddings for the processed data
    embeddings = mamba2_mrna.get_embeddings(processed_input_data)
    ```

    
    Parameters
    ----------
    configurer : Mamba2mRNAConfig
        The configuration object for the Helix-mRNA model.

    Notes
    ----------
    Helix_mRNA notes
    """
    default_configurer = Mamba2mRNAConfig()
    def __init__(self, configurer: Mamba2mRNAConfig = default_configurer):
        super().__init__()
        self.configurer = configurer
        self.config = configurer.config

        # downloader = Downloader()
        # for file in self.configurer.list_of_files_to_download:
        #         downloader.download_via_name(file)
        
        if self.config["model_name"] == "helix-mRNA-mamba":
            self.model = Mamba2Model.from_pretrained('helical-ai/Helix-mRNA', trust_remote=True)
            self.pretrained_config = Mamba2Config.from_pretrained('helical-ai/Helix-mRNA', trust_remote=True)
            self.tokenizer = AutoTokenizer.from_pretrained('helical-ai/Helix-mRNA', trust_remote=True)

        self.model.post_init()
        logger.info("Helix-mRNA initialized successfully.")

    def process_data(self, sequences: str) -> Dataset:
        """Process the RNA sequences and return a Dataset object.

        Parameters
        ----------
        sequences : str
            The RNA sequences.

        Returns
        -------
        Dataset
            The dataset object.
        """
        self.ensure_rna_sequence_validity(sequences)

        # arr_sequences = []
        # for sequence in tqdm(sequences, "Processing sequences"):
        #     arr_sequences.append(sequence)

        # tokenizer = AutoTokenizer.from_pretrained(self.config["model_name"], trust_remote_code=True)

        # return HelixmRNADataset(arr_sequences, tokenizer)
    
        tokenized_sequences = []
        for seq in sequences:
            tokenized_seq = self.tokenizer(seq, return_tensors="pt", padding="max_length", truncation=True, max_length=self.config['input_size'])
            tokenized_sequences.append(tokenized_seq)

        return Dataset.from_list(tokenized_sequences)

    def get_embeddings(self, dataset: Dataset) -> np.ndarray:
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
    