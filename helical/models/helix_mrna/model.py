from helical.models.base_models import HelicalRNAModel
from helical.models.helix_mrna.helix_mrna_config import HelixmRNAConfig
from datasets import Dataset
from transformers import AutoModel, AutoConfig, AutoTokenizer
from .modeling_helix_mrna import HelixmRNAPretrainedModel
from .helix_mrna_tokenizer import CharTokenizer
from .helix_mrna_pretrained_config import HelixmRNAPretrainedConfig
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
    ```python
    from helical import HelixmRNA, HelixmRNAConfig
    import torch
    
    helix_mrna_config = HelimRNAConfig(batch_size=5)
    helix_mrna = HelixmRNA(configurer=helix_mrna_config)
    
    rna_sequences = ["ACUEGGG", "ACUEGGG", "ACUEGGG", "ACUEGGG", "ACUEGGG"]
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
    Helix_mRNA was trained using a character in between each codon of the RNA sequence. 
    This is done to ensure that the model can learn the structure of the RNA sequence. 
    Although it can take a standard RNA sequence as input, it is recommended to add the letter E between each codon of the RNA sequence to get better embeddings.
    """
    default_configurer = HelixmRNAConfig()
    def __init__(self, configurer: HelixmRNAConfig = default_configurer):
        super().__init__()
        self.configurer = configurer
        self.config = configurer.config

        # downloader = Downloader()
        # for file in self.configurer.list_of_files_to_download:
        #         downloader.download_via_name(file)
        
        self.model = HelixmRNAPretrainedModel.from_pretrained(self.config["model_name"])
        self.pretrained_config = HelixmRNAPretrainedConfig.from_pretrained(self.config["model_name"], trust_remote=True)
        self.tokenizer = CharTokenizer.from_pretrained(self.config["model_name"], trust_remote=True)

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
    