import logging
from helical.models.hyena_dna.hyena_dna_config import HyenaDNAConfig
from helical.models.base_models import HelicalDNAModel
from tqdm import tqdm
from .hyena_dna_utils import HyenaDNADataset
from helical.models.hyena_dna.pretrained_model import HyenaDNAPreTrainedModel
import torch
from .standalone_hyenadna import CharacterTokenizer
from helical.utils.downloader import Downloader
from torch.utils.data import DataLoader
import numpy as np

LOGGER = logging.getLogger(__name__)

class HyenaDNA(HelicalDNAModel):
    """HyenaDNA model.
    This class represents the HyenaDNA model, which is a long-range genomic foundation model pretrained on context lengths of up to 1 million tokens at single nucleotide resolution.
    
    Example
    -------
    >>> from helical.models.hyena_dna.model import HyenaDNA, HyenaDNAConfig
    >>> hyena_config = HyenaDNAConfig(model_name = "hyenadna-tiny-1k-seqlen-d256")
    >>> model = HyenaDNA(configurer = hyena_config)   
    >>> sequence = 'ACTG' * int(1024/4)
    >>> tokenized_sequence = model.process_data(sequence)
    >>> embeddings = model.get_embeddings(tokenized_sequence)
    >>> print(embeddings.shape)

    Parameters
    ----------
        default_configurer : HyenaDNAConfig, optional, default = default_configurer
            The model configuration.
    
    Returns
    -------
    None

    Notes
    -----
    The link to the paper can be found `here <https://arxiv.org/abs/2306.15794>`_. 
    We use the implementation from the `hyena-dna <https://github.com/HazyResearch/hyena-dna>`_ repository.

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
            characters=['A', 'C', 'G', 'T', 'N'],  # add DNA characters, N is uncertain
            model_max_length=self.config['max_length'] + 2,  # to account for special tokens, like EOS
            add_special_tokens=False,  # we handle special tokens elsewhere
            padding_side='left', # since HyenaDNA is causal, we pad on the left
        )

        # prep model and forward
        self.device = self.config['device']
        self.model.to(self.device)
        self.model.eval()
        LOGGER.info(f"Model finished initializing.")

    def process_data(self, sequence: list[str]) -> HyenaDNADataset:
        """Process the input DNA sequence.

        Parameters 
        ----------
            sequences: list[str]
                The input DNA sequences to be processed.

        Returns
        -------
        HyenaDNADataset
            Containing processed DNA sequences.

        """
        LOGGER.info(f"Processing data")
        processed_sequences = []
        for seq in tqdm(sequence, desc="Processing sequences"):
            tok_seq = self.tokenizer(seq)
            tok_seq_input_ids = tok_seq["input_ids"]

            tensor = torch.LongTensor(tok_seq_input_ids)
            tensor = tensor.to(self.device)
            processed_sequences.append(tensor)

        dataset = HyenaDNADataset(torch.stack(processed_sequences))
        LOGGER.info(f"Data processing finished.")

        return dataset

    def get_embeddings(self, dataset: HyenaDNADataset) -> torch.Tensor:
        """Get the embeddings for the tokenized sequence.

        Args:
            dataset (HyenaDNADataset): The tokenized sequences.

        Returns:
            torch.Tensor: The embeddings for the tokenized sequences with a shape [batch_size, sequence_length, embeddings_size].

        """
        LOGGER.info(f"Inference started")

        train_data_loader = DataLoader(dataset, batch_size=self.config["batch_size"])
        with torch.inference_mode():
            embeddings = []
            for input_data in tqdm(train_data_loader, desc="Getting embeddings"):
                input_data = input_data.to(self.device)
                embeddings.append(self.model(input_data).detach().cpu().numpy())
        
        # output = torch.stack(embeddings)
        # other_dims = output.shape[2:]

        # reshaped_tensor = output.view(-1, *other_dims)
        return np.vstack(embeddings)
