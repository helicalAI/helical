import logging

from helical.models.hyena_dna.hyena_dna_config import HyenaDNAConfig
from helical.models.helical import HelicalBaseModel
from helical.models.hyena_dna.pretrained_model import HyenaDNAPreTrainedModel
import torch
from .standalone_hyenadna import CharacterTokenizer
from helical.services.downloader import Downloader
LOGGER = logging.getLogger(__name__)

class HyenaDNA(HelicalBaseModel):
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

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = HyenaDNAPreTrainedModel().from_pretrained(self.config)

        # create tokenizer
        self.tokenizer = CharacterTokenizer(
            characters=['A', 'C', 'G', 'T', 'N'],  # add DNA characters, N is uncertain
            model_max_length=self.config['max_length'] + 2,  # to account for special tokens, like EOS
            add_special_tokens=False,  # we handle special tokens elsewhere
            padding_side='left', # since HyenaDNA is causal, we pad on the left
        )

        # prep model and forward
        self.model.to(self.device)
        self.model.eval()
        LOGGER.info(f"Model finished initializing.")

    def process_data(self, sequence: str) -> torch.Tensor:
        """Process the input DNA sequence.

        Parameters 
        ----------
            sequence: str
                The input DNA sequence to be processed.

        Returns
        -------
            torch.Tensor
                The processed tokenized sequence.

        """
        tok_seq = self.tokenizer(sequence)
        tok_seq = tok_seq["input_ids"]  # grab ids
        
        # place on device, convert to tensor
        tok_seq = torch.LongTensor(tok_seq).unsqueeze(0)  # unsqueeze for batch dim
        tok_seq = tok_seq.to(self.device)
        return tok_seq

    def get_embeddings(self, tok_seq: torch.Tensor) -> torch.Tensor:
        """Get the embeddings for the tokenized sequence.

        Args:
            tok_seq: torch.Tensor
                The tokenized sequence.

        Returns:
            torch.Tensor: The embeddings for the tokenized sequence.

        """
        LOGGER.info(f"Inference started")
        with torch.inference_mode():
            return self.model(tok_seq)