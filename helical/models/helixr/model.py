from helical.models.base_models import HelicalRNAModel
from helical.models.helixr.helixr_config import HelixRConfig
from helical.models.helixr.hg38_char_tokenizer import CharTokenizer
from helical.models.helixr.dataset import HelixRDataset
from transformers import Mamba2Model
from helical.utils.downloader import Downloader

import logging

LOGGER = logging.getLogger(__name__)

class HelixR(HelicalRNAModel):

    default_configurer = HelixRConfig()
    def __init__(self, configurer: HelixRConfig = default_configurer):
        super().__init__()
        self.configurer = configurer
        self.config = configurer.config

        downloader = Downloader()
        downloader.download_model("helixr", self.configurer.model_dir)
        self.model = Mamba2Model.from_pretrained(self.configurer.model_dir)

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

        max_length = len(max(sequences, key=len))

        tokenizer = CharTokenizer(
            characters=["A", "C", "G", "U", "N"],
            model_max_length=max_length
        )

        return HelixRDataset(sequences, tokenizer, max_length)

    def get_embeddings(self, ):
        pass

    