import logging

from helical.models.hyena_dna.hyena_dna_config import HyenaDNAConfig
from helical.models.helical import HelicalBaseModel

class HyenaDNA(HelicalBaseModel):
    """HyenaDNA model."""
    default_configurer = HyenaDNAConfig()

    def __init__(self, configurer: HyenaDNAConfig = default_configurer) -> None:    
        super().__init__()
        self.config = configurer.config
        self.log = logging.getLogger("Hyena-DNA-Model")
                
    def process_data(self):
        pass

    def get_embeddings(self):
        pass