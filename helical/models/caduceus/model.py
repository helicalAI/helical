from helical.models.base_models import HelicalRNAModel
from helical.utils.downloader import Downloader
from .caduceus_config import CaduceusConfig
from .pretrained_model import CaduceusModel
from .caduceus_tokenizer import CaduceusTokenizer
import logging

LOGGER = logging.getLogger(__name__)

# default configuration if not specified
configurer = CaduceusConfig()

class Caduceus(HelicalRNAModel):
    def __init__(self, configurer: CaduceusConfig = configurer):
        super().__init__()

        self.configurer = configurer
        self.config = configurer.config
        self.files_config = configurer.files_config
        self.device = self.config['device']

        downloader = Downloader()
        for file in self.configurer.list_of_files_to_download:
            downloader.download_via_name(file)

        self.model =  CaduceusModel.from_pretrained(self.files_config['model_files_dir'])
        self.model.eval()
        self.model = self.model.to(self.device)

        print(self.model)
        self.tokenizer = CaduceusTokenizer(model_max_length=self.config['input_size'])
        
        LOGGER.info("Caduceus model initialized")

    def process_data(self):
        pass

    def get_embeddings(self):
        pass
        
