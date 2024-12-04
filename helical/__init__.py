import os
import logging

logging.captureWarnings(True)

class InfoAndErrorFilter(logging.Filter):
    def filter(self, record):
        return record.levelno in (logging.INFO, logging.ERROR)

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO) 

handler.addFilter(InfoAndErrorFilter())

formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
handler.setFormatter(formatter)

logger.addHandler(handler)

os.environ['TRANSFORMERS_VERBOSITY'] = 'error'


from .models.uce.model import UCEConfig, UCE
from .models.uce.fine_tuning_model import UCEFineTuningModel
from .models.geneformer.model import Geneformer,GeneformerConfig
from .models.geneformer.fine_tuning_model import GeneformerFineTuningModel
from .models.scgpt.model import scGPT, scGPTConfig
from .models.scgpt.fine_tuning_model import scGPTFineTuningModel
from .models.hyena_dna.model import HyenaDNA, HyenaDNAConfig
from .models.hyena_dna.fine_tuning_model import HyenaDNAFineTuningModel