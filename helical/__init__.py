import os
import logging

logging.captureWarnings(True)

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO) 

formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
handler.setFormatter(formatter)

logger.addHandler(handler)

os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

from .models.uce.model import UCEConfig, UCE
from .models.uce.fine_tuning_model import UCEFineTuningModel
from .models.geneformer.model import Geneformer,GeneformerConfig
from .models.geneformer.fine_tuning_model import GeneformerFineTuningModel
from .models.genept.model import GenePT,GenePTConfig
from .models.scgpt.model import scGPT, scGPTConfig
from .models.scgpt.fine_tuning_model import scGPTFineTuningModel
from .models.hyena_dna.model import HyenaDNA, HyenaDNAConfig
from .models.hyena_dna.fine_tuning_model import HyenaDNAFineTuningModel
from .models.helix_mrna import HelixmRNA, HelixmRNAConfig, HelixmRNAFineTuningModel
from .models.mamba2_mrna import Mamba2mRNA, Mamba2mRNAConfig, Mamba2mRNAFineTuningModel

try:
    from .models.caduceus import Caduceus, CaduceusConfig, CaduceusFineTuningModel
except:
    LOGGER = logging.getLogger(__name__)
    LOGGER.info("Caduceus not available: If you want to use this model, ensure you have a CUDA GPU and have installed the optional helical[mamba-ssm] dependencies.")
