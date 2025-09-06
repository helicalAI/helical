import logging
import sys

logger = logging.getLogger("state_model")
logging.basicConfig(level=logging.INFO)

# check if logger has been initialized
if not logger.hasHandlers() or len(logger.handlers) == 0:
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

from helical.models.state.state_embeddings import stateEmbeddingsModel
from helical.models.state.state_transition import stateTransitionModel
from helical.models.state.state_finetune import stateFineTuningModel
from helical.models.state.state_config import stateConfig
from helical.models.state.train_configs import trainingConfig

from helical.models.state.vcc_train_model import stateTransitionTrainModel
from helical.models.state._vcc_eval import vcc_eval
