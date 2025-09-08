import logging
import sys

logger = logging.getLogger("state")

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

from .state_config import stateConfig
from .train_configs import trainingConfig

from .state_embeddings import stateEmbeddingsModel
from .state_transition import stateTransitionModel
from .state_finetune import stateFineTuningModel
from .state_train import stateTransitionTrainModel

from ._vcc_eval import vcc_eval
