import logging
import sys

LOGGER = logging.getLogger("state")

if not LOGGER.hasHandlers() or len(LOGGER.handlers) == 0:
    LOGGER.propagate = False
    LOGGER.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)

from .state_config import stateConfig
from .train_configs import trainConfig

from .state_embeddings import stateEmbed
from .state_transition import stateTransitionModel
from .fine_tuning_model import stateFineTuningModel
from .state_train import stateTransitionTrainModel