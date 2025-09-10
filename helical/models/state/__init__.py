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
from .train_configs import trainingConfig

# from .state_embeddings import stateEmbeddingsModel
from .state_embeddings import stateEmbed
from .state_embeddings_torch import stateEmbedTorch
from .state_transition import stateTransitionModel
from .state_finetune import stateFineTuningModel
from .state_train import stateTransitionTrainModel

from .model_dir.vcc_eval import vcc_eval
