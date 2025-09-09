import logging
import sys

logger = logging.getLogger("state")

# TO DO: remove this dependency issue in the future
# pip install cell eval and then scipy==1.13.1 in the background and then run the code
# helical requires scipy==1.13.1 but cell-eval requires scipy==1.16.0
import subprocess, sys

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "cell-eval"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "scipy==1.13.1"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)

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

from .model_dir._vcc_eval import vcc_eval
