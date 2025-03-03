import logging

try:
    from .evo_2_config import Evo2Config
    from .model import Evo2
except:
    LOGGER = logging.getLogger(__name__)
    LOGGER.info(
        "Evo 2 not available: If you want to use this model, please follow the installation instructions in the evo_2/README."
    )
