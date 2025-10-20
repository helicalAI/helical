'''
This package is based on STATE by Adduri et al., 
available at https://www.biorxiv.org/content/10.1101/2025.06.26.661135v1. 
Licensed under CC BY-NC-SA 4.0.  
Modifications in this package: 
- Updated API call wrappers 
- Minor code restructuring

'''

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

from .state_config import StateConfig
from .state_embeddings import StateEmbed
from .state_perturb import StatePerturb
