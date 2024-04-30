from abc import ABC, abstractmethod
from helical.services.logger import Logger
from pathlib import Path
import json
from helical.constants.enums import LoggingType, LoggingLevel

class HelicalBaseModel(ABC, Logger):
    
    def __init__(self, logging_type = LoggingType.CONSOLE, level = LoggingLevel.INFO):
        super().__init__(logging_type, level)

    @abstractmethod
    def process_data():
        pass

    @abstractmethod
    def get_embeddings():
        pass