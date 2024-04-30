from abc import ABC, abstractmethod
from helical.services.logger import Logger
from pathlib import Path
import json
from helical.constants.enums import LoggingType, LoggingLevel

class HelicalBaseModel(ABC, Logger):
    
    def __init__(self, model_dir: str, model_args_path: Path, logging_type = LoggingType.CONSOLE, level = LoggingLevel.INFO):
        """ Initializes the HelicalBaseModel class 

        Parameters
        ----------
        model_dir : str
            The path to the model directory
        model_args_path : Path
            The path to the model arguments file
        logging_type : LoggingType
            The logging type
        level : LoggingLevel
            The logging level

        Returns
        -------
        None

        """

        super().__init__(logging_type, level)
        with open(model_args_path, "r") as f:
            model_config = json.load(f)
        self.model_config = model_config
        self.model_dir = Path(model_dir)

    @abstractmethod
    def process_data():
        pass

    @abstractmethod
    def get_embeddings():
        pass