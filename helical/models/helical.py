from abc import ABC, abstractmethod
from helical.services.logger import Logger
from helical.constants.enums import LoggingType, LoggingLevel

class HelicalBaseModel(ABC, Logger):
    """Helical Base Model Class which serves as the base class for all models in the helical package. Each new model should be a subclass of this class.

        Parameters
        ----------
        logging_type : LoggingType
            The logging type
        level : LoggingLevel
            The logging level

        Returns
        -------
        None

        """
    
    def __init__(self, logging_type = LoggingType.CONSOLE, level = LoggingLevel.INFO):

        super().__init__(logging_type, level)

    @abstractmethod
    def process_data():
        pass

    @abstractmethod
    def get_embeddings():
        pass