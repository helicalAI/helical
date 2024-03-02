from abc import ABC, abstractmethod
from helical.services.logger import Logger

class HelicalBaseModel(ABC, Logger):
    
    @abstractmethod
    def run():
        pass

    @abstractmethod
    def get_embeddings():
        pass