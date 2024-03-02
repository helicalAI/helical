from abc import ABC, abstractmethod
from helical.services.logger import Logger

class HelicalBaseModel(ABC, Logger):
    
    @abstractmethod
    def get_model():
        pass

    @abstractmethod
    def process_data():
        pass

    @abstractmethod
    def run():
        pass

    @abstractmethod
    def get_embeddings():
        pass