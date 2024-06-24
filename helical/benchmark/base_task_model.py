from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable
from anndata import AnnData
from datasets import Dataset

class BaseTaskModel(ABC):
    """
    Helical Base Task Model which serves as the base class for all models trained for a specific task.
    Each new model for a specific task should be a subclass of this class.
    """
    
    def __init__():
        pass

    @abstractmethod
    def compile(*args, **kwargs):
        pass

    @abstractmethod
    def train(*args, **kwargs):
        pass

    @abstractmethod
    def predict():
        pass

    @abstractmethod
    def save():
        pass

    @abstractmethod
    def load():
        pass
    

@runtime_checkable
class BaseModelProtocol(Protocol):
    def process_data(self, x: AnnData) -> Dataset:
        ...
    def get_embeddings(self, dataset: Dataset) -> ndarray:
        ...