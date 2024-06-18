from abc import ABC, abstractmethod

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