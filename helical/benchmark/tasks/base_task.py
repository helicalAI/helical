from abc import ABC, abstractmethod

class BaseTask(ABC):
    """Helical Base Task which serves as the base class for all tasks in the benchmarking process. Each new task should be a subclass of this class."""
    
    def __init__():
        pass

    @abstractmethod
    def train_task_models():
        pass

    @abstractmethod
    def get_predictions():
        pass

    @abstractmethod
    def get_evaluations():
        pass
