
from sklearn.model_selection import train_test_split
from helical.benchmark.task_models.base_task_model import BaseTaskModel
import logging
from numpy import ndarray
from anndata import AnnData
from helical.models.helical import HelicalBaseModel
from datasets import Dataset
from typing import Protocol, runtime_checkable, Any, Optional, Self
import numpy as np

@runtime_checkable
class CustomClassificationModelProtocol(Protocol):
    def predict(self, x: AnnData) -> None:
        ...

@runtime_checkable
class CustomBaseModelProtocol(Protocol):
    def process_data(self, x: AnnData) -> None:
        ...
    def get_embeddings(self, dataset: Dataset) -> None:
        ...

LOGGER = logging.getLogger(__name__)

class Classifier():
    def __init__(self) -> None:

        self.trained_task_model = None
        self.base_model = None
        self.name = None

    def get_predictions(self, data: AnnData) -> ndarray:
        """Based on the evaluation data, use the trained BaseTaskModels (saved as class variable 'self.tained_models' in a dict) and make predictions.

        Parameters
        ----------
        x_eval : dict[str, ndarray]
            The predictions for each model.
    
        Returns
        -------
        A dictionary containing the predictions for each model.
        """
        if self.base_model is not None:
            dataset = self.base_model.process_data(data)
            embeddings = self.base_model.get_embeddings(dataset)
            x = embeddings
        
        return self.trained_task_model.predict(x) 

    def train_classifier(self, base_model: Any, train_anndata: AnnData, head: BaseTaskModel) -> Self:
        self.base_model = base_model
        self.name = f"{base_model.__class__.__name__} with {head.__class__.__name__}"

        if isinstance(base_model, HelicalBaseModel):
            LOGGER.info(f"Getting training embeddings with {base_model.__class__.__name__}.")
            dataset = base_model.process_data(train_anndata)
            x = base_model.get_embeddings(dataset)
    
        else:
            pass
        
        LOGGER.info(f"Training classification model head of type '{head.__class__.__name__}'.")
        
        y = np.array(train_anndata.obs["cell_type"].tolist())
        num_classes = len(np.unique(y))

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        head.compile(num_classes, x.shape[1])
        self.trained_task_model = head.train(X_train, y_train, validation_data=(X_test, y_test))
       
        return self

    def load_custom_model(self, 
                          base_model: Optional[CustomBaseModelProtocol], 
                          classification_model: CustomClassificationModelProtocol, 
                          name: str) -> Self:
        
        if base_model and not isinstance(base_model, CustomBaseModelProtocol):
            message = "Expected an instance of a class implementing 'CustomBaseModelProtocol'."
            LOGGER.error(message)
            raise TypeError(message)
        
        if not isinstance(classification_model, CustomClassificationModelProtocol):
            message = "Expected an instance of a class implementing 'CustomClassificationModelProtocol'."
            LOGGER.error(message)
            raise TypeError(message)
        
        self.base_model = base_model
        self.trained_task_model = classification_model
        self.name = name
        return self