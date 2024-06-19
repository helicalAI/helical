
from sklearn.model_selection import train_test_split
from helical.benchmark.base_task_model import BaseTaskModel
import logging
import numpy as np
from numpy import ndarray
from anndata import AnnData
from datasets import Dataset
from typing import Protocol, runtime_checkable, Optional, Self, Union

@runtime_checkable
class BaseModelProtocol(Protocol):
    def process_data(self, x: AnnData) -> Dataset:
        ...
    def get_embeddings(self, dataset: Dataset) -> ndarray:
        ...
@runtime_checkable
class ClassificationModelProtocol(Protocol):
    def predict(self, x: Union[AnnData, ndarray]) -> ndarray:
        ...

LOGGER = logging.getLogger(__name__)

class Classifier():
    def __init__(self) -> None:

        self.trained_task_model = None
        self.base_model = None
        self.name = None

    def get_predictions(self, x: Union[AnnData, ndarray]) -> ndarray:
        """
        Make predictions on the data. 
        
        - If no base model is provided, it is assumed that the trained_task_model is a custom 
            standalone model that can handle the data directly.
        - If a base model is provided, the data is processed by the base model and the embeddings are used as input to the trained_task_model.

        Parameters
        ----------
        x : AnnData
            The predictions for each model.
    
        Returns
        -------
        A numpy array with the predictions.
        """
        if self.base_model is not None:
            dataset = self.base_model.process_data(x)
            embeddings = self.base_model.get_embeddings(dataset)
            x = embeddings
        
        return self.trained_task_model.predict(x) 

    def train_classifier_head(self, 
                              train_anndata: AnnData, 
                              base_model: BaseModelProtocol, 
                              head: BaseTaskModel, 
                              test_size: float = 0.2,
                              random_state: int = 42) -> Self:
        """Train the classification head. The base model is used to generate the embeddings, which are then used to train the model head.

        Parameters
        ----------
        train_anndata : AnnData
            The data to train the model head on.
        base_model : BaseModelProtocol
            The base model to generate the embeddings.
        head : BaseTaskModel
            The classification model head to train.
        test_size : float
            The size of the test set.
        random_state : int
            The random state for the train/test split.
    
        Raises
        ------
        TypeError
            If the base_model is not an instance of a class implementing 'BaseModelProtocol'.
            

        Returns
        -------
        An instance of the Classifier class, which contains the base_model and the trained model head.
        """
        
        self.base_model = base_model
        self.name = f"{base_model.__class__.__name__} with {head.__class__.__name__}"

        # first, get the embeddings
        if isinstance(base_model, BaseModelProtocol):
            LOGGER.info(f"Getting training embeddings with {base_model.__class__.__name__}.")
            dataset = base_model.process_data(train_anndata)
            x = base_model.get_embeddings(dataset)
    
        else:
            message = "To train a classifier head, a base model of type 'BaseModelProtocol' needs to generate the embeddings first."
            LOGGER.error(message)
            raise TypeError(message)
        
        # then, train the classification model
        LOGGER.info(f"Training classification model '{self.name}'.")
        y = np.array(train_anndata.obs["cell_type"].tolist())
        num_classes = len(np.unique(y))
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = test_size, random_state = random_state)
        head.compile(num_classes, x.shape[1])
        self.trained_task_model = head.train(X_train, y_train, validation_data=(X_test, y_test))
       
        return self

    def load_custom_model(self, 
                          base_model: Optional[BaseModelProtocol], 
                          classification_model: ClassificationModelProtocol, 
                          name: str) -> Self:
        """
        Load a custom classifier model.
        - If no base model is provided, it is assumed that the classification_model can directly classify data.
            This classificaiton_model must follow the ClassificationModelProtocol and implement a predict method.
        - If a base model is provided, the data is processed by the base model and the embeddings are used as input to the classification_model.
            The base_model must follow the BaseModelProtocol and implement a process_data and get_embeddings method.

        Parameters
        ----------
        base_model : None, BaseModelProtocol
            The optional base model to generate the embeddings.
        classification_model : ClassificationModelProtocol
            The classification model to load.
        name : str
            The name of the model.
    
        Raises
        ------
        TypeError
            If the base_model is not an instance of a class implementing 'BaseModelProtocol' or 
            the classification_model is not an instance of a class implementing 'ClassificationModelProtocol'.

        Returns
        -------
        An instance of the Classifier class, which contains the (optional) base_model and the trained model head.
        """

        if base_model and not isinstance(base_model, BaseModelProtocol):
            message = "Expected an instance of a class implementing 'BaseModelProtocol'."
            LOGGER.error(message)
            raise TypeError(message)
        
        if not isinstance(classification_model, ClassificationModelProtocol):
            message = "Expected an instance of a class implementing 'ClassificationModelProtocol'."
            LOGGER.error(message)
            raise TypeError(message)
        
        self.base_model = base_model
        self.trained_task_model = classification_model
        self.name = name
        return self