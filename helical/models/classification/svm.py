from helical.models.base_models import BaseTaskModel
from numpy import ndarray
from sklearn import svm
from typing_extensions import Self
from typing import Optional
import pickle
import os

class SupportVectorMachine(BaseTaskModel):
    def __init__(self, kernel='rbf', degree=3, C=1, decision_function_shape='ovr') -> None:
        self.kernel = kernel
        self.degree = degree
        self.C = C
        self.decision_function_shape = decision_function_shape

    def compile(self, num_classes: Optional[int] = None, input_shape: Optional[int] = None) -> None:
        """
        Compile a SVM. The input parameters are not needed for the SVM model, providing them makes for cleaner code.

        Parameters
        ----------
        num_classes : int, None
            The number of classes to predict, default is None. 
            The SVM should find this automatically.
        input_shape : int, None
            The input shape of the neural network, default is None.
            The SVM should find this automatically.
        """
        self.svm_model = svm.SVC(kernel = self.kernel, 
                                 degree = self.degree, 
                                 C = self.C, 
                                 decision_function_shape = self.decision_function_shape)  

    def train(self, X_train: ndarray, y_train: ndarray, **kwargs) -> Self:
        """Train an SVM on the training.

        Parameters
        ----------
        X_train : ndarray
            The training data features.
        y_train : ndarray
            The training data labels.

        Returns
        -------
        The neural network instance.
        """
        self.svm_model.fit(X_train, y_train)
        return self
  
    def predict(self, x: ndarray) -> ndarray:
        """Use the SVM to make predictions.

        Parameters
        ----------
        x : ndarray
            The data to make predictions upon.

        Returns
        -------
        The prediction of the SVM.
        """        
        return self.svm_model.predict(x)
    
    def save(self, path: str) -> None:
        """Save the SVM model to a file.

        Parameters
        ----------
        path : str
            The path to save the model.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)  
        file = f"{path}svm.h5"
        with open(file, 'wb') as f:
            pickle.dump(self.svm_model, f)

    def load(self, path: str) -> Self:
        """Load the SVM model from a file.

        Parameters
        ----------
        path : str
            The path to load the model from.

        Returns
        -------
        The SVM instance.
        """
        with open(path, 'rb') as f:
            self.svm_model = pickle.load(f)
        return self
    