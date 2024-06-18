from helical.benchmark.task_models.base_task_model import BaseTaskModel
from numpy import ndarray
from sklearn import svm
import numpy as np
from typing import Self
from copy import deepcopy

class SupportVectorMachine(BaseTaskModel):
    def __init__(self, kernel='rbf', degree=3, C=1, decision_function_shape='ovr') -> None:
        self.kernel = kernel
        self.degree = degree
        self.C = C
        self.decision_function_shape = decision_function_shape

    def compile(self, num_classes: int, input_shape: int) -> None:
        """Compile a SVM."""
        self.num_classes = num_classes
        self.input_shape = (input_shape,)
        self.svm_model = svm.SVC(kernel = self.kernel, degree = self.degree, C = self.C, decision_function_shape = self.decision_function_shape)  

    def train(self, X_train: ndarray, y_train: ndarray, **kwargs) -> Self:
        """Train the neural network on the training and validation data.

        Parameters
        ----------
        X_train : ndarray
            The training data features.
        y_train : ndarray
            The training data labels.
        validation_data : tuple(ndarray, ndarray)
            The validation data features and labels.

        Returns
        -------
        The neural network instance.
        """
        # Since SVM in scikit-learn does not directly support multi-label classification,
        # we need to train one SVM per class (one-vs-rest approach)
        # self.svms = []
        # for i in range(self.num_classes):
        #     svm = deepcopy(self.svm_model)
        #     svm.fit(X_train, y_train[:, i])
        #     self.svms.append(svm)
        # return self
        return self.svm_model.fit(X_train, y_train)
  
    def predict(self, x: ndarray) -> ndarray:
        """Use the neural network to make predictions.

        Parameters
        ----------
        x : ndarray
            The data to make predictions upon.

        Returns
        -------
        The prediction of the neural network.
        """        
        return self.svm_model.predict(x)
    
