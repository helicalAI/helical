from numpy import ndarray
from typing import Self
from sklearn.preprocessing import LabelEncoder
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import F1Score
from tensorflow.keras.utils import to_categorical
from helical.benchmark.base_task_model import BaseTaskModel

class NeuralNetwork(BaseTaskModel):
    def __init__(self, loss: str = "categorical_crossentropy", learning_rate: float = 0.001, epochs=10, batch_size=32) -> None:
        self.loss = loss
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.encoder = LabelEncoder()

    def compile(self, num_classes: int, input_shape: int) -> None:
        """Compile a neural network. The model is a simple feedforward neural network with 2 hidden layers. 
        TODO - Add more flexibility to the model architecture.
        
        Parameters
        ----------
        num_classes : int
            The number of classes to predict.
        input_shape : int
            The input shape of the neural network.
        """

        self.num_classes = num_classes
        self.input_shape = (input_shape,)

        self.model = Sequential()
        self.model.add(Dense(256, activation='relu', input_shape=self.input_shape))
        self.model.add(Dropout(0.4))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.4))
        self.model.add(Dense(self.num_classes, activation='softmax'))

        self.optimizer = Adam(learning_rate=self.learning_rate)
        self.f1_metric = F1Score(average='macro')
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.f1_metric)

    def train(self, X_train: ndarray, y_train: ndarray, validation_data: tuple[ndarray, ndarray]) -> Self:
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
        y_encoded = self.encoder.fit_transform(y_train)
        y_encoded = to_categorical(y_encoded, num_classes = self.num_classes)

        self.model.fit(X_train, y_encoded, self.epochs, self.batch_size, validation_data)
        return self
    
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
        predictions = self.model.predict(x)
        y_pred = np.argmax(predictions, axis=1)
        return self.encoder.inverse_transform(y_pred)
    
    def save(self, path: str) -> None:
        """Save the neural network model to a file.

        Parameters
        ----------
        path : str
            The path to save the model.
        """
        self.model.save(path)

    def load(self, path: str, classes: ndarray) -> Self:
        """Load the neural network model from a file.

        Parameters
        ----------
        path : str
            The path to load the model from.
        classes : ndarray
            The classes used for encoding the labels.

        Returns
        -------
        The neural network instance.
        """
        # set to None, showing this is a loaded model and not trained
        self.loss = None
        self.learning_rate = None
        self.epochs = None
        self.batch_size = None

        self.encoder.classes_ = np.load(classes)
        self.model = tf.keras.models.load_model(path)
        return self
