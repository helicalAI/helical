from numpy import ndarray
from typing_extensions import Self
from sklearn.preprocessing import LabelEncoder
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import F1Score
from tensorflow.keras.utils import to_categorical
from helical.models.base_models import BaseTaskModel
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl

class NeuralNetworkModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNetworkModel, self).__init__()
        self.save_hyperparameters()
        self.fc1 = nn.Linear(input_size, 256)
        self.dropout1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, 64)
        self.dropout2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(64, num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.softmax(self.fc3(x), dim=1)
        return x

    def training_step(self, batch, batch_idx):
        X_batch, y_batch = batch
        outputs = self(X_batch)
        loss = self.criterion(outputs, y_batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        X_batch, y_batch = batch
        outputs = self(X_batch)
        loss = self.criterion(outputs, y_batch)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

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
        self.input_shape = input_shape
        self.model = NeuralNetwork(input_shape, num_classes, self.learning_rate)

        # self.model = NeuralNetwork(input_shape, num_classes)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # self.criterion = nn.CrossEntropyLoss()

        # self.num_classes = num_classes
        # self.input_shape = (input_shape,)

        # self.model = Sequential()
        # self.model.add(Dense(256, activation='relu', input_shape=self.input_shape))
        # self.model.add(Dropout(0.4))
        # self.model.add(Dense(64, activation='relu'))
        # self.model.add(Dropout(0.4))
        # self.model.add(Dense(self.num_classes, activation='softmax'))

        # self.optimizer = Adam(learning_rate=self.learning_rate)
        # self.f1_metric = F1Score(average='macro')
        # self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=[self.f1_metric])

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
        x_val, y_val = validation_data
        self.encoder.fit_transform(np.concatenate((y_train, y_val), axis = 0))

        y_train_encoded = self.encoder.transform(y_train)
        y_val_encoded = self.encoder.transform(y_val)

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long)
        X_val_tensor = torch.tensor(x_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val_encoded, dtype=torch.long)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        trainer = pl.Trainer(max_epochs=self.epochs)
        trainer.fit(self.model, train_loader, val_loader)

        # y_train_encoded = self.encoder.transform(y_train)
        # y_train_encoded = to_categorical(y_train_encoded, num_classes = self.num_classes)

        # y_val_encoded = self.encoder.transform(y_val)
        # y_val_encoded = to_categorical(y_val_encoded, num_classes = self.num_classes)

        # self.model.fit(X_train, y_train_encoded, epochs = self.epochs, batch_size = self.batch_size, validation_data = (x_val, y_val_encoded))
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
        self.model.eval()
        X_tensor = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
        return self.encoder.inverse_transform(predicted.numpy())

        # predictions = self.model.predict(x)
        # y_pred = np.argmax(predictions, axis=1)
        # return self.encoder.inverse_transform(y_pred)
    
    def save(self, path: str) -> None:
        """Save the neural network model and its encoder to a directory.
        Any missing parents of this path are created as needed.

        Parameters
        ----------
        path : str
            The path to the directory to save the model and the encoder.
        """
        Path(path).mkdir(parents=True, exist_ok=True)
        np.save(f"{path}/encoder", self.encoder.classes_)
        torch.save(self.model.state_dict(), f"{path}/neural_network.pth")
        # self.model.save(f"{path}/neural_network.h5")

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
        self.model.load_state_dict(torch.load(path))                                                                                                                                                                                                                                                                       
        self.model.eval()

        # self.encoder.classes_ = np.load(classes)
        # self.model = tf.keras.models.load_model(path)
        return self
