from numpy import ndarray
from typing_extensions import Self
from sklearn.preprocessing import LabelEncoder
import numpy as np
from helical.models.base_models import BaseTaskModel
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging

LOGGER = logging.getLogger(__name__)
class NeuralNetwork(BaseTaskModel):
    def __init__(self, loss: str = "categorical_crossentropy",learning_rate: float = 0.001, epochs=10, batch_size=32) -> None:
        
        if loss == "categorical_crossentropy":
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            message = f"Loss function {loss} not implemented."
            LOGGER.error(message)
            raise NotImplementedError(message)
        
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
        self.model = nn.Sequential(
            nn.Linear(input_shape, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )

        # Set optimizer and loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.CrossEntropyLoss()

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
        # Ensure model is in training mode
        self.model.train()
        
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

        # Training loop
        for epoch in range(self.epochs):
            for batch_X, batch_y in train_loader:
                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(batch_X)

                # Compute loss
                loss = self.loss_fn(outputs, batch_y)

                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()

            # Validation phase (optional)
            self.model.eval()
            with torch.no_grad():
                val_losses = []
                for val_X, val_y in val_loader:
                    val_outputs = self.model(val_X)
                    val_loss = self.loss_fn(val_outputs, val_y)
                    val_losses.append(val_loss.item())
                
                print(f"Epoch {epoch+1}, Validation Loss: {sum(val_losses)/len(val_losses)}")
            
            # Set back to training mode for next epoch
            self.model.train()
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
        torch.save(self.model, f"{path}/neural_network.pth")

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

        self.encoder.classes_ = classes
        # self.model = nn.Sequential()
        self.model = torch.load(path)
        self.model.eval()

        return self