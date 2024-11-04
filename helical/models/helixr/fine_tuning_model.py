from typing import Literal, Optional
from helical.models.base_models import HelicalBaseFineTuningHead, HelicalBaseFineTuningModel
from helical.models.helixr.model import HelixR, HelixRConfig
from helical.models.helixr.dataset import HelixRDataset
from transformers import get_scheduler
import torch
from torch import optim
from torch.nn.modules import loss
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np

import logging

logger = logging.getLogger(__name__)

class HelixRFineTuningModel(HelicalBaseFineTuningModel, HelixR):
    """HelixRFineTuningModel
    Fine-tuning model for the HelixR model. This model can be used to fine-tune the HelixR model on a downstream task.

    Parameters
    ----------
    helixr_config : HelixRConfig
        The configuration object for the HelixR model. The same config object can be used for both the HelixR and HelixRFineTuningModel.
    fine_tuning_head : Literal["classification", "regression"] | HelicalBaseFineTuningHead
        The type of fine-tuning head to use for the model. This can be either a classification or regression head, or a custom fine-tuning head.
    output_size : Optional[int]
        The output size of the fine-tuning model. This is required if the fine_tuning_head is a string specified task. For a classification task this is number of unique classes.

    Methods
    ----------
    train(train_dataset, train_labels, optimizer, optimizer_params, loss_function, epochs, freeze_layers, validation_dataset, validation_labels, lr_scheduler_params)
        Fine-tunes the HelixR model on the given dataset.
    get_outputs(dataset)
        Returns the outputs of the model for the given dataset.
    """
    def __init__(self,
                 helixr_config: HelixRConfig, 
                 fine_tuning_head: Literal["classification", "regression"] | HelicalBaseFineTuningHead, 
                 output_size: Optional[int]=None):
        HelicalBaseFineTuningModel.__init__(self, fine_tuning_head, output_size)
        HelixR.__init__(self, helixr_config)

        self.fine_tuning_head.set_dim_size(self.pretrained_config.hidden_size)

    def _forward(self, input_ids, special_tokens_mask):
        """Forward pass for the HelixR fine-tuning model.
        
        Parameters
        ----------
        input_ids : torch.Tensor
            The input_ids tensor for the model.
        special_tokens_mask : torch.Tensor
            The special_tokens_mask tensor for the model.
        
        Returns
        -------
        torch.Tensor
            The output tensor from the model."""
        outputs = self.model(input_ids, special_tokens_mask=special_tokens_mask)
        last_hidden_states = outputs[0]

        if input_ids is not None:
            batch_size, _ = input_ids.shape[:2]

        if self.pretrained_config.pad_token_id is None and batch_size > 1:
            message = "Cannot handle batch sizes > 1 if no padding token is defined."
            logger.error(message)
            raise ValueError(message)

        if self.pretrained_config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = torch.eq(input_ids, self.pretrained_config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(last_hidden_states.device)
            else:
                sequence_lengths = -1

        pooled_last_hidden_states = last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
        ]

        head_outputs = self.fine_tuning_head(pooled_last_hidden_states)
        return head_outputs
    
    def train(
            self,
            train_dataset: HelixRDataset, 
            train_labels: np.ndarray,
            optimizer: optim = optim.AdamW,
            optimizer_params: dict = {'lr': 0.0001}, 
            loss_function: loss = loss.CrossEntropyLoss(), 
            epochs: int = 1,
            freeze_layers: int = 4,
            validation_dataset: Optional[HelixRDataset] = None,
            validation_labels: Optional[np.ndarray] = None,
            lr_scheduler_params: Optional[dict] = None):
        """Fine-tunes the HelixR model on the given dataset.
        
        Parameters
        ----------
        train_dataset : Dataset
            A helical processed dataset for fine-tuning
        optimizer : torch.optim, default = torch.optim.AdamW
            The optimizer to be used for training.
        optimizer_params : dict
            The optimizer parameters to be used for the optimizer specified. This list should NOT include model parameters.
            e.g. optimizer_params = {'lr': 0.0001}
        loss_function : torch.nn.modules.loss, default = torch.nn.modules.loss.CrossEntropyLoss()
            The loss function to be used.
        label : str, optional, default = "cell_types"
            The column in the dataset containing the training labels. These should be stored as unique per class integers.
        epochs : int, optional, default = 10
            The number of epochs to train the model
        freeze_layers : int, optional, default = 2
            The number of layers to freeze.
        validation_dataset : Dataset, default = None
            A helical processed dataset for per epoch validation. If this is not specified, no validation will be performed.
        lr_scheduler_params : dict, default = None
            The learning rate scheduler parameters for the transformers get_scheduler method. The optimizer will be taken from the optimizer input and should not be included in the learning scheduler parameters. If not specified, no scheduler will be used.
            e.g. lr_scheduler_params = { 'name': 'linear', 'num_warmup_steps': 0, 'num_training_steps': 5 }

        """
        # initialise optimizer
        optimizer = optimizer(self.parameters(), **optimizer_params)

        # set labels for the dataset
        train_dataset.set_labels(train_labels)
        if validation_labels is not None:
            validation_dataset.set_labels(validation_labels)

        # initialise lr_scheduler
        lr_scheduler = None
        if lr_scheduler_params is not None: 
            lr_scheduler = get_scheduler(optimizer=optimizer, **lr_scheduler_params)
        
        if freeze_layers > 0:
            logger.info(f"Freezing the first {freeze_layers} layers of the HelixR model.")

            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.layers[-freeze_layers:].parameters():
                param.requires_grad = True

        self.to(self.config["device"])

        self.model.train()
        self.fine_tuning_head.train()

        train_dataloader = DataLoader(train_dataset, batch_size=self.config["batch_size"], shuffle=False)

        if validation_dataset is not None:
            validation_dataloader = DataLoader(validation_dataset, batch_size=self.config["batch_size"], shuffle=False)

        logger.info("Starting Fine-Tuning")
        for j in range(epochs):
            training_loop = tqdm(train_dataloader, desc="Fine-Tuning")
            batch_loss = 0.0
            batches_processed = 0
            for batch in training_loop:
                input_ids = batch["input_ids"].to(self.config["device"])
                special_tokens_mask = batch["special_tokens_mask"].to(self.config["device"])
                labels = batch['labels'].to(self.config["device"])
                
                outputs = self._forward(input_ids, special_tokens_mask=special_tokens_mask)

                loss = loss_function(outputs, labels.unsqueeze(1))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                batch_loss += loss.item()
                batches_processed += 1
                training_loop.set_postfix({"loss": batch_loss/batches_processed})
                training_loop.set_description(f"Fine-Tuning: epoch {j+1}/{epochs}")

                del batch
                del outputs

            del training_loop

            if lr_scheduler is not None:
                lr_scheduler.step()

            if validation_dataset is not None:
                testing_loop = tqdm(validation_dataloader, desc="Fine-Tuning Validation")
                val_loss = 0.0
                count = 0.0
                for test_batch in testing_loop:
                    input_ids = test_batch["input_ids"].to(self.config["device"])
                    special_tokens_mask = test_batch["special_tokens_mask"].to(self.config["device"])
                    labels = test_batch['labels'].to(self.config["device"])
                
                    with torch.no_grad():
                        outputs = self._forward(input_ids, special_tokens_mask=special_tokens_mask)

                    val_loss += loss_function(outputs, labels.unsqueeze(1)).item()
                    count += 1.0
                    testing_loop.set_postfix({"val_loss": val_loss/count})

                    del test_batch
                    del outputs
                
                del testing_loop

        logger.info(f"Fine-Tuning Complete. Epochs: {epochs}")

    def get_outputs(self, dataset: HelixRDataset) -> np.ndarray:
        dataloader = DataLoader(dataset, batch_size=self.config["batch_size"], shuffle=False)
        outputs = []

        self.model.to(self.config["device"])

        progress_bar = tqdm(dataloader, desc="Getting embeddings")
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(self.config["device"])
            special_tokens_mask = batch["special_tokens_mask"].to(self.config["device"])

            with torch.no_grad():
                output = self._forward(input_ids, special_tokens_mask=special_tokens_mask)

            outputs.append(output.cpu().numpy())

            del batch
            del output

        return np.concatenate(outputs)

