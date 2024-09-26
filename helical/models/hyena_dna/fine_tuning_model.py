from typing import Literal, Optional
from helical.models.base_models import HelicalBaseFineTuningHead, HelicalBaseFineTuningModel, HelicalDNAModel
from helical.models import ClassificationHead
from torch import optim
import torch
from torch.nn.modules import loss
from .hyena_dna_utils import HyenaDNADataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_scheduler
from sklearn.metrics import accuracy_score
import logging

logger = logging.getLogger(__name__)

class HyenaDNAFineTuningModel(HelicalBaseFineTuningModel):
    """HyenaDNA fine-tuning model.
    """
    def __init__(self, hyena_model: HelicalDNAModel, fine_tuning_head: Literal["classification"]|HelicalBaseFineTuningHead, output_size: Optional[int]=None):
        super(HyenaDNAFineTuningModel, self).__init__()
        self.config = hyena_model.config
        self.hyena_model = hyena_model.model
        if isinstance(fine_tuning_head, str):
            if fine_tuning_head == "classification":
                if output_size is None:
                    message = "The output_size must be specified for a classification head."
                    logger.error(message)
                    raise ValueError(message)
                fine_tuning_head = ClassificationHead(output_size)
            else:
                message = "The fine_tuning_head must be a valid HelicalBaseFineTuningHead"
                logger.error(message)
                raise ValueError(message)
        else:
            fine_tuning_head = fine_tuning_head
        fine_tuning_head.set_dim_size(self.config["d_model"])
        self.fine_tuning_head = fine_tuning_head

    def forward(self, x):
        x = self.hyena_model(x)[:, 0, :] # take cls
        x = self.fine_tuning_head(x)
        return x

    def train(        
        self,
        train_input_data: HyenaDNADataset,
        train_labels: list[int],     
        validation_input_data: HyenaDNADataset = None,
        validation_labels: list[int] = None,
        optimizer: optim = optim.AdamW,
        optimizer_params: dict = {'lr': 0.0001}, 
        loss_function: loss = loss.CrossEntropyLoss(), 
        epochs: int = 1,
        lr_scheduler_params: Optional[dict] = None):
        """Fine-tunes the Hyena-DNA model with different head modules. 

        Parameters
        ----------
        train_input_data : HyenaDNADataset
            A helical Hyena-DNA processed dataset for fine-tuning
        train_labels : list[int]
            The labels for the training data. These should be stored as unique per class integers.
        validation_input_data : HyenaDNADataset, default = None
            A helical Hyena-DNA processed dataset for per epoch validation. If this is not specified, no validation will be performed.
        validation_labels : list[int], default = None
            The labels for the validation data. These should be stored as unique per class integers.
        optimizer : torch.optim, default = torch.optim.AdamW
            The optimizer to be used for training.
        optimizer_params : dict
            The optimizer parameters to be used for the optimizer specified. This list should NOT include model parameters.
            e.g. optimizer_params = {'lr': 0.0001}
        loss_function : torch.nn.modules.loss, default = torch.nn.modules.loss.CrossEntropyLoss()
            The loss function to be used.
        epochs : int, optional, default = 10
            The number of epochs to train the model
        freeze_layers : int, optional, default = 0
            The number of layers to freeze.
        lr_scheduler_params : dict, default = None
            The learning rate scheduler parameters for the transformers get_scheduler method. The optimizer will be taken from the optimizer input and should not be included in the learning scheduler parameters. If not specified, no scheduler will be used.
            e.g. lr_scheduler_params = { 'name': 'linear', 'num_warmup_steps': 0, 'num_training_steps': 5 }
        """
        train_input_data.set_labels(train_labels)
        train_data_loader = DataLoader(train_input_data, batch_size=self.config["batch_size"], shuffle=True)
     
        if validation_input_data is not None and validation_labels is not None:
            validation_input_data.set_labels(validation_labels)
            validation_data_loader = DataLoader(validation_input_data, batch_size=5, shuffle=False)

        self.to(self.config["device"])

        optimizer = optimizer(self.parameters(), **optimizer_params)

        lr_scheduler = None
        if lr_scheduler_params is not None: 
            lr_scheduler = get_scheduler(optimizer=optimizer, **lr_scheduler_params)

        logger.info("Starting Fine-Tuning")
        for i in range(epochs):
            batch_loss = 0.0
            batches_processed = 0
            training_loop = tqdm(train_data_loader)
            for input_data, labels in training_loop:
                input_data = input_data.to(self.config["device"])
                labels = labels.to(self.config["device"])
                optimizer.zero_grad()
                output = self.forward(input_data)
                loss = loss_function(output, labels)
                loss.backward()
                batch_loss += loss.item()
                batches_processed += 1
                optimizer.step()
                
                training_loop.set_postfix({"loss": batch_loss/batches_processed})
                training_loop.set_description(f"Fine-Tuning: epoch {i+1}/{epochs}")
            
            if lr_scheduler is not None:
                lr_scheduler.step()

            if validation_input_data is not None and validation_labels is not None:
                with torch.no_grad():
                    validation_batches_processed = 0
                    accuracy = 0.0
                    validation_loop = tqdm(validation_data_loader, desc="Fine-Tuning Validation")
                    for input_data, val_labels in validation_loop:
                        input_data = input_data.to(self.config["device"])
                        val_labels = val_labels.to(self.config["device"])
                        output = self.forward(input_data)
                        validation_batches_processed += 1
                        accuracy += accuracy_score(val_labels.cpu(), torch.argmax(output, dim=1).cpu())
                        validation_loop.set_postfix({"accuracy": accuracy/validation_batches_processed})
        logger.info(f"Fine-Tuning Complete. Epochs: {epochs}")
            
            