from helical.models.uce.uce_dataset import UCEDataset
import numpy as np
import torch
from torch import optim
from torch.nn.modules import loss
from torch.utils.data import DataLoader
from helical.models.base_models import HelicalBaseFineTuningHead
from helical.models.base_models import HelicalBaseFineTuningModel
from helical.models.uce import UCE, UCEConfig
from typing import Literal, Optional
from tqdm import tqdm
from transformers import get_scheduler
import logging

logger = logging.getLogger(__name__)

class UCEFineTuningModel(HelicalBaseFineTuningModel, UCE):
    """
    Fine-tuning model for the UCE model.

    Parameters
    ----------
    uce_config : UCE
        The UCE configs for fine-tuning model, the same configs that would be used to instantiate the standard UCE model.
    fine_tuning_head : Literal["classification", "regression"] | HelicalBaseFineTuningHead
        The fine-tuning head that is appended to the model. This can either be a string (options available: "classification", "regression") specifying the task or a custom fine-tuning head inheriting from HelicalBaseFineTuningHead.
    output_size : Optional[int]
        The output size of the fine-tuning model. This is required if the fine_tuning_head is a string specified task. For a classification task this is number of unique classes.

    Methods
    -------
    _forward(input_gene_ids: torch.Tensor, data_dict: dict, src_key_padding_mask: torch.Tensor, use_batch_labels: bool, device: str) -> torch.Tensor
        The forward method of the fine-tuning model.
    train(train_input_data: UCEDataset, train_labels: np.ndarray, validation_input_data = None, validation_labels = None, optimizer: optim = optim.AdamW, optimizer_params: dict = {'lr': 0.0001}, loss_function: loss = loss.CrossEntropyLoss(), epochs: int = 1, lr_scheduler_params: Optional[dict] = None)
        Fine-tunes the UCE model with different head modules.
    get_outputs(dataset: UCEDataset) -> np.ndarray
        Get the outputs of the fine-tuned model on a UCE processed dataset.

    """
    def __init__(self, 
                 uce_config: UCEConfig, 
                 fine_tuning_head: Literal["classification"] | HelicalBaseFineTuningHead, 
                 output_size: Optional[int]=None):
        
        HelicalBaseFineTuningModel.__init__(self, fine_tuning_head, output_size)
        UCE.__init__(self, uce_config)
        
        self.fine_tuning_head.set_dim_size(self.config["embsize"])

    def _forward(self, batch_sentences: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward method of the fine-tuning model.

        Parameters
        ----------
        batch_sentences : torch.Tensor
            The input tensor of the fine-tuning model.
        mask : torch.Tensor
            The mask tensor for the input tensor.
        
        Returns
        -------
        torch.Tensor
            The output tensor of the fine-tuning model.
        """
        _, embeddings = self.model.forward(batch_sentences, mask=mask)
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
            embeddings = self.accelerator.gather_for_metrics((embeddings))
            if self.accelerator.is_main_process:
                embeddings = embeddings
        else:
            embeddings = embeddings
        output = self.fine_tuning_head(embeddings)
        return output
    
    def train(
            self,
            train_input_data: UCEDataset, 
            train_labels: np.ndarray,     
            validation_input_data = None,
            validation_labels = None,
            optimizer: optim = optim.AdamW,
            optimizer_params: dict = {'lr': 0.0001}, 
            loss_function: loss = loss.CrossEntropyLoss(), 
            epochs: int = 1,
            # freeze_layers: int = 0,
            lr_scheduler_params: Optional[dict] = None):
        """
        Fine-tunes the UCE model with different head modules. 

        Parameters
        ----------
        train_input_data : Dataset
            A helical UCE processed dataset for fine-tuning
        train_labels : ndarray
            The labels for the training data. These should be stored as unique per class integers.
        validation_input_data : Dataset, default = None
            A helical UCE processed dataset for per epoch validation. If this is not specified, no validation will be performed.
        validation_labels : ndarray, default = None,
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
        lr_scheduler_params : dict, default = None
            The learning rate scheduler parameters for the transformers get_scheduler method. The optimizer will be taken from the optimizer input and should not be included in the learning scheduler parameters. If not specified, no scheduler will be used.
            e.g. lr_scheduler_params = { 'name': 'linear', 'num_warmup_steps': 0, 'num_training_steps': 5 }

        Returns
        -------
        torch.nn.Module
            The fine-tuned model.
        """
        batch_size = self.config["batch_size"]
        dataloader = DataLoader(train_input_data, 
                                batch_size=batch_size, 
                                shuffle=False,
                                collate_fn=train_input_data.collator_fn,
                                num_workers=0)
        
        if validation_input_data is not None:
            validation_dataloader = DataLoader(validation_input_data, 
                        batch_size=batch_size, 
                        shuffle=False,
                        collate_fn=validation_input_data.collator_fn,
                        num_workers=0)

        if self.accelerator is not None:
            dataloader = self.accelerator.prepare(dataloader)
            if validation_input_data is not None:
                validation_dataloader = self.accelerator.prepare(validation_dataloader)

        self.model.train()
        self.fine_tuning_head.train()

        # disable progress bar if not the main process
        # if self.accelerator is not None:
        #     pbar = tqdm(dataloader, disable=not self.accelerator.is_local_main_process)
        # else:
        #     pbar = tqdm(dataloader)

        self.to(self.device)

        optimizer = optimizer(self.parameters(), **optimizer_params)

        lr_scheduler = None
        if lr_scheduler_params is not None: 
            lr_scheduler = get_scheduler(optimizer=optimizer, **lr_scheduler_params)

        logger.info("Starting Fine-Tuning")
        for j in range(epochs):
            batch_count = 0
            batch_loss = 0.0
            batches_processed = 0
            training_loop = tqdm(dataloader, desc="Fine-Tuning")
            for batch in training_loop:
                batch_sentences, mask, idxs = batch[0], batch[1], batch[2]
                batch_sentences = batch_sentences.permute(1, 0)
                if self.config["multi_gpu"]:
                    batch_sentences = self.model.module.pe_embedding(batch_sentences.long())
                else:
                    batch_sentences = self.model.pe_embedding(batch_sentences.long())
                batch_sentences = torch.nn.functional.normalize(batch_sentences, dim=2)  # normalize token outputs
                output = self._forward(batch_sentences, mask=mask)
                labels = torch.tensor(train_labels[batch_count: batch_count + self.config["batch_size"]], device=self.device)
                batch_count += self.config["batch_size"]
                loss = loss_function(output, labels)
                loss.backward()
                batch_loss += loss.item()
                batches_processed += 1

                optimizer.step()
                optimizer.zero_grad()

                training_loop.set_postfix({"loss": batch_loss/batches_processed})
                training_loop.set_description(f"Fine-Tuning: epoch {j+1}/{epochs}")

            if lr_scheduler is not None:
                lr_scheduler.step()

            if validation_input_data is not None:
                testing_loop = tqdm(validation_dataloader, desc="Fine-Tuning Validation")
                val_loss = 0.0
                count = 0.0
                validation_batch_count = 0
                for validation_data in testing_loop:
                    batch_sentences, mask, idxs = validation_data[0], validation_data[1], validation_data[2]
                    batch_sentences = batch_sentences.permute(1, 0)
                    if self.config["multi_gpu"]:
                        batch_sentences = self.model.module.pe_embedding(batch_sentences.long())
                    else:
                        batch_sentences = self.model.pe_embedding(batch_sentences.long())
                    batch_sentences = torch.nn.functional.normalize(batch_sentences, dim=2)  # normalize token outputs
                    output = self._forward(batch_sentences, mask=mask)
                    val_labels = torch.tensor(validation_labels[validation_batch_count: validation_batch_count + self.config["batch_size"]], device=self.device)
                    validation_batch_count += self.config["batch_size"]
                    val_loss += loss_function(output, val_labels).item()
                    count += 1.0
                    testing_loop.set_postfix({"val_loss": val_loss/count})
        logger.info(f"Fine-Tuning Complete. Epochs: {epochs}")
        self.model.eval()
        self.fine_tuning_head.eval()

    def get_outputs(
        self,
        dataset: UCEDataset
    ) -> np.ndarray:
        """
        Get the outputs of the fine-tuned model on a dataset.

        Parameters
        ----------
        dataset : UCEDataset
            The dataset to get the outputs for.
        
        Returns
        -------
        np.ndarray
            The outputs of the model.
        """
        self.to(self.device)

        batch_size = self.config["batch_size"]
        dataloader = DataLoader(dataset, 
                                batch_size=batch_size, 
                                shuffle=False,
                                collate_fn=dataset.collator_fn,
                                num_workers=0)
        

        if self.accelerator is not None:
            dataloader = self.accelerator.prepare(dataloader)
        
        self.model.eval()
        self.fine_tuning_head.eval()

        testing_loop = tqdm(dataloader, desc="Fine-Tuning Validation")
        outputs = []
        for validation_data in testing_loop:
            batch_sentences, mask, idxs = validation_data[0], validation_data[1], validation_data[2]
            batch_sentences = batch_sentences.permute(1, 0)
            if self.config["multi_gpu"]:
                batch_sentences = self.model.module.pe_embedding(batch_sentences.long())
            else:
                batch_sentences = self.model.pe_embedding(batch_sentences.long())
            batch_sentences = torch.nn.functional.normalize(batch_sentences, dim=2)  # normalize token outputs
            output = self._forward(batch_sentences, mask=mask)
            outputs.append(output.detach().cpu().numpy())
        
        return np.vstack(outputs)