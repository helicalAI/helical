from typing import Literal, Optional
from helical.models.base_models import (
    HelicalBaseFineTuningHead,
    HelicalBaseFineTuningModel,
)
from helical.models.hyena_dna import HyenaDNA, HyenaDNAConfig
from torch import optim
import torch
from torch.nn.modules import loss
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_scheduler
import logging
import numpy as np

logger = logging.getLogger(__name__)


class HyenaDNAFineTuningModel(HelicalBaseFineTuningModel, HyenaDNA):
    """HyenaDNA fine-tuning model.

    This class represents the HyenaDNA fine-tuning model, which is a long-range genomic foundation model pretrained on context lengths of up to 1 million tokens at single nucleotide resolution.

    Example
    ----------
    ```python
    from helical.models.hyena_dna import HyenaDNAFineTuningModel, HyenaDNAConfig
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_sequences = ["ACT"*20, "ATG"*20, "ATG"*20, "ACT"*20, "ATT"*20]
    labels = [0, 2, 2, 0, 1]

    hyena_dna_config = HyenaDNAConfig(batch_size=1, device=device)
    hyena_dna_fine_tune = HyenaDNAFineTuningModel(hyena_config=hyena_dna_config, fine_tuning_head="classification", output_size=3)

    train_dataset = hyena_dna_fine_tune.process_data(input_sequences)

    hyena_dna_fine_tune.train(train_dataset=train_dataset, train_labels=labels)

    outputs = hyena_dna_fine_tune.get_outputs(train_dataset)

    print(outputs.shape)
    ```

    Parameters
    ----------
    hyena_config : HyenaDNAConfig
        The HyenaDNA configs for fine-tuning model, the same configs that would be used to instantiate the standard HyenaDNA model.
    fine_tuning_head : Literal["classification", "regression"]|HelicalBaseFineTuningHead
        The fine-tuning head that is appended to the model. This can either be a string (options available: "classification", "regression") specifying the task or a custom fine-tuning head inheriting from HelicalBaseFineTuningHead.
    output_size : Optional[int], default=None
        The output size of the fine-tuning head. This is required if a predefined head is selected.

    Methods
    -------
    train(train_dataset: HyenaDNADataset, train_labels: list[int], validation_dataset: HyenaDNADataset = None, validation_labels: list[int] = None, optimizer: torch.optim, optimizer_params: dict, loss_function: torch.nn.modules.loss, epochs: int, lr_scheduler_params: dict = None)
        Fine-tunes the Hyena-DNA model with different head modules.
    get_outputs(input_data: HyenaDNADataset) -> np.ndarray
        Get the outputs of the fine-tuned model.
    """

    def __init__(
        self,
        hyena_config: HyenaDNAConfig,
        fine_tuning_head: (
            Literal["classification", "regression"] | HelicalBaseFineTuningHead
        ),
        output_size: Optional[int] = None,
    ):
        HelicalBaseFineTuningModel.__init__(self, fine_tuning_head, output_size)
        HyenaDNA.__init__(self, hyena_config)

        self.fine_tuning_head.set_dim_size(self.config["d_model"])

    def _forward(self, x):
        x = self.model(x)
        x = torch.mean(x, dim=1)
        x = self.fine_tuning_head(x)
        return x

    def train(
        self,
        train_dataset: Dataset,
        train_labels: list[int],
        validation_dataset: Dataset = None,
        validation_labels: list[int] = None,
        optimizer: optim = optim.AdamW,
        optimizer_params: dict = {"lr": 0.0001},
        loss_function: loss = loss.CrossEntropyLoss(),
        epochs: int = 1,
        lr_scheduler_params: Optional[dict] = None,
        shuffle: bool = True,
    ):
        """Fine-tunes the Hyena-DNA model with different head modules.

        Parameters
        ----------
        train_dataset : HyenaDNADataset
            A helical Hyena-DNA processed dataset for fine-tuning
        train_labels : list[int]
            The labels for the training data. These should be stored as unique per class integers.
        validation_dataset : HyenaDNADataset, default=None
            A helical Hyena-DNA processed dataset for per epoch validation. If this is not specified, no validation will be performed.
        validation_labels : list[int], default=None
            The labels for the validation data. These should be stored as unique per class integers.
        optimizer : torch.optim, default=torch.optim.AdamW
            The optimizer to be used for training.
        optimizer_params : dict
            The optimizer parameters to be used for the optimizer specified. This list should NOT include model parameters.
            e.g. optimizer_params = {'lr': 0.0001}
        loss_function : torch.nn.modules.loss, default=torch.nn.CrossEntropyLoss()
            The loss function to be used.
        epochs : int, optional, default=10
            The number of epochs to train the model for.
        lr_scheduler_params : dict, default=None
            The learning rate scheduler parameters for the transformers get_scheduler method. The optimizer will be taken from the optimizer input and should not be included in the learning scheduler parameters. If not specified, a constant learning rate will be used.
            e.g. lr_scheduler_params = { 'name': 'linear', 'num_warmup_steps': 0 }. num_steps will be calculated based on the number of epochs and the length of the training dataset.
        shuffle : bool, default=True
            Whether to shuffle the training data or not.
        """
        train_dataset = self._add_data_column(train_dataset, np.array(train_labels))
        train_dataloader = DataLoader(
            train_dataset,
            collate_fn=self._collate_fn,
            batch_size=self.config["batch_size"],
            shuffle=shuffle,
        )

        if validation_dataset is not None and validation_labels is not None:
            validation_dataset = self._add_data_column(
                validation_dataset, np.array(validation_labels)
            )
            validation_dataloader = DataLoader(
                validation_dataset,
                collate_fn=self._collate_fn,
                batch_size=self.config["batch_size"],
            )

        self.to(self.config["device"])
        self.model.train()
        self.fine_tuning_head.train()
        optimizer = optimizer(self.parameters(), **optimizer_params)

        lr_scheduler = None
        if lr_scheduler_params is not None:
            lr_scheduler = get_scheduler(
                optimizer=optimizer,
                num_training_steps=epochs * len(train_dataloader),
                **lr_scheduler_params,
            )

        logger.info("Starting Fine-Tuning")
        for i in range(epochs):
            batch_loss = 0.0
            batches_processed = 0
            training_loop = tqdm(train_dataloader)
            for batch in training_loop:
                input_data = batch["input_ids"].to(self.config["device"])
                labels = batch["labels"].to(self.config["device"])
                optimizer.zero_grad()
                output = self._forward(input_data)
                loss = loss_function(output, labels)
                batch_loss += loss.item()
                loss.backward()
                batches_processed += 1
                optimizer.step()

                training_loop.set_postfix({"loss": batch_loss / batches_processed})
                training_loop.set_description(f"Fine-Tuning: epoch {i+1}/{epochs}")

                if lr_scheduler is not None:
                    lr_scheduler.step()

            if validation_dataset is not None and validation_labels is not None:
                with torch.no_grad():
                    validation_batches_processed = 0
                    val_loss = 0.0
                    validation_loop = tqdm(
                        validation_dataloader, desc="Fine-Tuning Validation"
                    )
                    for batch in validation_loop:
                        input_data = batch["input_ids"].to(self.config["device"])
                        val_labels = batch["labels"].to(self.config["device"])
                        output = self._forward(input_data)
                        validation_batches_processed += 1
                        val_loss += loss_function(output, val_labels).item()
                        validation_loop.set_postfix(
                            {"val_loss": val_loss / validation_batches_processed}
                        )
        logger.info(f"Fine-Tuning Complete. Epochs: {epochs}")

    def get_outputs(self, dataset: Dataset) -> np.ndarray:
        """Get the outputs of the fine-tuned model.

        Parameters
        ----------
        dataset : HyenaDNADataset
            The input data to get the outputs for.

        Returns
        -------
        np.ndarray
            The outputs of the model
        """
        data_loader = DataLoader(
            dataset, collate_fn=self._collate_fn, batch_size=self.config["batch_size"]
        )

        self.to(self.config["device"])
        self.model.eval()
        self.fine_tuning_head.eval()

        batch_loop = tqdm(data_loader)
        outputs = []
        for batch in batch_loop:
            input_data = batch["input_ids"].to(self.config["device"])
            output = self._forward(input_data)
            outputs.append(output.detach().cpu().numpy())

        return np.vstack(outputs)

    def _add_data_column(
        self, dataset: Dataset, data: list, column_name: str = "labels"
    ) -> Dataset:
        """
        Add a column to the dataset.

        Parameters
        ----------
        dataset : Dataset
            The dataset to add the column to.
        data : list
            The data to add to the column.
        column_name : str, optional, default="labels"
            The name of the column to add.
        """
        if len(data.shape) > 1:
            for i in range(len(data[0])):  # Assume all inner lists are the same length
                dataset = dataset.add_column(f"{column_name}", [row[i] for row in data])
        else:  # If 1D
            dataset = dataset.add_column(column_name, data)
        return dataset
