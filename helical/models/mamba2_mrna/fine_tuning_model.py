from typing import Literal, Optional
from helical.models.base_models import (
    HelicalBaseFineTuningHead,
    HelicalBaseFineTuningModel,
)
from helical.models.mamba2_mrna import Mamba2mRNA, Mamba2mRNAConfig
from datasets import Dataset
from transformers import get_scheduler
import torch
from torch import optim
from torch.nn.modules import loss
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np

import logging

logger = logging.getLogger(__name__)


class Mamba2mRNAFineTuningModel(HelicalBaseFineTuningModel, Mamba2mRNA):
    """Mamba2mRNAFineTuningModel
    Fine-tuning model for the Mamba2-mRNA model. This model can be used to fine-tune the Mamba2-mRNA model on downstream tasks.

    Example
    ----------
    ```python
    from helical.models.mamba2_mrna import Mamba2mRNAFineTuningModel, Mamba2mRNAConfig
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_sequences = ["ACUG"*20, "AUGC"*20, "AUGC"*20, "ACUG"*20, "AUUG"*20]
    labels = [0, 2, 2, 0, 1]

    mamba2_mrna_config = Mamba2mRNAConfig(batch_size=5, device=device, max_length=100)
    mamba2_mrna_fine_tune = Mamba2mRNAFineTuningModel(mamba2_mrna_config=mamba2_mrna_config, fine_tuning_head="classification", output_size=3)

    train_dataset = mamba2_mrna_fine_tune.process_data(input_sequences)

    mamba2_mrna_fine_tune.train(train_dataset=train_dataset, train_labels=labels)

    outputs = mamba2_mrna_fine_tune.get_outputs(train_dataset)
    print(outputs.shape)
    ```

    Parameters
    ----------
    mamba2_mrna_config : Mamba2mRNAConfigRConfig
        The configuration object for the Mamba2-mRNA model. The same config object can be used for both the Mamba2mRNA and Mamba2mRNAFineTuningModel.
    fine_tuning_head : Literal["classification", "regression"] | HelicalBaseFineTuningHead
        The type of fine-tuning head to use for the model. This can be either a classification or regression head, or a custom fine-tuning head.
    output_size : Optional[int]
        The output size of the fine-tuning model. This is required if the fine_tuning_head is a string specified task. For a classification task this is number of unique classes.

    Methods
    ----------
    train(train_dataset, train_labels, optimizer, optimizer_params, loss_function, epochs, freeze_layers, validation_dataset, validation_labels, lr_scheduler_params)
        Fine-tunes the Mamba2-mRNA model on the given dataset.
    get_outputs(dataset)
        Returns the outputs of the model for the given dataset.
    """

    def __init__(
        self,
        mamba2_mrna_config: Mamba2mRNAConfig,
        fine_tuning_head: (
            Literal["classification", "regression"] | HelicalBaseFineTuningHead
        ),
        output_size: Optional[int] = None,
    ):
        HelicalBaseFineTuningModel.__init__(self, fine_tuning_head, output_size)
        Mamba2mRNA.__init__(self, mamba2_mrna_config)

        self.fine_tuning_head.set_dim_size(self.pretrained_config.hidden_size)

    def _forward(self, input_ids, special_tokens_mask, attention_mask):
        """Forward pass for the Mamba2-mRNA fine-tuning model.

        Parameters
        ----------
        input_ids : torch.Tensor
            The input_ids tensor for the model.
        special_tokens_mask : torch.Tensor
            The special_tokens_mask tensor for the model.
        attention_mask : torch.Tensor
            The attention_mask tensor for the model.

        Returns
        -------
        torch.Tensor
            The output tensor from the model."""
        outputs = self.model(
            input_ids,
            special_tokens_mask=special_tokens_mask,
            attention_mask=attention_mask,
        )
        last_hidden_states = outputs[0]

        if input_ids is not None:
            batch_size, _ = input_ids.shape[:2]

        if self.pretrained_config.pad_token_id is None and batch_size > 1:
            raise ValueError(
                "Cannot handle batch sizes > 1 if no padding token is defined."
            )

        if self.pretrained_config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = (
                    torch.eq(input_ids, self.pretrained_config.pad_token_id)
                    .int()
                    .argmax(-1)
                    - 1
                )
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(last_hidden_states.device)
            else:
                sequence_lengths = -1

        mask = (
            torch.arange(last_hidden_states.size(1), device=last_hidden_states.device)[
                None, :
            ]
            < sequence_lengths[:, None]
        )
        masked_tensor = last_hidden_states * mask.unsqueeze(-1)
        sum_tensor = masked_tensor.sum(dim=1)
        mean_last_hidden_states = sum_tensor / sequence_lengths.unsqueeze(-1).float()

        head_outputs = self.fine_tuning_head(mean_last_hidden_states)
        return head_outputs

    def train(
        self,
        train_dataset: Dataset,
        train_labels: np.ndarray,
        optimizer: optim = optim.AdamW,
        optimizer_params: dict = {"lr": 0.0001},
        loss_function: loss = loss.CrossEntropyLoss(),
        epochs: int = 1,
        trainable_layers: int = 2,
        shuffle: bool = True,
        validation_dataset: Optional[Dataset] = None,
        validation_labels: Optional[np.ndarray] = None,
        lr_scheduler_params: Optional[dict] = None,
    ):
        """Fine-tunes the Mamba2-mRNA model on the given dataset.

        Parameters
        ----------
        train_dataset : Dataset
            A helical processed dataset for fine-tuning.
        train_labels : np.ndarray
            The labels for the training dataset.
        optimizer : torch.optim, default=torch.optim.AdamW
            The optimizer to be used for training.
        optimizer_params : dict, optional, default={'lr': 0.0001}
            The optimizer parameters to be used for the optimizer specified. This list should NOT include model parameters.
            e.g. optimizer_params = {'lr': 0.0001}
        loss_function : torch.nn.modules.loss, default=torch.nn.modules.loss.CrossEntropyLoss()
            The loss function to be used.
        epochs : int, optional, default=10
            The number of epochs to train the model
        trainable_layers : int, optional, default=2
            The number of layers to train in the model. The last n layers will be trained and the rest will be frozen.
        shuffle : bool, default=True
            Whether to shuffle the training dataset during training.
        validation_dataset : Dataset, default=None
            A helical processed dataset for per epoch validation. If this is not specified, no validation will be performed.
        lr_scheduler_params : dict, default=None
            The learning rate scheduler parameters for the transformers get_scheduler method. The optimizer will be taken from the optimizer input and should not be included in the learning scheduler parameters. If not specified, no scheduler will be used.
            e.g. lr_scheduler_params = { 'name': 'linear', 'num_warmup_steps': 0 }. num_steps will be calculated based on the number of epochs and the length of the training dataset.
        """
        # initialise optimizer
        optimizer = optimizer(self.parameters(), **optimizer_params)

        # set labels for the dataset
        train_dataset = self._add_data_column(
            train_dataset, "labels", np.array(train_labels)
        )
        if validation_labels is not None and validation_dataset is not None:
            validation_dataset = self._add_data_column(
                validation_dataset, "labels", np.array(validation_labels)
            )

        if trainable_layers > 0:
            logger.info(
                f"Unfreezing the last {trainable_layers} layers of the Mamba2_mRNA model."
            )

            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.layers[-trainable_layers:].parameters():
                param.requires_grad = True

        self.to(self.config["device"])

        self.model.train()
        self.fine_tuning_head.train()

        train_dataloader = DataLoader(
            train_dataset,
            collate_fn=self._collate_fn,
            batch_size=self.config["batch_size"],
            shuffle=shuffle,
        )

        # initialise lr_scheduler
        lr_scheduler = None
        if lr_scheduler_params is not None:
            lr_scheduler = get_scheduler(
                optimizer=optimizer,
                num_training_steps=epochs * len(train_dataloader),
                **lr_scheduler_params,
            )

        if validation_dataset is not None:
            validation_dataloader = DataLoader(
                validation_dataset,
                collate_fn=self._collate_fn,
                batch_size=self.config["batch_size"],
            )

        logger.info("Starting Fine-Tuning")
        for j in range(epochs):
            training_loop = tqdm(train_dataloader, desc="Fine-Tuning")
            batch_loss = 0.0
            batches_processed = 0
            for batch in training_loop:
                input_ids = batch["input_ids"].to(self.config["device"])
                special_tokens_mask = batch["special_tokens_mask"].to(
                    self.config["device"]
                )
                attention_mask = batch["attention_mask"].to(self.config["device"])
                labels = batch["labels"].to(self.config["device"])

                outputs = self._forward(
                    input_ids,
                    special_tokens_mask=special_tokens_mask,
                    attention_mask=attention_mask,
                )

                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                batch_loss += loss.item()
                batches_processed += 1
                training_loop.set_postfix({"loss": batch_loss / batches_processed})
                training_loop.set_description(f"Fine-Tuning: epoch {j+1}/{epochs}")

                del batch
                del outputs

                if lr_scheduler is not None:
                    lr_scheduler.step()

            del training_loop

            if validation_dataset is not None:
                testing_loop = tqdm(
                    validation_dataloader, desc="Fine-Tuning Validation"
                )
                val_loss = 0.0
                count = 0.0
                for test_batch in testing_loop:
                    input_ids = test_batch["input_ids"].to(self.config["device"])
                    special_tokens_mask = test_batch["special_tokens_mask"].to(
                        self.config["device"]
                    )
                    attention_mask = test_batch["attention_mask"].to(
                        self.config["device"]
                    )
                    labels = test_batch["labels"].to(self.config["device"])

                    with torch.no_grad():
                        outputs = self._forward(
                            input_ids,
                            special_tokens_mask=special_tokens_mask,
                            attention_mask=attention_mask,
                        )

                    val_loss += loss_function(outputs, labels).item()
                    count += 1.0
                    testing_loop.set_postfix({"val_loss": val_loss / count})

                    del test_batch
                    del outputs

                del testing_loop

        logger.info(f"Fine-Tuning Complete. Epochs: {epochs}")

    def get_outputs(self, dataset: Dataset) -> np.ndarray:
        """
        Returns the outputs of the model for the given dataset.

        Parameters
        ----------
        dataset : Dataset
            The dataset object returned by the `process_data` function.

        Returns
        ----------
        np.ndarray
            The outputs of the model for the given dataset.
        """
        self.model.eval()
        self.fine_tuning_head.eval()

        dataloader = DataLoader(
            dataset,
            collate_fn=self._collate_fn,
            batch_size=self.config["batch_size"],
            shuffle=False,
        )
        outputs = []

        self.model.to(self.config["device"])

        progress_bar = tqdm(dataloader, desc="Generating outputs")
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(self.config["device"])
            special_tokens_mask = batch["special_tokens_mask"].to(self.config["device"])
            attention_mask = batch["attention_mask"].to(self.config["device"])

            with torch.no_grad():
                output = self._forward(
                    input_ids,
                    special_tokens_mask=special_tokens_mask,
                    attention_mask=attention_mask,
                )

            outputs.append(output.cpu().numpy())

            del batch
            del output

        return np.concatenate(outputs)

    def _add_data_column(self, dataset, column_name, data):
        if len(data.shape) > 1:
            for i in range(len(data[0])):  # Assume all inner lists are the same length
                dataset = dataset.add_column(f"{column_name}", [row[i] for row in data])
        else:  # If 1D
            dataset = dataset.add_column(column_name, data)
        return dataset
