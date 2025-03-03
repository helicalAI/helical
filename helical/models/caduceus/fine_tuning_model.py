from typing import Literal, Optional
from helical.models.base_models import (
    HelicalBaseFineTuningHead,
    HelicalBaseFineTuningModel,
)
from helical.models.caduceus import Caduceus, CaduceusConfig
from datasets import Dataset
from transformers import get_scheduler
import torch
from torch import optim
from torch.nn.modules import loss
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np

import logging

LOGGER = logging.getLogger(__name__)


class CaduceusFineTuningModel(HelicalBaseFineTuningModel, Caduceus):
    """CaduceusFineTuningModel
    This model can be used to fine-tune the Caduceus model on a downstream task.

    Example
    ----------
    ```python
    from helical.models.caduceus import CaduceusFineTuningModel, CaduceusConfig

    input_sequences = ["ACT"*20, "ATG"*20, "ATG"*20, "CTG"*20, "TTG"*20]
    labels = [0, 2, 2, 0, 1]

    caduceus_config = CaduceusConfig(model_name="caduceus-ph-4L-seqlen-1k-d118", batch_size=5)
    caduceus_fine_tune = CaduceusFineTuningModel(caduceus_config=caduceus_config, fine_tuning_head="classification", output_size=3)

    train_dataset = caduceus_fine_tune.process_data(input_sequences)

    caduceus_fine_tune.train(train_dataset=train_dataset, train_labels=labels)

    outputs = caduceus_fine_tune.get_outputs(train_dataset)
    print(outputs.shape)
    ```

    Parameters
    ----------
    caduceus_config : CaduceusConfig
        The configuration object for the Caduceus model. The same config object can be used for both the Caduceus and CaduceusFineTuningModel.
    fine_tuning_head : Literal["classification", "regression"] | HelicalBaseFineTuningHead
        The type of fine-tuning head to use for the model. This can be either a classification or regression head, or a custom fine-tuning head.
    output_size : Optional[int]
        The output size of the fine-tuning model. This is required if the fine_tuning_head is a string specified task. For a classification task this is number of unique classes.

    Methods
    ----------
    train(train_dataset, train_labels, optimizer, optimizer_params, loss_function, epochs, freeze_layers, validation_dataset, validation_labels, lr_scheduler_params)
        Fine-tunes the Caduceus model on the given dataset.
    get_outputs(dataset)
        Returns the outputs of the model for the given dataset.
    """

    def __init__(
        self,
        caduceus_config: CaduceusConfig,
        fine_tuning_head: (
            Literal["classification", "regression"] | HelicalBaseFineTuningHead
        ),
        output_size: Optional[int] = None,
    ):
        HelicalBaseFineTuningModel.__init__(self, fine_tuning_head, output_size)
        Caduceus.__init__(self, caduceus_config)

        self.fine_tuning_head.set_dim_size(self.config["embedding_size"])

    def _forward(
        self,
        input_ids: torch.LongTensor = None,
        output_hidden_states: Optional[bool] = None,
        conjoin_train: bool = False,
        conjoin_eval: bool = False,
        training: bool = False,
    ):
        # Get hidden representations from the backbone
        if self.model.config.rcps:  # Hidden states have 2 * d_model channels for RCPS
            transformer_outputs = self.model(
                input_ids=input_ids, output_hidden_states=output_hidden_states
            )
            hidden_states = torch.stack(
                [
                    transformer_outputs[0][..., : self.model.config.d_model],
                    torch.flip(
                        transformer_outputs[0][..., self.model.config.d_model :],
                        dims=[1, 2],
                    ),
                ],
                dim=-1,
            )
        elif conjoin_train or (
            conjoin_eval and not training
        ):  # For conjoining / post-hoc conjoining
            assert input_ids is not None, "`input_ids` must be provided for conjoining."
            assert (
                input_ids.ndim == 3
            ), "`input_ids` must be 3D tensor: channels corresponds to forward and rc strands."
            transformer_outputs = self.model(
                input_ids[..., 0], output_hidden_states=output_hidden_states
            )
            transformer_outputs_rc = self.model(
                input_ids[..., 1], output_hidden_states=output_hidden_states
            )
            # Stack along channel dimension (dim=-1)
            hidden_states = torch.stack(
                [transformer_outputs[0], transformer_outputs_rc[0]], dim=-1
            )
        else:
            transformer_outputs = self.model(
                input_ids, output_hidden_states=output_hidden_states
            )
            hidden_states = transformer_outputs[0]

        # Pool and get logits
        pooled_hidden_states = self._pool_hidden_states(
            hidden_states=hidden_states, sequence_length_dim=1
        )
        # Potentially run `fine_tuning_head` twice (with parameters shared) for conjoining
        if (
            hidden_states.ndim == 4
        ):  # bsz, seq_len, hidden_dim, 2 where last channel has the stacked fwd and rc reps
            logits_fwd = self.fine_tuning_head(pooled_hidden_states[..., 0])
            logits_rc = self.fine_tuning_head(pooled_hidden_states[..., 1])
            logits = (logits_fwd + logits_rc) / 2
        else:
            logits = self.fine_tuning_head(pooled_hidden_states)

        return logits

    def train(
        self,
        train_dataset: Dataset,
        train_labels: np.ndarray,
        optimizer: optim = optim.AdamW,
        optimizer_params: dict = {"lr": 0.0001},
        loss_function: loss = loss.CrossEntropyLoss(),
        epochs: int = 1,
        trainable_layers: int = 2,
        validation_dataset: Optional[Dataset] = None,
        validation_labels: Optional[np.ndarray] = None,
        lr_scheduler_params: Optional[dict] = None,
        conjoin_train: bool = False,
        conjoin_eval: bool = False,
    ):
        """Fine-tunes the Caduceus model on the given dataset.

        Parameters
        ----------
        train_dataset : Dataset
            A helical processed dataset for fine-tuning
        optimizer : torch.optim, default=torch.optim.AdamW
            The optimizer to be used for training.
        optimizer_params : dict, optional, default={'lr': 0.0001}
            The optimizer parameters to be used for the optimizer specified. This list should NOT include model parameters.
            e.g. optimizer_params = {'lr': 0.0001}
        loss_function : torch.nn.modules.loss, default=torch.nn.modules.loss.CrossEntropyLoss()
            The loss function to be used.
        train_labels : np.ndarray
            training labels for the dataset.
        epochs : int, optional, default=10
            The number of epochs to train the model
        trainable_layers : int, optional, default=2
            The number of layers to train in the model. The last n layers will be trained and the rest will be frozen.
        validation_dataset : Dataset, default=None
            A helical processed dataset for per epoch validation. If this is not specified, no validation will be performed.
        lr_scheduler_params : dict, default=None
            The learning rate scheduler parameters for the transformers get_scheduler method. The optimizer will be taken from the optimizer input and should not be included in the learning scheduler parameters. If not specified, no scheduler will be used.
            e.g. lr_scheduler_params = { 'name': 'linear', 'num_warmup_steps': 0 }. num_steps will be calculated based on the number of epochs and the length of the training dataset.
        conjoin_train : bool, default=False
            Whether to conjoin the forward and reverse complement sequences during training.
        conjoin_eval : bool, default=False
            Whether to conjoin the forward and reverse complement sequences during evaluation.
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
            LOGGER.info(
                f"Unfreezing the last {trainable_layers} layers of the Caduceus model."
            )

            for param in self.model.backbone.parameters():
                param.requires_grad = False
            for param in self.model.backbone.layers[-trainable_layers:].parameters():
                param.requires_grad = True

        self.to(self.config["device"])

        train_dataloader = DataLoader(
            train_dataset,
            collate_fn=self._collate_fn,
            batch_size=self.config["batch_size"],
            num_workers=self.config["nproc"],
        )

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
                num_workers=self.config["nproc"],
            )

        LOGGER.info("Starting Fine-Tuning")
        for j in range(epochs):
            training_loop = tqdm(train_dataloader, desc="Fine-Tuning")
            self.model.train()
            self.fine_tuning_head.train()
            batch_loss = 0.0
            batches_processed = 0
            for batch in training_loop:
                input_ids = batch["input_ids"].to(self.config["device"])
                labels = batch["labels"].to(self.config["device"])

                outputs = self._forward(
                    input_ids=input_ids,
                    conjoin_train=conjoin_train,
                    conjoin_eval=conjoin_eval,
                    training=True,
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
                self.model.eval()
                self.fine_tuning_head.eval()
                val_loss = 0.0
                count = 0.0
                for test_batch in testing_loop:
                    input_ids = test_batch["input_ids"].to(self.config["device"])
                    labels = test_batch["labels"].to(self.config["device"])

                    with torch.no_grad():
                        outputs = self._forward(
                            input_ids=input_ids,
                            conjoin_train=conjoin_train,
                            conjoin_eval=conjoin_eval,
                            training=False,
                        )

                    val_loss += loss_function(outputs, labels).item()
                    count += 1.0
                    testing_loop.set_postfix({"val_loss": val_loss / count})

                    del test_batch
                    del outputs

                del testing_loop
                self.model.train()
                self.fine_tuning_head.train()

        LOGGER.info(f"Fine-Tuning Complete. Epochs: {epochs}")

    def get_outputs(self, dataset: Dataset, conjoin: bool = False) -> np.ndarray:
        """Get the embeddings for the tokenized sequence.

        Parameters
        ----------
        dataset : Dataset
            The output dataset from `process_data`.
        conjoin : bool, default=False
            Whether to conjoin the forward and reverse complement sequences.
        """
        dataloader = DataLoader(
            dataset,
            collate_fn=self._collate_fn,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config["nproc"],
        )
        outputs = []

        self.model.to(self.config["device"])
        self.model.eval()
        self.fine_tuning_head.eval()

        progress_bar = tqdm(dataloader, desc="Generating outputs")
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(self.config["device"])

            with torch.no_grad():
                output = self._forward(
                    input_ids=input_ids, conjoin_eval=conjoin, training=False
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
