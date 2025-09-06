from typing import Literal, Optional
from helical.models.state.state_config import stateConfig
import torch
from torch import optim
from torch.nn.modules import loss
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import get_scheduler
from helical.models.base_models import HelicalBaseFineTuningHead
from helical.models.state._perturb_utils.state_transition_model import (
    StateTransitionPerturbationModel,
)
from helical.models.base_models import HelicalBaseFineTuningModel
import logging
import numpy as np

logger = logging.getLogger(__name__)


class stateFineTuningModel(HelicalBaseFineTuningModel):
    def __init__(
        self,
        configurer: stateConfig = None,
        fine_tuning_head: Literal["classification"] | HelicalBaseFineTuningHead = "classification",
        output_size: Optional[int] = None,
    ):
        if configurer is None:
            self.config = stateConfig().config["finetune"]
        else:
            self.config = configurer.config["finetune"]

        HelicalBaseFineTuningModel.__init__(self, fine_tuning_head, output_size)
        
        # Load the pre-trained state model
        self.model = StateTransitionPerturbationModel.load_from_checkpoint(
            self.config["checkpoint_path"]
        )

        # Get the actual output dimension from the loaded model
        self.embed_dim = self.model.output_dim
        self.cell_sentence_len = self.model.cell_sentence_len
        self.device = next(self.model.parameters()).device
        self.fine_tuning_head.set_dim_size(self.embed_dim)

    def _forward(self, data_dict: dict) -> torch.Tensor:
        """
        Forward method for fine-tuning.
        """
        # Move data to device
        data_dict = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in data_dict.items()
        }
        
        # Use the state model's forward method to get embeddings
        embeddings = self.model.forward(data_dict, padded=True)  # Shape: [B*S, output_dim]

        batch_size = data_dict["pert_emb"].shape[0] // self.cell_sentence_len
        embeddings = embeddings.reshape(
            batch_size, self.cell_sentence_len, self.embed_dim
        )
        cell_embeddings = embeddings.mean(dim=1)  # Shape: [B, output_dim]

        output = self.fine_tuning_head(cell_embeddings)
        return output

    def train(
        self,
        train_input_data: torch.Tensor,
        train_labels: np.ndarray,
        validation_input_data=None,
        validation_labels=None,
        optimizer: optim = optim.AdamW,
        optimizer_params: dict = {"lr": 0.0001},
        loss_function: loss = loss.CrossEntropyLoss(),
        epochs: int = 1,
        lr_scheduler_params: Optional[dict] = None,
    ):
        """Fine-tunes the state model with different head modules.

        Parameters
        ----------
        train_input_data : Dataset
            A state model processed dataset for fine-tuning
        train_labels : ndarray
            The labels for the training data. These should be stored as unique per class integers.
        validation_input_data : Dataset, default = None
            A state model processed dataset for per epoch validation. If this is not specified, no validation will be performed.
        validation_labels : ndarray, default = None
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
        """

        # Simple collator for state model
        def state_collator(batch):
            """Collate function for state model batches"""
            return {
                "pert_emb": torch.stack([item["pert_emb"] for item in batch]),
                "ctrl_cell_emb": torch.stack([item["ctrl_cell_emb"] for item in batch]),
                "batch": (
                    torch.stack([item["batch"] for item in batch])
                    if "batch" in batch[0]
                    else None
                ),
                "pert_name": [item["pert_name"] for item in batch],
            }

        data_loader = DataLoader(
            train_input_data,
            batch_size=self.config["batch_size"],
            sampler=SequentialSampler(train_input_data),
            collate_fn=state_collator,
            drop_last=False,
            pin_memory=True,
        )

        if validation_input_data is not None:
            validation_data_loader = DataLoader(
                validation_input_data,
                batch_size=self.config["batch_size"],
                sampler=SequentialSampler(validation_input_data),
                collate_fn=state_collator,
                drop_last=False,
                pin_memory=True,
            )

        self.to(self.device)
        self.model.train()
        self.fine_tuning_head.train()
        optimizer = optimizer(self.parameters(), **optimizer_params)

        lr_scheduler = None
        if lr_scheduler_params is not None:
            lr_scheduler = get_scheduler(optimizer=optimizer, **lr_scheduler_params)

        logger.info("Starting Fine-Tuning")
        for j in range(epochs):
            batch_count = 0
            batch_loss = 0.0
            batches_processed = 0
            training_loop = tqdm(data_loader)

            for data_dict in training_loop:
                # Forward pass
                output = self._forward(data_dict)

                # Get labels for this batch
                batch_size = data_dict["pert_emb"].shape[0] // self.cell_sentence_len
                labels = torch.tensor(
                    train_labels[batch_count : batch_count + batch_size],
                    device=self.device,
                )
                batch_count += batch_size

                # Compute loss
                loss = loss_function(output, labels)
                loss.backward()
                batch_loss += loss.item()
                batches_processed += 1
                optimizer.step()
                optimizer.zero_grad()

                training_loop.set_postfix({"loss": batch_loss / batches_processed})
                training_loop.set_description(f"Fine-Tuning: epoch {j+1}/{epochs}")

            if lr_scheduler is not None:
                lr_scheduler.step()

            if validation_input_data is not None:
                testing_loop = tqdm(
                    validation_data_loader, desc="Fine-Tuning Validation"
                )
                val_loss = 0.0
                count = 0.0
                validation_batch_count = 0

                for validation_data_dict in testing_loop:
                    # Forward pass
                    output = self._forward(validation_data_dict)

                    # Get validation labels
                    val_batch_size = (
                        validation_data_dict["pert_emb"].shape[0]
                        // self.cell_sentence_len
                    )
                    val_labels = torch.tensor(
                        validation_labels[
                            validation_batch_count : validation_batch_count
                            + val_batch_size
                        ],
                        device=self.device,
                    )
                    validation_batch_count += val_batch_size

                    # Compute validation loss
                    val_loss += loss_function(output, val_labels).item()
                    count += 1.0
                    testing_loop.set_postfix({"val_loss": val_loss / count})

        # save the model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'fine_tuning_head_state_dict': self.fine_tuning_head.state_dict(),
            'embed_dim': self.embed_dim,
            'cell_sentence_len': self.cell_sentence_len,
            'config': self.config,
        }, f"{self.config['model_path']}_finetuned.pt")
        # save just the fine-tuning head separately
        self.fine_tuning_head.save_model(f"{self.config['model_path']}_head.pt")
        logger.info(f"Fine-Tuning Complete. Epochs: {epochs}")

    def get_outputs(
        self,
        dataset: torch.Tensor,
    ) -> np.ndarray:
        """Get the outputs of the fine-tuned model.

        Parameters
        ----------
        dataset : Dataset
            The dataset to get the outputs from.

        Returns
        -------
        np.ndarray
            The outputs of the fine-tuned model.
        """
        self.to(self.device)
        self.model.eval()
        self.fine_tuning_head.eval()

        # Simple collator for state model
        def state_collator(batch):
            """Collate function for state model batches"""
            return {
                "pert_emb": torch.stack([item["pert_emb"] for item in batch]),
                "ctrl_cell_emb": torch.stack([item["ctrl_cell_emb"] for item in batch]),
                "batch": (
                    torch.stack([item["batch"] for item in batch])
                    if "batch" in batch[0]
                    else None
                ),
                "pert_name": [item["pert_name"] for item in batch],
            }

        data_loader = DataLoader(
            dataset,
            batch_size=self.config["batch_size"],
            sampler=SequentialSampler(dataset),
            collate_fn=state_collator,
            drop_last=False,
            pin_memory=True,
        )

        testing_loop = tqdm(data_loader, desc="Inference")
        outputs = []

        with torch.no_grad():
            for data_dict in testing_loop:
                # Forward pass
                output = self._forward(data_dict)
                outputs.append(output.detach().cpu().numpy())

        return np.vstack(outputs)