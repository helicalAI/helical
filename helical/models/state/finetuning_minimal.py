from typing import Literal, Optional
import torch
from torch import optim
from torch.nn.modules import loss
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import get_scheduler
from helical.models.base_models import HelicalBaseFineTuningHead, HelicalBaseFineTuningModel
from .state_transition import stateTransitionModel
from .state_config import stateConfig
import logging
import numpy as np
import anndata as ad

LOGGER = logging.getLogger(__name__)


class EmbeddingDataset:
    """Dataset class for embedding-based fine-tuning (labels handled separately like scGPT)."""

    def __init__(self, embeddings):
        """
        Parameters
        ----------
        embeddings : array-like
            Cell embeddings from the state model
        """
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return {"embedding": self.embeddings[idx]}


def embedding_collator(batch):
    """Collate function for embedding batches (labels handled separately like scGPT)."""
    return {"embeddings": torch.stack([item["embedding"] for item in batch])}


class stateFineTuningModelMinimal(HelicalBaseFineTuningModel, stateTransitionModel):
    """Minimal fine-tuning model for the state model.

    This version inherits from both HelicalBaseFineTuningModel and stateTransitionModel,
    following the scGPT pattern. It loads the model as done in stateTransition and places
    it into train mode with a fine-tuning head on top.

    Example
    ----------
    ```python
    from helical.models.state import stateFineTuningModelMinimal, stateConfig
    import scanpy as sc

    # Load the desired dataset
    adata = sc.read_h5ad("yolksac_human.h5ad")

    # Get the desired label class
    cell_types = list(adata.obs['LVL1'])
    label_set = set(cell_types)

    # Create the fine-tuning model
    config = stateConfig(
        batch_size=8,
        freeze_backbone=True
    )

    model = stateFineTuningModelMinimal(
        configurer=config,
        fine_tuning_head="classification",
        output_size=len(label_set),
    )

    # Process the data for training
    data = model.process_data(adata)

    # Create a dictionary mapping the classes to unique integers for training
    class_id_dict = dict(zip(label_set, [i for i in range(len(label_set))]))
    cell_type_labels = [class_id_dict[ct] for ct in cell_types]

    # Fine-tune
    model.train(train_input_data=data, train_labels=cell_type_labels)
    ```
    """

    def __init__(
        self,
        configurer: stateConfig = None,
        fine_tuning_head: (
            Literal["classification"] | HelicalBaseFineTuningHead
        ) = "classification",
        output_size: Optional[int] = None,
    ):
        # Initialize the base fine-tuning model first
        HelicalBaseFineTuningModel.__init__(self, fine_tuning_head, output_size)
        
        # Initialize the state transition model
        stateTransitionModel.__init__(self, configurer)
        
        # Set the model to training mode
        self.model.train()
        
        # Set up the fine-tuning head dimensions
        self.embed_dim = self.model.output_dim
        self.fine_tuning_head.set_dim_size(self.embed_dim)
        
        # Store config for training
        self.config = configurer.config["finetune"]
        self.freeze_backbone = self.config["freeze_backbone"]
        
        if self.freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
            LOGGER.info("Backbone model frozen - only fine-tuning head will be trained")
        else:
            LOGGER.info("Full model fine-tuning - both backbone and head will be trained")

    def _forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward method for fine-tuning.
        """
        output = self.fine_tuning_head(embeddings)
        return output

    def process_data(self, adata: ad.AnnData) -> EmbeddingDataset:
        """
        Process AnnData through the state model to get embeddings.

        Parameters
        ----------
        adata : AnnData
            Input AnnData with perturbation data

        Returns
        -------
        EmbeddingDataset
            Processed dataset containing embeddings (no labels - user provides them separately)
        """
        LOGGER.info("Processing data for state model fine-tuning.")
        
        # Process data using the parent stateTransitionModel
        adata_processed = stateTransitionModel.process_data(self, adata)
        
        # Get embeddings using the model's forward method
        self.model.eval()
        with torch.no_grad():
            # Create a simple batch for getting embeddings
            batch_size = len(adata_processed)
            
            # Use the expression data as control cell embeddings
            if hasattr(adata_processed.X, "toarray"):
                ctrl_cell_emb = torch.tensor(adata_processed.X.toarray(), dtype=torch.float32)
            else:
                ctrl_cell_emb = torch.tensor(adata_processed.X, dtype=torch.float32)
            
            # Create dummy perturbation embeddings (all zeros for control)
            pert_emb = torch.zeros(batch_size, self.model.pert_dim, dtype=torch.float32)
            if self.model.pert_dim > 0:
                pert_emb[:, 0] = 1.0  # Set first dimension to 1 for control
            
            # Create batch dict
            batch = {
                "pert_emb": pert_emb,
                "ctrl_cell_emb": ctrl_cell_emb,
                "pert_name": ["non-targeting"] * batch_size,
            }
            
            # Add batch info if available
            if hasattr(self, 'batch_indices_all') and self.batch_indices_all is not None:
                batch["batch"] = torch.tensor(self.batch_indices_all, dtype=torch.long)
            
            # Get embeddings using the model's forward method
            embeddings = self.model.forward(batch, padded=False)
        
        LOGGER.info("Successfully processed the data for state model fine-tuning.")
        return EmbeddingDataset(embeddings.cpu().numpy())

    def train(
        self,
        train_input_data: EmbeddingDataset,
        train_labels: np.ndarray,
        validation_input_data: Optional[EmbeddingDataset] = None,
        validation_labels: Optional[np.ndarray] = None,
        optimizer: optim = optim.AdamW,
        optimizer_params: dict = {"lr": 0.0001},
        loss_function: loss = loss.CrossEntropyLoss(),
        epochs: int = 1,
        lr_scheduler_params: Optional[dict] = None,
    ):
        """Fine-tunes the state model with different head modules, similar to scGPT pattern.

        Parameters
        ----------
        train_input_data : EmbeddingDataset
            Processed dataset for fine-tuning (from process_data method)
        train_labels : array-like
            The labels for the training data. Should be integer labels.
        validation_input_data : EmbeddingDataset, default = None
            Processed dataset for per epoch validation. If this is not specified, no validation will be performed.
        validation_labels : array-like, default = None
            The labels for the validation data. Should be integer labels.
        optimizer : torch.optim, default = torch.optim.AdamW
            The optimizer to be used for training.
        optimizer_params : dict
            The optimizer parameters to be used for the optimizer specified.
        loss_function : torch.nn.modules.loss, default = torch.nn.modules.loss.CrossEntropyLoss()
            The loss function to be used.
        epochs : int, optional, default = 1
            The number of epochs to train the model
        lr_scheduler_params : dict, default = None
            The learning rate scheduler parameters for the transformers get_scheduler method.
        """

        # Create data loaders
        data_loader = DataLoader(
            train_input_data,
            batch_size=self.config["batch_size"],
            sampler=SequentialSampler(train_input_data),
            collate_fn=embedding_collator,
            drop_last=False,
            pin_memory=True,
        )

        if validation_input_data is not None:
            validation_data_loader = DataLoader(
                validation_input_data,
                batch_size=self.config["batch_size"],
                sampler=SequentialSampler(validation_input_data),
                collate_fn=embedding_collator,
                drop_last=False,
                pin_memory=True,
            )

        self.to(self.device)
        self.model.train()
        self.fine_tuning_head.train()

        # Set up optimizer based on freeze_backbone setting
        if self.freeze_backbone:
            # Only train the fine-tuning head
            optimizer = optimizer(
                self.fine_tuning_head.parameters(), **optimizer_params
            )
            LOGGER.info("Optimizer set up for fine-tuning head only")
        else:
            # Train the entire model
            optimizer = optimizer(self.parameters(), **optimizer_params)
            LOGGER.info("Optimizer set up for full model fine-tuning")

        lr_scheduler = None
        if lr_scheduler_params is not None:
            lr_scheduler = get_scheduler(optimizer=optimizer, **lr_scheduler_params)

        LOGGER.info("Starting Fine-Tuning")
        for j in range(epochs):
            batch_count = 0
            batch_loss = 0.0
            batches_processed = 0
            training_loop = tqdm(data_loader)

            for data_dict in training_loop:
                # Forward pass
                embeddings = data_dict["embeddings"].to(self.device)

                # Get labels by batch index (like scGPT)
                labels = torch.tensor(
                    train_labels[batch_count : batch_count + self.config["batch_size"]],
                    device=self.device,
                )
                batch_count += self.config["batch_size"]

                output = self._forward(embeddings)

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
                    embeddings = validation_data_dict["embeddings"].to(self.device)

                    # Get validation labels by batch index (like scGPT)
                    val_labels = torch.tensor(
                        validation_labels[
                            validation_batch_count : validation_batch_count
                            + self.config["batch_size"]
                        ],
                        device=self.device,
                    )
                    validation_batch_count += self.config["batch_size"]

                    output = self._forward(embeddings)

                    # Compute validation loss
                    val_loss += loss_function(output, val_labels).item()
                    count += 1.0
                    testing_loop.set_postfix({"val_loss": val_loss / count})

        LOGGER.info(f"Fine-Tuning Complete. Epochs: {epochs}")

    def get_outputs(self, dataset: EmbeddingDataset) -> np.ndarray:
        """Get the outputs of the fine-tuned model.

        Parameters
        ----------
        dataset : EmbeddingDataset
            Processed dataset to get the outputs from.

        Returns
        -------
        np.ndarray
            The outputs of the fine-tuned model.
        """
        self.to(self.device)
        self.model.eval()
        self.fine_tuning_head.eval()

        data_loader = DataLoader(
            dataset,
            batch_size=self.config["batch_size"],
            sampler=SequentialSampler(dataset),
            collate_fn=embedding_collator,
            drop_last=False,
            pin_memory=True,
        )

        testing_loop = tqdm(data_loader, desc="Inference")
        outputs = []

        with torch.no_grad():
            for data_dict in testing_loop:
                # Forward pass
                embeddings = data_dict["embeddings"].to(self.device)
                output = self._forward(embeddings)
                outputs.append(output.detach().cpu().numpy())

        return np.vstack(outputs)
