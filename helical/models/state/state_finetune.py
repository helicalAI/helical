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
import scanpy as sc
import anndata as ad

logger = logging.getLogger(__name__)


class stateFineTuningModel(HelicalBaseFineTuningModel):
    """Fine-tuning model for the state model.

    Example
    ----------
    ```python
    from helical.models.state import stateFineTuningModel, stateConfig

    # Load the desired dataset
    adata = sc.read_h5ad("competition_val_template.h5ad")

    # Get the desired label class
    cell_types = list(adata.obs.cell_type)

    # Get unique labels
    label_set = set(cell_types)

    # Create the fine-tuning model with the relevant configs
    config = stateConfig()
    model = stateFineTuningModel(
        configurer=config, 
        fine_tuning_head="classification", 
        output_size=len(label_set),
        freeze_backbone=True
    )

    # Fine-tune
    model.train(train_input_data=adata, train_labels=cell_types)
    ```
    """

    def __init__(
        self,
        configurer: stateConfig = None,
        fine_tuning_head: Literal["classification"] | HelicalBaseFineTuningHead = "classification",
        output_size: Optional[int] = None,
        freeze_backbone: bool = True,
    ):

        if configurer is None:
            self.config = stateConfig().config["finetune"]
        else:
            self.config = configurer.config["finetune"]

        HelicalBaseFineTuningModel.__init__(self, fine_tuning_head, output_size)
        
        # Load the pre-trained state model
        self.model = StateTransitionPerturbationModel.load_from_checkpoint(
            self.config["checkpoint"]
        )

        # Get the actual output dimension from the loaded model
        self.embed_dim = self.model.output_dim
        self.cell_sentence_len = self.model.cell_sentence_len
        self.device = next(self.model.parameters()).device
        self.fine_tuning_head.set_dim_size(self.embed_dim)
        
        # Freeze backbone if requested
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
            logger.info("Backbone model frozen - only fine-tuning head will be trained")
        else:
            logger.info("Full model fine-tuning - both backbone and head will be trained")
        
        # We'll process data directly using the model, no need for inference wrapper

    def _forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward method for fine-tuning.
        """
        output = self.fine_tuning_head(embeddings)
        return output

    def process_data(self, adata: ad.AnnData, label_column: str = "cell_type"):
        """
        Process AnnData through the state model to get embeddings.
        
        Parameters
        ----------
        adata : AnnData
            Input AnnData with perturbation data
        label_column : str
            Column name in adata.obs containing the labels for classification
            
        Returns
        -------
        tuple
            (embeddings, labels) where embeddings are numpy arrays
        """
        # Extract labels first
        if label_column not in adata.obs.columns:
            raise ValueError(f"Label column '{label_column}' not found in adata.obs")
        
        labels = adata.obs[label_column].values
        
        # Convert AnnData to the format expected by the model
        # We need to create a batch dict similar to what the model expects
        batch_size = len(adata)
        
        # Get perturbation info
        if "target_gene" in adata.obs.columns:
            pert_names = adata.obs["target_gene"].values
        else:
            # Default to non-targeting if no perturbation column
            pert_names = ["non-targeting"] * batch_size
        
        # Create dummy perturbation embeddings (zeros) - the model will handle this
        pert_emb = torch.zeros(batch_size, self.model.pert_dim, dtype=torch.float32)
        
        # Use the expression data as control cell embeddings
        if hasattr(adata.X, 'toarray'):
            ctrl_cell_emb = torch.tensor(adata.X.toarray(), dtype=torch.float32)
        else:
            ctrl_cell_emb = torch.tensor(adata.X, dtype=torch.float32)
        
        # Create batch dict
        batch = {
            "pert_emb": pert_emb,
            "ctrl_cell_emb": ctrl_cell_emb,
            "pert_name": pert_names,
        }
        
        # Add batch info if available
        if "batch_var" in adata.obs.columns:
            batch["batch"] = torch.zeros(batch_size, dtype=torch.long)  # Dummy batch indices
        
        # Get embeddings using the model's forward method
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model.forward(batch, padded=False)
        
        return embeddings.cpu().numpy(), labels

    def create_label_mapping(self, labels):
        """
        Create integer mapping for string labels, similar to scGPT example.
        
        Parameters
        ----------
        labels : array-like
            String labels to convert to integers
            
        Returns
        -------
        tuple
            (integer_labels, label_to_int_dict)
        """
        unique_labels = list(set(labels))
        label_to_int_dict = dict(zip(unique_labels, range(len(unique_labels))))
        integer_labels = [label_to_int_dict[label] for label in labels]
        
        logger.info(f"Created label mapping: {label_to_int_dict}")
        return np.array(integer_labels), label_to_int_dict

    def train(
        self,
        train_input_data,
        train_labels,
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
        train_input_data : AnnData
            AnnData object for fine-tuning
        train_labels : array-like
            The labels for the training data. Can be string labels (will be converted to integers) or integer labels.
        validation_input_data : AnnData, default = None
            AnnData object for per epoch validation. If this is not specified, no validation will be performed.
        validation_labels : array-like, default = None
            The labels for the validation data. Can be string labels (will be converted to integers) or integer labels.
        optimizer : torch.optim, default = torch.optim.AdamW
            The optimizer to be used for training.
        optimizer_params : dict
            The optimizer parameters to be used for the optimizer specified.
        loss_function : torch.nn.modules.loss, default = torch.nn.modules.loss.CrossEntropyLoss()
            The loss function to be used.
        epochs : int, optional, default = 10
            The number of epochs to train the model
        lr_scheduler_params : dict, default = None
            The learning rate scheduler parameters for the transformers get_scheduler method.
        """

        # Convert string labels to integers if needed
        if isinstance(train_labels[0], str):
            train_labels, self.label_mapping = self.create_label_mapping(train_labels)
            logger.info(f"Converted string labels to integers: {len(self.label_mapping)} classes")
        
        if validation_labels is not None and isinstance(validation_labels[0], str):
            # Use the same mapping for validation labels
            if hasattr(self, 'label_mapping'):
                validation_labels = np.array([self.label_mapping[label] for label in validation_labels])
            else:
                validation_labels, _ = self.create_label_mapping(validation_labels)

        # Process training data
        train_embeddings, _ = self.process_data(train_input_data)
        
        # Create simple dataset from embeddings
        class EmbeddingDataset:
            def __init__(self, embeddings, labels):
                self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
                self.labels = labels
            
            def __len__(self):
                return len(self.embeddings)
            
            def __getitem__(self, idx):
                return {
                    "embedding": self.embeddings[idx],
                    "label": self.labels[idx]
                }
        
        train_dataset = EmbeddingDataset(train_embeddings, train_labels)
        
        # Handle validation data if provided
        val_dataset = None
        if validation_input_data is not None:
            val_embeddings, _ = self.process_data(validation_input_data)
            val_dataset = EmbeddingDataset(val_embeddings, validation_labels)

        # Simple collator for embedding dataset
        def embedding_collator(batch):
            """Collate function for embedding batches"""
            return {
                "embeddings": torch.stack([item["embedding"] for item in batch]),
                "labels": torch.tensor([item["label"] for item in batch])
            }

        data_loader = DataLoader(
            train_dataset,
            batch_size=self.config["batch_size"],
            sampler=SequentialSampler(train_dataset),
            collate_fn=embedding_collator,
            drop_last=False,
            pin_memory=True,
        )

        if val_dataset is not None:
            validation_data_loader = DataLoader(
                val_dataset,
                batch_size=self.config["batch_size"],
                sampler=SequentialSampler(val_dataset),
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
            optimizer = optimizer(self.fine_tuning_head.parameters(), **optimizer_params)
            logger.info("Optimizer set up for fine-tuning head only")
        else:
            # Train the entire model
            optimizer = optimizer(self.parameters(), **optimizer_params)
            logger.info("Optimizer set up for full model fine-tuning")

        lr_scheduler = None
        if lr_scheduler_params is not None:
            lr_scheduler = get_scheduler(optimizer=optimizer, **lr_scheduler_params)

        logger.info("Starting Fine-Tuning")
        for j in range(epochs):
            batch_loss = 0.0
            batches_processed = 0
            training_loop = tqdm(data_loader)

            for data_dict in training_loop:
                # Forward pass
                embeddings = data_dict["embeddings"].to(self.device)
                labels = data_dict["labels"].to(self.device)
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

            if val_dataset is not None:
                testing_loop = tqdm(
                    validation_data_loader, desc="Fine-Tuning Validation"
                )
                val_loss = 0.0
                count = 0.0

                for validation_data_dict in testing_loop:
                    # Forward pass
                    embeddings = validation_data_dict["embeddings"].to(self.device)
                    val_labels = validation_data_dict["labels"].to(self.device)
                    output = self._forward(embeddings)

                    # Compute validation loss
                    val_loss += loss_function(output, val_labels).item()
                    count += 1.0
                    testing_loop.set_postfix({"val_loss": val_loss / count})
        logger.info(f"Fine-Tuning Complete. Epochs: {epochs}")

    def get_outputs(
        self,
        dataset,
    ) -> np.ndarray:
        """Get the outputs of the fine-tuned model.

        Parameters
        ----------
        dataset : AnnData
            AnnData object to get the outputs from.

        Returns
        -------
        np.ndarray
            The outputs of the fine-tuned model.
        """
        self.to(self.device)
        self.model.eval()
        self.fine_tuning_head.eval()

        # Process data to get embeddings
        embeddings, _ = self.process_data(dataset)
        
        # Create simple dataset from embeddings
        class EmbeddingDataset:
            def __init__(self, embeddings):
                self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
            
            def __len__(self):
                return len(self.embeddings)
            
            def __getitem__(self, idx):
                return {"embedding": self.embeddings[idx]}
        
        dataset = EmbeddingDataset(embeddings)

        # Simple collator for embedding dataset
        def embedding_collator(batch):
            """Collate function for embedding batches"""
            return {
                "embeddings": torch.stack([item["embedding"] for item in batch])
            }

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