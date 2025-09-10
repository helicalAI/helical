from typing import Literal, Optional
from helical.models.state.state_config import stateConfig
import torch
from torch import optim
from torch.nn.modules import loss
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import get_scheduler
from helical.models.base_models import HelicalBaseFineTuningHead
from .model_dir.perturb_utils.state_transition_model import (
    StateTransitionPerturbationModel,
)
from helical.models.base_models import HelicalBaseFineTuningModel
import logging
import numpy as np
import os
import anndata as ad
import scanpy as sc
import pickle
import yaml

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


class stateFineTuningModel(HelicalBaseFineTuningModel):
    """Enhanced fine-tuning model for the state model that can generate var_dims from adata.

    This version automatically generates var_dims.pkl and config.yaml from the input data,
    following the same logic as state_train.py, so you don't need to specify file locations.

    Example
    ----------
    ```python
    from helical.models.state import stateFineTuningModelV2, stateConfig
    import scanpy as sc

    # Load the desired dataset
    adata = sc.read_h5ad("competition_support_set/competition_val_template.h5ad")

    # Get the desired label class
    cell_types = list(adata.obs.cell_type)
    label_set = set(cell_types)

    # Create the fine-tuning model (no need to specify var_dims location)
    config = stateConfig(
        batch_size=8,
        model_dir="competition/first_run",
        freeze_backbone=False
    )

    model = stateFineTuningModelV2(
        configurer=config,
        fine_tuning_head="classification",
        output_size=len(label_set),
        freeze_backbone=False
    )

    # Process the data for training
    data = model.process_data(adata)

    # Create a dictionary mapping the classes to unique integers for training
    class_id_dict = dict(zip(label_set, [i for i in range(len(label_set))]))

    for i in range(len(cell_types)):
        cell_types[i] = class_id_dict[cell_types[i]]

    # Fine-tune
    model.train(train_input_data=data, train_labels=cell_types)
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

        if configurer is None:
            self.config = stateConfig().config["finetune"]
        else:
            self.config = configurer.config["finetune"]

        HelicalBaseFineTuningModel.__init__(self, fine_tuning_head, output_size)

        if not os.path.exists(self.config["model_config"]):
            raise FileNotFoundError(
                f"config.yaml not found at {self.config['model_config']}. Please ensure it exists."
            )

        LOGGER.info(f"Loading existing config.yaml from: {self.config['model_config']}")
        with open(self.config["model_config"], "r") as f:
            self._model_config = yaml.safe_load(f)

        # Initialize attributes that might be used later
        self._var_dims = None
        self._gene_dim = None
        self.has_var_dims = True
        self.use_perturbation_embeddings = self.config["use_perturbation_embeddings"]
        self.default_perturbation_type = self.config["control_pert"]

        # Load the pre-trained state model or initialize fresh
        checkpoint_path = os.path.join(
            self.config["model_dir"], self.config["checkpoint_name"]
        )
        self.model_dir = self.config["model_dir"]

        if os.path.exists(checkpoint_path):
            LOGGER.info(f"Loading pre-trained model from: {checkpoint_path}")
            self.model = StateTransitionPerturbationModel.load_from_checkpoint(
                checkpoint_path
            )
        else:
            LOGGER.info(
                f"No checkpoint found at {checkpoint_path}, initializing fresh model from config"
            )
            self._initialize_fresh_model()

        # Check if model was successfully initialized
        if self.has_var_dims:
            # Get the actual output dimension from the loaded model
            self.embed_dim = self.model.output_dim
            self.cell_sentence_len = self.model.cell_sentence_len
            self.device = next(self.model.parameters()).device
            self.fine_tuning_head.set_dim_size(self.embed_dim)
        else:
            LOGGER.info(
                "Model will be initialized when data is inputted via process_data()"
            )
            self.embed_dim = None
            self.cell_sentence_len = None
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Check for .pt weight files and load them if available
        self.backbone_weights_path = (
            os.path.join(self.model_dir, "model_weights.pt")
            if os.path.exists(os.path.join(self.model_dir, "model_weights.pt"))
            else None
        )
        self.head_weights_path = (
            os.path.join(self.model_dir, "head_weights.pt")
            if os.path.exists(os.path.join(self.model_dir, "head_weights.pt"))
            else None
        )

        if self.has_var_dims:
            self.load_pt_weights()

        self.freeze_backbone = self.config["freeze_backbone"]
        if self.has_var_dims:
            if self.freeze_backbone:
                for param in self.model.parameters():
                    param.requires_grad = False
                LOGGER.info(
                    "Backbone model frozen - only fine-tuning head will be trained"
                )
            else:
                LOGGER.info(
                    "Full model fine-tuning - both backbone and head will be trained"
                )
        else:
            LOGGER.info(
                f"Freeze backbone setting: {self.freeze_backbone} (will be applied when model is initialized)"
            )

    def _create_var_dims_from_adata(self, adata):
        """Create var_dims.pkl from adata following the same logic as state_train.py."""
        LOGGER.info("Creating var_dims from adata")

        os.makedirs(self.model_dir, exist_ok=True)

        # Analyze the data to create var_dims (following state_train.py logic)
        n_genes = adata.n_vars
        n_cells = adata.n_obs

        LOGGER.info(f"Data shape: {adata.shape}")
        LOGGER.info(f"Available columns: {list(adata.obs.columns)}")

        # Get perturbation info
        if self.config["pert_col"] in adata.obs.columns:
            unique_perts = adata.obs[self.config["pert_col"]].unique()
            n_perts = len(unique_perts)
            pert_names = list(unique_perts)
            LOGGER.info(f"Found {n_perts} unique perturbations: {pert_names[:5]}...")
        else:
            unique_perts = [self.config["control_pert"]]
            n_perts = 1
            pert_names = [self.config["control_pert"]]
            LOGGER.info(
                f"No target_gene column found, using default '{self.config['control_pert']}'"
            )

        # Get batch info
        if self.config["batch_col"] in adata.obs.columns:
            unique_batches = adata.obs[self.config["batch_col"]].unique()
            n_batches = len(unique_batches)
            batch_names = list(unique_batches)
            LOGGER.info(f"Found {n_batches} unique batches: {batch_names}")
        else:
            unique_batches = [self.config["batch_col"]]
            n_batches = 1
            batch_names = [self.config["batch_col"]]
            LOGGER.info(
                f"No {self.config['batch_col']} column found, using default '{self.config['batch_col']}'"
            )

        # Get cell type info
        if "cell_type" in adata.obs.columns:
            unique_cell_types = adata.obs["cell_type"].unique()
            cell_type_names = list(unique_cell_types)
            LOGGER.info(
                f"Found {len(cell_type_names)} unique cell types: {cell_type_names}"
            )
        else:
            cell_type_names = ["unknown"]
            LOGGER.info("No cell_type column found, using default 'unknown'")

        # Create var_dims dictionary (following state_train.py structure)
        var_dims = {
            "input_dim": n_genes,  # Number of input genes
            "output_dim": n_genes,  # Number of output genes (same as input for most cases)
            "hvg_dim": n_genes,  # Number of highly variable genes (using all genes for simplicity)
            "gene_dim": n_genes,  # Total number of genes
            "pert_dim": n_perts,  # Number of different perturbations
            "batch_dim": n_batches,  # Number of different batches
            "gene_names": list(adata.var_names),  # List of gene names
            "pert_names": pert_names,  # List of perturbation names
            "batch_names": batch_names,  # List of batch names
            "cell_type_names": cell_type_names,  # List of cell type names
        }

        LOGGER.info(f"Created var_dims:")
        for key, value in var_dims.items():
            if isinstance(value, (list, np.ndarray)):
                LOGGER.info(f"  {key}: {len(value)} items")
            else:
                LOGGER.info(f"  {key}: {value}")

        # Save var_dims
        var_dims_path = os.path.join(self.model_dir, "var_dims.pkl")
        with open(var_dims_path, "wb") as f:
            pickle.dump(var_dims, f)

        LOGGER.info(f"Saved var_dims to: {var_dims_path}")
        return var_dims

    def _create_gene_dim_from_var_dims(self, var_dims, output_space="gene"):
        """Calculate gene_dim following the same logic as state_train.py."""
        LOGGER.info(f"Creating gene_dim (output_space: {output_space})")

        if output_space == "gene":
            gene_dim = var_dims.get("hvg_dim", 2000)
            LOGGER.info(f"Using hvg_dim for gene_dim: {gene_dim}")
        else:
            gene_dim = var_dims.get("gene_dim", 2000)
            LOGGER.info(f"Using gene_dim: {gene_dim}")

        return gene_dim

    def _initialize_fresh_model(self):
        """Initialize a fresh model using config.yaml and var_dims.pkl (same as training model)."""
        var_dims_path = os.path.join(self.model_dir, "var_dims.pkl")

        if not os.path.exists(var_dims_path):
            LOGGER.info(
                "var_dims.pkl not found, will be created when process_data is called"
            )
            self.has_var_dims = False
            return

        # Load var_dims.pkl
        with open(var_dims_path, "rb") as f:
            var_dims = pickle.load(f)

        # Calculate gene_dim (same logic as training)
        output_space = self._model_config["data"]["kwargs"].get("output_space", "gene")
        self._gene_dim = self._create_gene_dim_from_var_dims(var_dims, output_space)

        # Initialize model with same parameters as training
        self._initialize_model_from_config(var_dims, self._gene_dim)

        # Store var_dims for future use
        self._var_dims = var_dims

        LOGGER.info(
            "Successfully initialized fresh model from existing config.yaml and var_dims.pkl"
        )

    def _initialize_model_from_config(self, var_dims, gene_dim):
        """Initialize model following the exact same logic as state_train.py."""
        LOGGER.info("Initializing model")

        training_config = self._model_config["training"]
        data_config = self._model_config["data"]["kwargs"]
        model_kwargs = self._model_config["model"]["kwargs"]

        module_config = {**model_kwargs, **training_config}
        module_config["embed_key"] = data_config["embed_key"]
        module_config["output_space"] = data_config["output_space"]
        module_config["gene_names"] = var_dims["gene_names"]
        module_config["batch_size"] = training_config["batch_size"]
        module_config["control_pert"] = data_config.get("control_pert", "non-targeting")

        # Initialize model with same parameters as state_train.py
        self.model = StateTransitionPerturbationModel(
            input_dim=var_dims["input_dim"],
            gene_dim=gene_dim,
            hvg_dim=var_dims["hvg_dim"],
            output_dim=var_dims["output_dim"],
            pert_dim=var_dims["pert_dim"],
            batch_dim=var_dims["batch_dim"],
            **module_config,
        )

        LOGGER.info("Successfully initialized model")
        LOGGER.info(f"Model input_dim: {self.model.input_dim}")
        LOGGER.info(f"Model output_dim: {self.model.output_dim}")
        LOGGER.info(f"Model pert_dim: {self.model.pert_dim}")
        LOGGER.info(f"Model gene_dim: {gene_dim}")

    def _forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward method for fine-tuning.
        """
        output = self.fine_tuning_head(embeddings)
        return output

    def load_pt_weights(self):
        """
        Load the model and head weights from .pt files.
        """
        if self.backbone_weights_path is not None:
            self.model.load_state_dict(
                torch.load(self.backbone_weights_path, weights_only=True)
            )
            LOGGER.info(f"Loaded model weights from {self.backbone_weights_path}")
        if self.head_weights_path is not None:
            self.fine_tuning_head.load_state_dict(
                torch.load(self.head_weights_path, weights_only=True)
            )
            LOGGER.info(f"Loaded head weights from {self.head_weights_path}")
        return

    def process_data(self, adata: ad.AnnData) -> EmbeddingDataset:
        """
        Process AnnData through the state model to get embeddings, similar to scGPT's process_data.
        If var_dims.pkl and config.yaml don't exist, they will be created from the data.

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

        # If we don't have var_dims yet, create them from the data
        if self.has_var_dims is False:
            LOGGER.info("Creating var_dims from adata")

            # Create var_dims from adata
            var_dims = self._create_var_dims_from_adata(adata)

            # Calculate gene_dim
            output_space = self._model_config["data"]["kwargs"].get(
                "output_space", "gene"
            )
            gene_dim = self._create_gene_dim_from_var_dims(var_dims, output_space)

            # Initialize the model
            self._initialize_model_from_config(var_dims, gene_dim)

            # Update embed_dim and device after model initialization
            self.embed_dim = self.model.output_dim
            self.cell_sentence_len = self.model.cell_sentence_len
            self.device = next(self.model.parameters()).device
            self.fine_tuning_head.set_dim_size(self.embed_dim)

            # Apply freeze_backbone setting now that model is initialized
            if self.freeze_backbone:
                for param in self.model.parameters():
                    param.requires_grad = False
                LOGGER.info(
                    "Backbone model frozen - only fine-tuning head will be trained"
                )
            else:
                LOGGER.info(
                    "Full model fine-tuning - both backbone and head will be trained"
                )

            # Store for future use
            self._var_dims = var_dims
            self._gene_dim = gene_dim

        # Convert AnnData to the format expected by the model
        # We need to create a batch dict similar to what the model expects
        batch_size = len(adata)

        # Get perturbation info
        if "target_gene" in adata.obs.columns:
            pert_names = adata.obs["target_gene"].values
        else:
            # Default to non-targeting if no perturbation column
            pert_names = ["non-targeting"] * batch_size

        # Create proper perturbation embeddings
        pert_emb = self._create_perturbation_embeddings(pert_names, batch_size)

        # Use the expression data as control cell embeddings
        if hasattr(adata.X, "toarray"):
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
            batch["batch"] = torch.zeros(
                batch_size, dtype=torch.long
            )  # Dummy batch indices

        # Get embeddings using the model's forward method
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model.forward(batch, padded=False)

        LOGGER.info("Successfully processed the data for state model fine-tuning.")
        return EmbeddingDataset(embeddings.cpu().numpy())

    def _create_perturbation_embeddings(self, pert_names, batch_size):
        """
        Create perturbation embeddings for each cell based on perturbation names.

        Parameters
        ----------
        pert_names : list
            List of perturbation names for each cell
        batch_size : int
            Number of cells

        Returns
        -------
        torch.Tensor
            Perturbation embeddings of shape (batch_size, pert_dim)
        """
        if not self.use_perturbation_embeddings:
            # Use zeros if perturbation embeddings are disabled
            LOGGER.info("Perturbation embeddings disabled, using zeros")
            return torch.zeros(batch_size, self.model.pert_dim, dtype=torch.float32)

        # Try to load the actual perturbation one-hot mapping from the model directory
        model_dir = self.model_dir
        pert_onehot_map_path = os.path.join(model_dir, "pert_onehot_map.pt")

        if os.path.exists(pert_onehot_map_path):
            try:
                # Load the actual perturbation mapping
                pert_onehot_map = torch.load(pert_onehot_map_path, weights_only=False)
                LOGGER.info(
                    f"Loaded perturbation mapping with {len(pert_onehot_map)} perturbations"
                )

                # Create embeddings for each cell
                pert_embeddings = []
                for pert_name in pert_names:
                    if pert_name in pert_onehot_map:
                        pert_embeddings.append(pert_onehot_map[pert_name].float())
                    else:
                        # Use default perturbation vector for unknown perturbations
                        if self.default_perturbation_type in pert_onehot_map:
                            default_vec = pert_onehot_map[
                                self.default_perturbation_type
                            ].float()
                        else:
                            # Fallback to control vector
                            default_vec = torch.zeros(
                                self.model.pert_dim, dtype=torch.float32
                            )
                            if self.model.pert_dim > 0:
                                default_vec[0] = 1.0
                        pert_embeddings.append(default_vec)
                        LOGGER.warning(
                            f"Unknown perturbation '{pert_name}', using {self.default_perturbation_type} vector"
                        )

                return torch.stack(pert_embeddings)

            except Exception as e:
                LOGGER.warning(f"Failed to load perturbation mapping: {e}")

        # Fallback: create default control embeddings for all cells
        LOGGER.info(
            f"Using default {self.default_perturbation_type} perturbation embeddings for all cells"
        )
        default_pert_vec = torch.zeros(self.model.pert_dim, dtype=torch.float32)
        if self.model.pert_dim > 0:
            default_pert_vec[0] = 1.0  # Control perturbation

        return default_pert_vec.unsqueeze(0).repeat(batch_size, 1)

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

        if self.freeze_backbone is True:
            # we only need to save the head weights
            torch.save(
                self.fine_tuning_head.state_dict(),
                os.path.join(self.model_dir, "head_weights.pt"),
            )
        else:
            # we need to save the model and head weights
            torch.save(
                self.model.state_dict(),
                os.path.join(self.model_dir, "model_weights.pt"),
            )
            torch.save(
                self.fine_tuning_head.state_dict(),
                os.path.join(self.model_dir, "head_weights.pt"),
            )
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
