import time
import logging
from contextlib import contextmanager
from lightning.pytorch.loggers.csv_logs import CSVLogger as BaseCSVLogger
import csv
import os
from lightning.pytorch.callbacks import ModelCheckpoint
from os.path import join
import scanpy as sc
import scipy.sparse as sp
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import numpy as _np

LOGGER = logging.getLogger(__name__)

class RobustCSVLogger(BaseCSVLogger):
    """
    A CSV logger that handles dynamic metrics by allowing new columns to be added during training.
    This fixes the issue where PyTorch Lightning's default CSV logger fails when new metrics
    are added after the CSV file is created.
    """

    def log_metrics(self, metrics, step):
        """Override to handle dynamic metrics gracefully"""
        try:
            super().log_metrics(metrics, step)
        except ValueError as e:
            if "dict contains fields not in fieldnames" in str(e):
                # Recreate the CSV file with the new fieldnames
                self._recreate_csv_with_new_fields(metrics)
                # Try logging again
                super().log_metrics(metrics, step)
            else:
                raise e

    def _recreate_csv_with_new_fields(self, new_metrics):
        """Recreate the CSV file with additional fields to accommodate new metrics"""
        if not hasattr(self.experiment, "metrics_file_path"):
            return

        # Read existing data
        existing_data = []
        csv_file = self.experiment.metrics_file_path

        if os.path.exists(csv_file):
            with open(csv_file, "r", newline="") as f:
                reader = csv.DictReader(f)
                existing_data = list(reader)

        # Get all unique fieldnames from existing data and new metrics
        all_fieldnames = set()
        for row in existing_data:
            all_fieldnames.update(row.keys())
        all_fieldnames.update(new_metrics.keys())

        # Sort fieldnames for consistent ordering
        sorted_fieldnames = sorted(all_fieldnames)

        # Rewrite the CSV file with new fieldnames
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted_fieldnames)
            writer.writeheader()

            # Write existing data (missing fields will be empty)
            for row in existing_data:
                writer.writerow(row)

        # Update the experiment's fieldnames
        self.experiment.metrics_keys = sorted_fieldnames


@contextmanager
def time_it(timer_name: str):
    LOGGER.debug(f"Starting timer {timer_name}")
    start_time = time.perf_counter()
    try:
        yield
    finally:
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        LOGGER.debug(f"Elapsed time {timer_name}: {elapsed_time:.4f} seconds")


def get_loggers(
    output_dir: str,
    name: str,
):

    loggers = []
    csv_logger = RobustCSVLogger(save_dir=output_dir, name=name, version=0)
    loggers.append(csv_logger)

    return loggers


def get_checkpoint_callbacks(
    output_dir: str, name: str, val_freq: int, ckpt_every_n_steps: int
):
    """
    Create checkpoint callbacks based on validation frequency.

    Returns a list of callbacks.
    """
    checkpoint_dir = join(output_dir, name, "checkpoints")
    callbacks = []

    # Save best checkpoint based on validation loss
    best_ckpt = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="step={step}-val_loss={val_loss:.4f}",
        save_last="link",  # Will create last.ckpt symlink to best checkpoint
        monitor="val_loss",
        mode="min",
        save_top_k=1,  # Only keep the best checkpoint
        every_n_train_steps=val_freq,
    )
    callbacks.append(best_ckpt)

    # Also save periodic checkpoints (without affecting the "last" symlink)
    periodic_ckpt = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="{step}",
        save_last=False,  # Don't create/update symlink
        every_n_train_steps=ckpt_every_n_steps,
        save_top_k=-1,  # Keep all periodic checkpoints
    )
    callbacks.append(periodic_ckpt)

    return callbacks


# -----------------------
# Helpers for inference
# -----------------------
def to_dense(mat):
    """Return a dense numpy array for a variety of AnnData .X backends."""
    try:
        if sp.issparse(mat):
            return mat.toarray()
    except Exception:
        pass
    return np.asarray(mat)

def argmax_index_from_any(v, expected_dim: Optional[int]) -> Optional[int]:
    """
    Convert a saved mapping value (one-hot tensor, numpy array, or int) to an index.
    """
    if v is None:
        return None
    try:
        if torch.is_tensor(v):
            if v.ndim == 1:
                return int(torch.argmax(v).item())
            else:
                return None
    except Exception:
        pass
    try:
        if isinstance(v, _np.ndarray):
            if v.ndim == 1:
                return int(v.argmax())
            else:
                return None
    except Exception:
        pass
    if isinstance(v, (int, np.integer)):
        return int(v)
    return None


def prepare_batch(
    ctrl_basal_np: np.ndarray,
    pert_onehots: torch.Tensor,
    batch_indices: Optional[torch.Tensor],
    pert_names: List[str],
    device: torch.device,
) -> Dict[str, torch.Tensor | List[str]]:
    """
    Construct a model batch with variable-length sentence (B=1, S=T, ...).
    IMPORTANT: All tokens in this batch share the same perturbation.
    """
    X_batch = torch.tensor(
        ctrl_basal_np, dtype=torch.float32, device=device
    )  # [T, E_in]
    batch = {
        "ctrl_cell_emb": X_batch,
        "pert_emb": pert_onehots.to(device),  # [T, pert_dim] (same row repeated)
        "pert_name": pert_names,  # list[str], all identical
    }
    if batch_indices is not None:
        batch["batch"] = batch_indices.to(device)  # [T]
    return batch


def pad_adata_with_tsv(
    adata: "sc.AnnData",
    tsv_path: str,
    pert_col: str,
    control_pert: str,
    rng: np.random.RandomState,
    quiet: bool = False,
) -> "sc.AnnData":
    """
    Pad AnnData with additional perturbation cells by copying random control cells
    and updating their perturbation labels according to the TSV specification.

    Args:
        adata: Input AnnData object
        tsv_path: Path to TSV file with 'perturbation' and 'num_cells' columns
        pert_col: Name of perturbation column in adata.obs
        control_pert: Label for control perturbation
        rng: Random number generator for sampling
        quiet: Whether to suppress logging

    Returns:
        AnnData object with padded cells
    """
    # Load TSV file
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(f"TSV file not found: {tsv_path}")

    try:
        tsv_df = pd.read_csv(tsv_path, sep="\t")
    except Exception as e:
        raise ValueError(f"Error reading TSV file {tsv_path}: {e}")

    # Validate TSV format
    required_cols = ["perturbation", "num_cells"]
    missing_cols = [col for col in required_cols if col not in tsv_df.columns]
    if missing_cols:
        raise ValueError(
            f"TSV file missing required columns: {missing_cols}. Found columns: {list(tsv_df.columns)}"
        )

    # Find control cells
    ctl_mask = adata.obs[pert_col].astype(str) == str(control_pert)
    control_indices = np.where(ctl_mask)[0]

    if len(control_indices) == 0:
        raise ValueError(
            f"No control cells found with perturbation '{control_pert}' in column '{pert_col}'"
        )

    if not quiet:
        LOGGER.info(f"Found {len(control_indices)} control cells for padding")

    # Collect cells to add
    new_cells_data = []
    total_to_add = 0

    for _, row in tsv_df.iterrows():
        pert_name = str(row["perturbation"])
        num_cells = int(row["num_cells"])
        total_to_add += num_cells

        if num_cells <= 0:
            continue

        # Sample control cells with replacement
        sampled_indices = rng.choice(control_indices, size=num_cells, replace=True)

        for idx in sampled_indices:
            new_cells_data.append(
                {"original_index": idx, "new_perturbation": pert_name}
            )

    if len(new_cells_data) == 0:
        if not quiet:
            LOGGER.info("No cells to add from TSV file")
        return adata

    if not quiet:
        LOGGER.info(f"Adding {total_to_add} cells from TSV specification")

    # Create new AnnData with padded cells
    original_n_obs = adata.n_obs
    new_n_obs = original_n_obs + len(new_cells_data)

    # Copy X data
    if hasattr(adata.X, "toarray"):  # sparse matrix
        new_X = np.vstack(
            [
                adata.X.toarray(),
                adata.X[
                    np.array([cell["original_index"] for cell in new_cells_data])
                ].toarray(),
            ]
        )
    else:  # dense matrix
        new_X = np.vstack(
            [
                adata.X,
                adata.X[np.array([cell["original_index"] for cell in new_cells_data])],
            ]
        )

    # Copy obs data
    new_obs = adata.obs.copy()
    for i, cell_data in enumerate(new_cells_data):
        orig_idx = cell_data["original_index"]
        new_pert = cell_data["new_perturbation"]

        # Copy the original control cell's metadata
        new_row = adata.obs.iloc[orig_idx].copy()
        # Update perturbation label
        new_row[pert_col] = new_pert

        new_obs.loc[original_n_obs + i] = new_row

    # Copy obsm data
    new_obsm = {}
    for key, matrix in adata.obsm.items():
        padded_matrix = np.vstack(
            [
                matrix,
                matrix[np.array([cell["original_index"] for cell in new_cells_data])],
            ]
        )
        new_obsm[key] = padded_matrix

    # Copy varm, uns, var (unchanged)
    new_varm = adata.varm.copy()
    new_uns = adata.uns.copy()
    new_var = adata.var.copy()

    # Create new AnnData object
    new_adata = sc.AnnData(
        X=new_X, obs=new_obs, var=new_var, obsm=new_obsm, varm=new_varm, uns=new_uns
    )

    if not quiet:
        LOGGER.info(f"Padded AnnData: {original_n_obs} -> {new_n_obs} cells")

    return new_adata