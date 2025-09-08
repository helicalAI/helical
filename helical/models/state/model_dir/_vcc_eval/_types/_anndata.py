import logging
from dataclasses import dataclass, field
from typing import Iterator, Literal

import anndata as ad
import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy.sparse import issparse
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PerturbationAnndataPair:
    """Pair of AnnData objects with perturbation information."""

    real: ad.AnnData
    pred: ad.AnnData
    pert_col: str
    control_pert: str
    embed_key: str | None = None
    perts: NDArray[np.str_] = field(init=False)
    genes: NDArray[np.str_] = field(init=False)

    # Masks of indices for each perturbation
    pert_mask_real: dict[str, np.ndarray] = field(init=False)
    pert_mask_pred: dict[str, np.ndarray] = field(init=False)

    # Bulk anndata by embedding key and perturbation
    bulk_real: dict[str, tuple[NDArray[np.str_], NDArray[np.float64]]] | None = field(
        init=False
    )
    bulk_pred: dict[str, tuple[NDArray[np.str_], NDArray[np.float64]]] | None = field(
        init=False
    )

    def __post_init__(self) -> None:
        if self.real.shape[1] != self.pred.shape[1]:
            raise ValueError(
                f"Shape mismatch: real {self.real.shape[1]} != pred {self.pred.shape[1]}"
                " Expected to be the same number of genes"
            )

        if self.embed_key:
            if self.embed_key not in self.real.obsm:
                raise ValueError(
                    f"Embed key {self.embed_key} not found in real AnnData"
                )
            if self.embed_key not in self.pred.obsm:
                raise ValueError(
                    f"Embed key {self.embed_key} not found in pred AnnData"
                )

        var_names_real = np.array(self.real.var.index.values)
        var_names_pred = np.array(self.pred.var.index.values)
        if not np.array_equal(var_names_real, var_names_pred):
            raise ValueError(
                f"Gene names order mismatch: real {var_names_real} != pred {var_names_pred}"
            )
        object.__setattr__(self, "genes", var_names_real)

        if self.pert_col not in self.real.obs.columns:
            raise ValueError(
                f"Perturbation column ({self.pert_col}) not found in real AnnData: {self.real.obs.columns}"
            )
        if self.pert_col not in self.pred.obs.columns:
            raise ValueError(
                f"Perturbation column ({self.pert_col}) not found in pred AnnData: {self.pred.obs.columns}"
            )

        perts_real = np.unique(self.real.obs[self.pert_col].to_numpy(str))
        perts_pred = np.unique(self.pred.obs[self.pert_col].to_numpy(str))
        if not np.array_equal(perts_real, perts_pred):
            raise ValueError(
                f"Perturbation mismatch: real {perts_real} != pred {perts_pred}"
            )

        if self.control_pert not in perts_real:
            raise ValueError(
                f"Control perturbation ({self.control_pert}) not found in real AnnData: {perts_real}"
            )
        if self.control_pert not in perts_pred:
            raise ValueError(
                f"Control perturbation ({self.control_pert}) not found in pred AnnData: {perts_pred}"
            )

        perts = np.union1d(perts_real, perts_pred)
        perts = np.array([p for p in perts if p != self.control_pert])

        pert_mask_real = self.pert_mask(
            self.real.obs[self.pert_col].to_numpy(str),
        )
        pert_mask_pred = self.pert_mask(
            self.pred.obs[self.pert_col].to_numpy(str),
        )

        object.__setattr__(self, "perts", perts)
        object.__setattr__(self, "pert_mask_real", pert_mask_real)
        object.__setattr__(self, "pert_mask_pred", pert_mask_pred)

        # Initialize bulk anndata
        object.__setattr__(self, "bulk_real", {})
        object.__setattr__(self, "bulk_pred", {})

    @staticmethod
    def _bulk_anndata(
        adata: ad.AnnData,
        groupby_key: str,
        embed_key: str | None = None,
    ) -> tuple[NDArray[np.str_], NDArray[np.float64]]:
        """Get bulk anndata for a groupby key."""

        matrix = adata.X if not embed_key else adata.obsm[embed_key]
        if issparse(matrix):
            # Convert sparse matrix to dense array
            logger.info("Converting sparse matrix to dense array for bulk calculation")
            matrix = matrix.toarray()  # type: ignore

        # Create a polars dataframe with the groupby key
        frame = pl.DataFrame(
            matrix,
        ).with_columns(
            groupby_key=adata.obs[groupby_key].to_numpy(str),  # type: ignore
        )

        # Pseudobulk (mean) the dataframe by the groupby key
        bulked = frame.group_by("groupby_key").mean().sort("groupby_key")

        # identify the key column
        keys = bulked["groupby_key"].to_numpy()

        # identify the pseudobulks
        values = bulked.drop("groupby_key").to_numpy()

        return (keys, values)

    @staticmethod
    def pert_mask(perts: NDArray[np.str_]) -> dict[str, NDArray[np.int_]]:
        unique_perts, inverse = np.unique(perts, return_inverse=True)
        return {pert: np.where(inverse == i)[0] for i, pert in enumerate(unique_perts)}

    def get_perts(self, include_control: bool = False) -> NDArray[np.str_]:
        """Get all perturbations."""
        if include_control:
            return self.perts
        return self.perts[self.perts != self.control_pert]

    def _initialize_bulk_arrays(self, embed_key: str | None = None):
        """Initialize bulk arrays if necessary (memoized)"""
        if embed_key is None:
            embed_key = "_default"
        rebuilt = False
        assert self.bulk_real is not None
        if embed_key not in self.bulk_real:
            logger.info(
                "Building pseudobulk embeddings for real anndata on: {}".format(
                    embed_key if embed_key != "_default" else ".X"
                )
            )
            self.bulk_real[embed_key] = self._bulk_anndata(
                self.real,
                self.pert_col,
                embed_key=embed_key if embed_key != "_default" else None,
            )
            rebuilt = True

        assert self.bulk_pred is not None
        if embed_key not in self.bulk_pred:
            logger.info(
                "Building pseudobulk embeddings for predicted anndata on: {}".format(
                    embed_key if embed_key != "_default" else ".X"
                )
            )
            self.bulk_pred[embed_key] = self._bulk_anndata(
                self.pred,
                self.pert_col,
                embed_key=embed_key if embed_key != "_default" else None,
            )
            rebuilt = True

        if rebuilt:
            real_id = self.bulk_real[embed_key][0]
            pred_id = self.bulk_pred[embed_key][0]
            if not np.array_equal(real_id, pred_id):
                raise ValueError(
                    f"Real and predicted embeddings are missing perturbations for {embed_key}"
                )
            if self.control_pert not in real_id:
                raise ValueError(
                    f"Control perturbation {self.control_pert} is missing in embeddings for {embed_key}"
                )

    def build_bulk_array(self, pert: str, embed_key: str | None = None) -> "BulkArrays":
        """Build bulk array for a perturbation."""
        if not embed_key:
            embed_key = "_default"

        assert self.bulk_real is not None, "Bulk real data is missing"
        assert self.bulk_pred is not None, "Bulk pred data is missing"

        # Get the perturbation indices
        pert_pos = np.flatnonzero(self.bulk_real[embed_key][0] == pert)[0]
        ctrl_pos = np.flatnonzero(self.bulk_real[embed_key][0] == self.control_pert)[0]
        return BulkArrays(
            key=pert,
            pert_real=self.bulk_real[embed_key][1][pert_pos],
            pert_pred=self.bulk_pred[embed_key][1][pert_pos],
            ctrl_real=self.bulk_real[embed_key][1][ctrl_pos],
            ctrl_pred=self.bulk_pred[embed_key][1][ctrl_pos],
        )

    def build_cell_array(self, pert: str, embed_key: str | None = None) -> "CellArrays":
        """Build cell array for a perturbation."""

        if not embed_key:
            matrix_real = self.real.X
            matrix_pred = self.pred.X
        else:
            matrix_real = self.real.obsm[embed_key]
            matrix_pred = self.pred.obsm[embed_key]

        pert_pos_real = self.pert_mask_real[pert]
        pert_pos_pred = self.pert_mask_pred[pert]
        ctrl_pos_real = self.pert_mask_real[self.control_pert]
        ctrl_pos_pred = self.pert_mask_pred[self.control_pert]

        return CellArrays(
            key=pert,
            pert_real=matrix_real[pert_pos_real],  # type: ignore
            pert_pred=matrix_pred[pert_pos_pred],  # type: ignore
            ctrl_real=matrix_real[ctrl_pos_real],  # type: ignore
            ctrl_pred=matrix_pred[ctrl_pos_pred],  # type: ignore
        )

    def iter_bulk_arrays(self, embed_key: str | None = None) -> Iterator["BulkArrays"]:
        """Iterate over bulk arrays for all perturbations."""
        self._initialize_bulk_arrays(embed_key)
        for pert in tqdm(self.perts, desc="Iterating over perturbations..."):
            yield self.build_bulk_array(pert, embed_key=embed_key)

    def iter_cell_arrays(self, embed_key: str | None = None) -> Iterator["CellArrays"]:
        """Iterate over subarrays of cells for all perturbations"""
        for pert in tqdm(self.perts, desc="Iterating over perturbations..."):
            yield self.build_cell_array(pert, embed_key=embed_key)

    def ctrl_matrix(
        self, which: Literal["real", "pred"], embed_key: str | None = None
    ) -> np.ndarray:
        """Build a CellArrays object for the control perturbation."""
        if not embed_key:
            matrix = self.real.X if which == "real" else self.pred.X
        else:
            matrix = (
                self.real.obsm[embed_key]
                if which == "real"
                else self.pred.obsm[embed_key]
            )
        mask_lookup = self.pert_mask_real if which == "real" else self.pert_mask_pred
        return matrix[mask_lookup[self.control_pert]]  # type: ignore


@dataclass(frozen=True)
class BulkArrays:
    """Arrays of bulk results for a perturbation."""

    key: str
    pert_real: np.ndarray
    pert_pred: np.ndarray
    ctrl_real: np.ndarray
    ctrl_pred: np.ndarray

    def perturbation_effect(
        self,
        which: Literal["real", "pred"],
        abs: bool = False,
    ) -> np.ndarray:
        match which:
            case "real":
                effect = self.pert_real - self.ctrl_real
            case "pred":
                effect = self.pert_pred - self.ctrl_pred
            case _:
                raise ValueError(f"Invalid value for `which`: {which}")
        if abs:
            effect = np.abs(effect)
        return effect


@dataclass(frozen=True)
class CellArrays:
    """Arrays of single-cell results for a perturbation."""

    key: str
    pert_real: np.ndarray
    pert_pred: np.ndarray
    ctrl_real: np.ndarray
    ctrl_pred: np.ndarray
