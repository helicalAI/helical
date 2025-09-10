import logging
import os
import shutil
import subprocess
from tempfile import TemporaryDirectory

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
from scipy.sparse import csr_matrix, issparse

from ._evaluator import _convert_to_normlog

LOGGER = logging.getLogger(__name__)

VALID_ENCODINGS = [64, 32]


# set the defaults here
def strip_anndata(
    adata: ad.AnnData,
    output_path: str,
    genelist: list[str],
    pert_col: str = "target_gene",
    celltype_col: str | None = None,
    output_pert_col: str = "target_gene",
    output_celltype_col: str = "celltype",
    ntc_name: str = "non-targeting",
    encoding: int = 64,
    allow_discrete: bool = False,
    max_cell_dim: int | None = 100000,
    exp_gene_dim: int | None = 18080,
):
    # Force anndata var to string
    adata.var.index = adata.var.index.astype(str)

    if pert_col not in adata.obs:
        raise ValueError(
            f"Provided perturbation column: '{pert_col}' missing from anndata: {adata.obs.columns}"
        )
    if celltype_col:
        if celltype_col not in adata.obs:
            raise ValueError(
                f"Provided celltype column: '{celltype_col}' missing from anndata: {adata.obs.columns}"
            )
    if ntc_name not in adata.obs[pert_col].unique():
        raise ValueError(
            f"Provided negative control name: '{ntc_name}' missing from anndata: {adata.obs[pert_col].unique()}"
        )

    # Check if expected dimension is provided and matches the length of the genelist
    if exp_gene_dim and len(genelist) != exp_gene_dim:
        LOGGER.warning(
            f"Provided gene dimension: {len(genelist)} does not match expected gene dimension: {exp_gene_dim}."
        )
        LOGGER.info(f"Setting expected gene dimension to {len(genelist)}")
        exp_gene_dim = len(genelist)

    if adata.var_names.tolist() != genelist:
        missing_genes = set(genelist) - set(adata.var_names.tolist())
        extra_genes = set(adata.var_names.tolist()) - set(genelist)
        if len(missing_genes) == 0 and len(extra_genes) == 0:
            LOGGER.warning(
                "Provided anndata contains all expected genes but they are out of order."
            )
            LOGGER.info("Reordering genes...")
            adata = adata[:, np.array(genelist)]
        else:
            raise ValueError(
                "Provided gene list does not match anndata gene names:\n"
                f"Missing genes: {missing_genes}\n"
                f"Extra genes: {extra_genes}"
            )

    if exp_gene_dim and adata.shape[1] != exp_gene_dim:
        raise ValueError(
            f"Provided gene dimension: {adata.shape[1]} does not match expected gene dimension: {exp_gene_dim}"
        )

    if max_cell_dim and adata.shape[0] > max_cell_dim:
        raise ValueError(
            f"Provided cell dimension: {adata.shape[0]} exceeds maximum cell dimension: {max_cell_dim}"
        )

    if encoding not in VALID_ENCODINGS:
        raise ValueError(f"Encoding must be in {VALID_ENCODINGS}")

    dtype = np.dtype(np.float64)  # force bound
    match encoding:
        case 64:
            LOGGER.info("Using 64-bit float encoding")
            dtype = np.dtype(np.float64)
        case 32:
            LOGGER.info("Using 32-bit float encoding")
            dtype = np.dtype(np.float32)

    LOGGER.info("Setting data to sparse if not already")
    new_x = (
        adata.X.astype(dtype)  # type: ignore
        if issparse(adata.X)
        else csr_matrix(adata.X.astype(dtype))  # type: ignore
    )

    LOGGER.info("Simplifying obs dataframe")
    new_obs = pd.DataFrame(
        {output_pert_col: adata.obs[pert_col].values},
        index=np.arange(adata.shape[0]).astype(str),
    )
    if celltype_col:
        new_obs[output_celltype_col] = adata.obs[celltype_col].values

    LOGGER.info("Simplifying var dataframe")
    new_var = pd.DataFrame(
        index=adata.var.index.values,
    )

    LOGGER.info("Creating final minimal AnnData object")
    minimal = ad.AnnData(
        X=new_x,
        obs=new_obs,
        var=new_var,
    )

    LOGGER.info("Applying normlog transformation if required")
    _convert_to_normlog(minimal, allow_discrete=allow_discrete)

    # Create a temporary directory to work in
    with TemporaryDirectory() as temp_dir:
        # Create temp files with specific names
        tmp_h5ad = os.path.join(temp_dir, "pred.h5ad")
        tmp_watermark = os.path.join(temp_dir, "watermark.txt")

        # Write the h5ad file
        LOGGER.info(f"Writing h5ad output to {tmp_h5ad}")
        minimal.write_h5ad(tmp_h5ad)

        # Zstd compress the h5ad file (will create pred.h5ad.zst)
        LOGGER.info(f"Zstd compressing {tmp_h5ad}")
        subprocess.run(["zstd", "-T0", "-f", "--rm", tmp_h5ad])

        # Write the watermark file
        with open(tmp_watermark, "w") as f:
            f.write("vcc-prep")

        # Pack the files into a tarball
        LOGGER.info(f"Packing files into {output_path}")
        subprocess.run(
            [
                "tar",
                "-cf",
                output_path,
                "-C",
                temp_dir,
                "pred.h5ad.zst",
                "watermark.txt",
            ]
        )

        LOGGER.info("Done")


def _validate_tools_in_path():
    if shutil.which("tar") is None:
        raise ValueError("tar is not installed")
    if shutil.which("zstd") is None:
        raise ValueError("zstd is not installed")
    return True


def vcc_eval(configs: dict):

    _validate_tools_in_path()

    LOGGER.info("Reading input anndata")
    adata = ad.read_h5ad(configs["input"])

    LOGGER.info("Reading gene list")
    genelist = (
        pl.read_csv(configs["genes"], has_header=False).to_series(0).cast(str).to_list()
    )

    LOGGER.info("Preparing anndata")
    strip_anndata(
        adata,
        genelist=genelist,
        output_path=(
            configs["output"]
            if configs["output"]
            else configs["input"].replace(".h5ad", ".prep.vcc")
        ),
        pert_col=configs["pert_col"],
        celltype_col=configs["celltype_col"],
        encoding=configs["encoding"],
        allow_discrete=configs["allow_discrete"],
        exp_gene_dim=(
            configs["expected_gene_dim"] if configs["expected_gene_dim"] != -1 else None
        ),
        max_cell_dim=configs["max_cell_dim"] if configs["max_cell_dim"] != -1 else None,
    )


# if __name__ == "__main__":

# configs = {
#     "input": "competition/prediction.h5ad",
#     "genes": "competition_support_set/gene_names.csv",
#     "output": None,
#     "pert_col": DEFAULT_PERT_COL,
#     "celltype_col": None,
#     "ntc_name": DEFAULT_NTC_NAME,
#     "output_pert_col": DEFAULT_PERT_COL,
#     "output_celltype_col": DEFAULT_CELLTYPE_COL,
#     "encoding": 32,
#     "allow_discrete": False,
#     "expected_gene_dim": EXPECTED_GENE_DIM,
#     "max_cell_dim": MAX_CELL_DIM,
# }

# vcc_eval(configs)
