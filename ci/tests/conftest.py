import pytest
import scanpy as sc

@pytest.fixture(scope="session")
def adata_pbmc_template():
    """
    This fixture generates a concatenated version of the pbmc3k and pbmc68k datasets.
    It is saved (via debugger with the anndata.write method) and tracked with git-lfs to speed up testing.
    """
    adata_ref = sc.datasets.pbmc3k_processed()
    adata = sc.datasets.pbmc68k_reduced()

    var_names = adata_ref.var_names.intersection(adata.var_names)
    adata_ref = adata_ref[:, var_names]
    adata = adata[:, var_names]

    sc.pp.pca(adata_ref)
    sc.pp.neighbors(adata_ref)
    sc.tl.umap(adata_ref)

    # merge cell type labels
    sc.tl.ingest(adata, adata_ref, obs="louvain")
    adata_concat = adata_ref.concatenate(adata, batch_categories=["ref", "new"])
    adata_concat.obs.louvain = adata_concat.obs.louvain.astype("category")
    # fix category ordering
    adata_concat.obs["louvain"] = adata_concat.obs["louvain"].cat.set_categories(
        adata_ref.obs["louvain"].cat.categories
    )
    adata_concat.obs["celltype"] = adata_concat.obs["louvain"]

    del adata_concat.obs["louvain"]
    del adata_concat.uns
    del adata_concat.obsm
    del adata_concat.varm

    yield adata_concat
    del adata_concat

@pytest.fixture()
def load_adata_pbmc_template():
    """
    Load the saved file from the data folder. The saved file is a concatenated version of the pbmc3k and pbmc68k datasets
    and was generated using the code in the adata_pbmc_template fixture.
    Loading from disk speeds up testing.
    """
    adata_obj = sc.read_h5ad("./ci/tests/data/pbmc3k_concat_68k.h5ad")
    yield adata_obj
    del adata_obj

@pytest.fixture()
def adata(load_adata_pbmc_template):
    adata_obj = load_adata_pbmc_template.copy()
    yield adata_obj
    del adata_obj

@pytest.fixture()
def adata_pca(adata):
    """
    Save the PCA results to the adata object in the X_pca key.
    Taken and modified to not use highly variable genes from the scIB preprocessing reduce_data function.

    Parameters
    ----------
        adata: AnnData
            The AnnData object to perform PCA on.
    """
    sc.tl.pca(
        adata,
        n_comps=50,
        use_highly_variable=False,
        svd_solver="arpack",
        return_info=True,
    )
    yield adata

def assert_near_exact(x: float, y: float, diff: float = 1e-5):
    """
    Asserts that two floating-point numbers are nearly equal within a given tolerance.

    Parameters
    ----------
        x: float,
            The first floating-point number.
        y: float
            The second floating-point number.
        diff: float, optional
            The maximum allowed difference between x and y. Defaults to 1e-5.

    Raises
    ------
        AssertionError: If the absolute difference between x and y is greater than diff.
    """
    assert abs(x - y) <= diff, f"{x} != {y} with error margin {diff}"
