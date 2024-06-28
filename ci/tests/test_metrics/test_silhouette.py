import numpy as np
from helical.benchmark.metrics.silhouette import silhouette_batch
from ci.tests.conftest import assert_near_exact

# silhouette from scib not yet included
# def test_silhouette(adata_pca):
#     score = scib.me.silhouette(
#         adata_pca, label_key="celltype", embed="X_pca", scale=True
#     )
#     assert_near_exact(score, 0.5626532882452011, diff=1e-2)

def test_silhouette_batch(adata_pca):
    score = silhouette_batch(
        adata_pca,
        batch_obs="batch",
        label_obs="celltype",
        embedding_obsm="X_pca",
        scale=True,
    )
    assert_near_exact(score, 0.9014384369842835, diff=1e-2)

def test_silhouette_batch_empty(adata_pca):
    adata_pca.obs["label"] = "label"
    adata_pca.obs["batch"] = "batch"
    asw, sil_means, sil_df = silhouette_batch(
        adata_pca,
        batch_obs="batch",
        label_obs="celltype",
        embedding_obsm="X_pca",
        scale=True,
        return_all=True,
    )
    assert np.isnan(asw)
    assert np.isnan(sil_means)
    assert sil_df.shape[0] == 0
