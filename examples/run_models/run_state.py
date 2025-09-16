from helical.models.state import (
    StateConfig, 
    StateEmbed, 
    StatePerturb
    )
import hydra
from omegaconf import DictConfig
import anndata as ad
import scanpy as sc
import numpy as np

@hydra.main(version_base=None, config_path="configs", config_name="state_config")
def run_state(cfg: DictConfig):

    adata = sc.read_h5ad("yolksac_human.h5ad")
    # for demonstration we subset to 10 cells and 2000 genes
    adata = adata[:10, :2000].copy()

    # embedding model
    state_config = StateConfig(batch_size=16)
    state_embed = StateEmbed(configurer=state_config)
    
    processed_data = state_embed.process_data(adata=adata)
    embeddings = state_embed.get_embeddings(processed_data)

    perturbations = [
        "[('DMSO_TF', 0.0, 'uM')]",  # Control
        "[('Aspirin', 0.5, 'uM')]",
        "[('Dexamethasone', 1.0, 'uM')]",
    ]

    n_cells = adata.n_obs
    # we assign perturbations to cells randomly
    adata.obs['target_gene'] = np.random.choice(perturbations, size=n_cells)
    adata.obs['cell_type'] = adata.obs['LVL1']  # Use your cell type column
    # we can also add a batch variable to take into account batch effects
    batch_labels = np.random.choice(['batch_1', 'batch_2', 'batch_3', 'batch_4'], size=n_cells)
    adata.obs['batch_var'] = batch_labels

    config = StateConfig(
        embed_key=None,
        pert_col="target_gene",
        celltype_col="cell_type",
        control_pert="[('DMSO_TF', 0.0, 'uM')]",
        output_path="yolksac_perturbed.h5ad",
    )

    state_perturb = StatePerturb(configurer=config)

    # again we process the data and get the perturbed embeddings
    processed_data = state_perturb.process_data(adata)
    perturbed_embeds = state_perturb.get_embeddings(processed_data)

    return

if __name__ == "__main__":
    run_state()
