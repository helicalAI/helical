
import hydra
from omegaconf import DictConfig
import anndata as ad
from helical.models.c2s import Cell2Sen, Cell2SenConfig

@hydra.main(version_base=None, config_path="configs", config_name="c2s_config")
def run(cfg: DictConfig):

    adata = ad.read_h5ad("./yolksac_human.h5ad")
    n_cells = 10
    n_genes = 200
    adata = adata[:n_cells, :n_genes].copy()
    perturbation_column = "perturbation"
    adata.obs[perturbation_column] = ["IFNg"] * n_cells

    config = Cell2SenConfig(**cfg)
    c2s = Cell2Sen(configurer=config)

    processes_dataset = c2s.process_data(adata)
    embeddings = c2s.get_embeddings(processes_dataset)
    perturbed_dataset, perturbed_cell_sentences = c2s.get_perturbations(processes_dataset)

if __name__ == "__main__":
    run()
