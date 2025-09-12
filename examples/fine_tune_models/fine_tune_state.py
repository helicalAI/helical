from helical.models.state import stateFineTuningModel
import scanpy as sc
from helical.models.state import stateConfig
import numpy as np
from omegaconf import DictConfig
import hydra

@hydra.main(version_base=None, config_path="../run_models/configs", config_name="state_config")
def run_fine_tuning(cfg: DictConfig):

    adata = sc.read_h5ad("./yolksac_human.h5ad")
    adata = adata[:10, :2000].copy()

    perturbations = [
        "[('DMSO_TF', 0.0, 'uM')]",  # Control
        "[('Aspirin', 0.5, 'uM')]",
        "[('Dexamethasone', 1.0, 'uM')]",
    ]

    n_cells = adata.n_obs
    adata.obs['target_gene'] = np.random.choice(perturbations, size=n_cells)
    adata.obs['cell_type'] = adata.obs['LVL1']  # Use your cell type column
    # we can also add a batch variable to take into account batch effects
    batch_labels = np.random.choice(['batch_1', 'batch_2', 'batch_3', 'batch_4'], size=n_cells)
    adata.obs['batch_var'] = batch_labels

    # Dummy cell types and labels for demonstration
    cell_types = list(adata.obs['LVL1'])
    label_set = set(cell_types)
    print(f"Found {len(label_set)} unique cell types:")

    config = stateConfig(
        embed_key=None,
        pert_col="target_gene",
        celltype_col="cell_type",
        control_pert="[('DMSO_TF', 0.0, 'uM')]",
        batch_size=8,
    )

    # Create the fine-tuning model - we use a classification head for demonstration
    model = stateFineTuningModel(
        configurer=config, 
        fine_tuning_head="classification", 
        output_size=len(label_set),
    )

    # Process the data for training - returns a dataset object
    dataset = model.process_data(adata)

    # Create a dictionary mapping the classes to unique integers for training
    class_id_dict = dict(zip(label_set, [i for i in range(len(label_set))]))

    # Convert cell type labels to integers
    cell_type_labels = [class_id_dict[ct] for ct in cell_types]

    print(f"Class mapping: {class_id_dict}")

    # Fine-tune
    model.train(train_input_data=dataset, train_labels=cell_type_labels)

    return
    
if __name__ == "__main__":
    run_fine_tuning()   