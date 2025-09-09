from helical.models.scgpt import stateConfig, stateFineTuningModel
from omegaconf import DictConfig
import hydra
import scanpy as sc


@hydra.main(version_base=None, config_path="../run_models/configs", config_name="state_train_configs")
def run_fine_tuning(cfg: DictConfig):

    # Load the desired dataset
    adata = sc.read_h5ad("competition_support_set/competition_val_template.h5ad")

    # Get the desired label class
    cell_types = list(adata.obs.cell_type)
    label_set = set(cell_types)

    # Create the fine-tuning model (no need to specify var_dims location)
    config = stateConfig(
        batch_size=8,
        model_dir="competition/first_run",
        model_config="../run_models/configs/state_config.yaml",
        freeze_backbone=True
    )

    model = stateFineTuningModel(
        configurer=config, 
        fine_tuning_head="classification", 
        output_size=len(label_set),
    )

    # Process the data for training
    data = model.process_data(adata)

    # Create a dictionary mapping the classes to unique integers for training
    class_id_dict = dict(zip(label_set, [i for i in range(len(label_set))]))

    for i in range(len(cell_types)):
        cell_types[i] = class_id_dict[cell_types[i]]

    # Fine-tune
    model.train(train_input_data=data, train_labels=cell_types)
    return
    
if __name__ == "__main__":
    run_fine_tuning()   