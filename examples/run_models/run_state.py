from helical.models.scgpt import (
    stateConfig, 
    stateEmbeddingsModel, 
    trainingConfig, 
    stateTransitionTrainModel, 
    stateTransitionModel
    )
import hydra
from omegaconf import DictConfig
import anndata as ad
import scanpy as sc

@hydra.main(version_base=None, config_path="configs", config_name="state_config")
def run_state(cfg: DictConfig):

    original_data = sc.read_h5ad("./competition_support_set/competition_val_template.h5ad")
    state_config = stateConfig()

    # embedding model
    state_model = stateEmbeddingsModel(configurer=state_config)
    ann_data = original_data[:10].copy()

    ann_data = state_model.process_data(ann_data)
    embeddings = state_model.get_embeddings(ann_data)

    # run training loop - intialises the model for competition data
    train_config = trainingConfig(
        output_dir="competition",
        name="first_run",
        toml_config_path="competition_support_set/starter.toml",
        checkpoint_name="final.ckpt",
        max_steps=40000,
        max_epochs=1,
        ckpt_every_n_steps=20000,
        num_workers=4,
        batch_col="batch_var",
        pert_col="target_gene",
        cell_type_key="cell_type",
        control_pert="non-targeting",
        perturbation_features_file="competition_support_set/ESM2_pert_features.pt"
        )
    
    state_train = stateTransitionTrainModel(configurer=train_config)
    state_train.train() 
    state_train.predict() 

    # run inference 
    state_config = stateConfig(
        output = "competition/prediction.h5ad",
        model_dir = "competition/first_run",
        model_config = "configs/config.yaml",
        pert_col = "target_gene",
    )

    state_transition = stateTransitionModel(configurer=state_config)
    adata = state_transition.process_data(original_data)
    embeds = state_transition.get_embeddings(adata)

    return

if __name__ == "__main__":
    run_state()
