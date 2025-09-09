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
from datasets import load_dataset
from helical.utils import get_anndata_from_hf_dataset
import scanpy as sc

@hydra.main(version_base=None, config_path="configs", config_name="state_config")
def run_embeddings(cfg: DictConfig):

    ann_data = sc.read_h5ad("./competition_support_set/competition_val_template.h5ad")
    state_config = stateConfig()
    state_model = stateEmbeddingsModel(configurer=state_config)

    ann_data = ann_data[:10].copy()

    ann_data = state_model.process_data()
    embeddings = state_model.get_embeddings(ann_data)

    print(embeddings.shape)


def run_training(cfg: DictConfig):
    ann_data = sc.read_h5ad("./competition_support_set/competition_val_template.h5ad")
    train_config = trainingConfig()
    state_model = stateTransitionTrainModel(configurer=train_config)
    ann_data = state_model.process_data(ann_data)
    state_model.train()
    state_model.predict()
    return



def run_inference(cfg: DictConfig):
    ann_data = sc.read_h5ad("./competition_support_set/competition_val_template.h5ad")
    state_config = stateConfig()
    state_model = stateTransitionModel(configurer=state_config)
    ann_data = state_model.process_data(ann_data)
    state_model.predict()
    return


if __name__ == "__main__":
    run_embeddings()
    run_training()
    run_inference()
