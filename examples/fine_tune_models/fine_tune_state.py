from helical.models.scgpt import stateConfig, stateFineTuningModel
from omegaconf import DictConfig
import hydra
import scanpy as sc


@hydra.main(version_base=None, config_path="../run_models/configs", config_name="state_config")
def run_fine_tuning(cfg: DictConfig):
    ann_data = sc.read_h5ad("./competition_support_set/competition_val_template.h5ad")
    state_config = stateConfig()
    state_model = stateFineTuningModel(configurer=state_config)
    ann_data = state_model.process_data(ann_data)
    state_model.train(ann_data)
    state_model.predict()
    return 
    
if __name__ == "__main__":
    run_fine_tuning()   