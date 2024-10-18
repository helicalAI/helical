from helical import UCEConfig, UCEFineTuningModel
from helical.utils import get_anndata_from_hf_dataset
from datasets import load_dataset
from omegaconf import DictConfig
import hydra

@hydra.main(version_base=None, config_path="../run_models/configs", config_name="uce_config")
def run_fine_tuning(cfg: DictConfig):
    hf_dataset = load_dataset("helical-ai/yolksac_human",split="train[:5%]", trust_remote_code=True, download_mode="reuse_cache_if_exists")
    ann_data = get_anndata_from_hf_dataset(hf_dataset)

    cell_types = ann_data.obs["LVL1"][:10].tolist()

    label_set = set(cell_types)

    uce_config=UCEConfig(**cfg)
    uce_fine_tune = UCEFineTuningModel(uce_config=uce_config, fine_tuning_head="classification", output_size=len(label_set))

    dataset = uce_fine_tune.process_data(ann_data[:10], name="train")

    class_id_dict = {label: i for i, label in enumerate(label_set)}
    cell_types = [class_id_dict[cell] for cell in cell_types]

    uce_fine_tune.train(train_input_data=dataset, train_labels=cell_types)
    
if __name__ == "__main__":
    run_fine_tuning()