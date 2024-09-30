from helical import GeneformerConfig, Geneformer, GeneformerFineTuningModel
from helical.services.converter import get_anndata_from_hf_dataset
from datasets import load_dataset
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="../run_models/configs", config_name="geneformer_config")
def run_fine_tuning(cfg: DictConfig):
    geneformer_config = GeneformerConfig(**cfg)
    geneformer = Geneformer(configurer = geneformer_config)
                            
    hf_dataset = load_dataset("helical-ai/yolksac_human",split="train[:5%]", trust_remote_code=True, download_mode="reuse_cache_if_exists")
    ann_data = get_anndata_from_hf_dataset(hf_dataset)

    cell_types = list(ann_data.obs["LVL1"][:10])

    dataset = geneformer.process_data(ann_data[:10])

    dataset = dataset.add_column('cell_types', cell_types)
    label_set = set(dataset["cell_types"])
    class_id_dict = dict(zip(label_set, [i for i in range(len(label_set))]))

    def classes_to_ids(example):
        example["cell_types"] = class_id_dict[example["cell_types"]]
        return example

    dataset = dataset.map(classes_to_ids, num_proc=1)

    geneformer_fine_tune = GeneformerFineTuningModel(geneformer_model=geneformer, fine_tuning_head="classification", output_size=len(label_set))
    geneformer_fine_tune.train(train_dataset=dataset)

if __name__ == "__main__":
    run_fine_tuning()