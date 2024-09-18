from helical.models.geneformer.geneformer_config import GeneformerConfig
from helical.models.geneformer.fine_tuning_model import GeneformerFineTuningModel
from helical.models.geneformer.model import Geneformer
import anndata as ad
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="configs", config_name="geneformer_config")
def run_fine_tuning(cfg: DictConfig):
    geneformer_config = GeneformerConfig(**cfg)
    geneformer = Geneformer(configurer = geneformer_config)
                            
    ann_data = ad.read_h5ad("./10k_pbmcs_proc.h5ad")

    cell_types = list(ann_data.obs.cell_type[:10])
    dataset = geneformer.process_data(ann_data[:10])

    dataset = dataset.add_column('cell_types', cell_types)
    label_set = set(dataset["cell_types"])
    class_id_dict = dict(zip(label_set, [i for i in range(len(label_set))]))

    def classes_to_ids(example):
        example["cell_types"] = class_id_dict[example["cell_types"]]
        return example

    dataset = dataset.map(classes_to_ids, num_proc=1)

    geneformer_fine_tune = GeneformerFineTuningModel(geneformer_model=geneformer, fine_tuning_head="classification", output_size=len(label_set))
    geneformer_fine_tune.fine_tune(train_dataset=dataset["train"], validation_dataset=dataset["test"])

if __name__ == "__main__":
    run_fine_tuning()