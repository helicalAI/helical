from helical.models.uce.model import UCE, UCEConfig
import anndata as ad
from helical.models.uce.fine_tuning_model import UCEFineTuningModel
from omegaconf import DictConfig
import hydra

@hydra.main(version_base=None, config_path="configs", config_name="uce_config")
def run_fine_tuning(cfg: DictConfig):
    uce_config=UCEConfig(**cfg)
    uce = UCE(configurer=uce_config)

    ann_data = ad.read_h5ad("./10k_pbmcs_proc.h5ad")

    dataset = uce.process_data(ann_data[:10], name="train")
    cell_types = list(ann_data.obs.cell_type[:10])

    label_set = set(cell_types)
    class_id_dict = dict(zip(label_set, [i for i in range(len(label_set))]))

    for i in range(len(cell_types)):
        cell_types[i] = class_id_dict[cell_types[i]]

    uce_fine_tune = UCEFineTuningModel(uce_model=uce, fine_tuning_head="classification", output_size=len(label_set))
    uce_fine_tune.fine_tune(train_input_data=dataset, train_labels=cell_types)
