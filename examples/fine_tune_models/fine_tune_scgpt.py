from helical.models.scgpt.fine_tuning_model import scGPTFineTuningModel
from helical.models.scgpt.model import scGPT,scGPTConfig
import anndata as ad
from omegaconf import DictConfig
import hydra


@hydra.main(version_base=None, config_path="configs", config_name="scgpt_config")
def run_fine_tuning(cfg: DictConfig):
    scgpt_config=scGPTConfig(**cfg)
    scgpt = scGPT(configurer=scgpt_config)

    ann_data = ad.read_h5ad("./10k_pbmcs_proc.h5ad")
    dataset = scgpt.process_data(ann_data[:10])
    cell_types = list(ann_data.obs.cell_type[:10])

    label_set = set(cell_types)

    class_id_dict = dict(zip(label_set, [i for i in range(len(label_set))]))

    for i in range(len(cell_types)):
        cell_types[i] = class_id_dict[cell_types[i]]

    scgpt_fine_tune = scGPTFineTuningModel(scGPT_model=scgpt, fine_tuning_head="classification", output_size=len(label_set))
    scgpt_fine_tune.train(train_input_data=dataset, train_labels=cell_types)

if __name__ == "__main__":
    run_fine_tuning()