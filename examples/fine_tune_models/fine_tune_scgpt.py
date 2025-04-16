from helical.models.scgpt import scGPTConfig, scGPTFineTuningModel
from helical.utils import get_anndata_from_hf_dataset
from datasets import load_dataset
import anndata as ad
from omegaconf import DictConfig
import hydra


@hydra.main(
    version_base=None, config_path="../run_models/configs", config_name="scgpt_config"
)
def run_fine_tuning(cfg: DictConfig):

    # either load via huggingface
    # hf_dataset = load_dataset("helical-ai/yolksac_human",split="train[:5%]", trust_remote_code=True, download_mode="reuse_cache_if_exists")
    # ann_data = get_anndata_from_hf_dataset(hf_dataset)

    ann_data = ad.read_h5ad("./yolksac_human.h5ad")

    cell_types = ann_data.obs["LVL1"][:10].tolist()
    label_set = set(cell_types)

    scgpt_config = scGPTConfig(**cfg)
    scgpt_fine_tune = scGPTFineTuningModel(
        scGPT_config=scgpt_config,
        fine_tuning_head="classification",
        output_size=len(label_set),
    )

    dataset = scgpt_fine_tune.process_data(ann_data[:10])

    class_id_dict = {label: i for i, label in enumerate(label_set)}
    cell_types = [class_id_dict[cell] for cell in cell_types]

    scgpt_fine_tune.train(train_input_data=dataset, train_labels=cell_types)

    outputs = scgpt_fine_tune.get_outputs(dataset)
    print(outputs)

    # save and load model
    scgpt_fine_tune.save_model("./scgpt_fine_tuned_model.pt")
    scgpt_fine_tune.load_model("./scgpt_fine_tuned_model.pt")

    outputs = scgpt_fine_tune.get_outputs(dataset)
    print(outputs)

if __name__ == "__main__":
    run_fine_tuning()
