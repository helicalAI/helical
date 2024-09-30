from helical.benchmark.benchmark import evaluate_classification, evaluate_integration
from helical.models.geneformer.model import Geneformer
from helical.models.geneformer.geneformer_config import GeneformerConfig
from helical.models.scgpt.model import scGPT
from helical.models.scgpt.scgpt_config import scGPTConfig
from helical.models.uce.model import UCE
from helical.models.uce.uce_config import UCEConfig
from helical.models.classification.svm import SupportVectorMachine
from helical.models.classification.classifier import Classifier
from helical.utils import get_anndata_from_hf_dataset
from datasets import load_dataset
import anndata as ad
from omegaconf import DictConfig
import hydra
import json
import logging
import os
from anndata import AnnData
import scanpy as sc
import scanpy.external as sce

LOGGER = logging.getLogger(__name__)

class Scanorama():
    def __init__(self, batch_key: str):
        self.key = batch_key

    def process_data(self, data, **kwargs) -> AnnData:
        data = data[data.obs[self.key].sort_values().index]
        return data

    def get_embeddings(self, data: AnnData):
        sc.pp.recipe_zheng17(data)
        sc.pp.pca(data)
        sce.pp.scanorama_integrate(data, self.key, verbose=1)
        return data.obsm["X_scanorama"]


def write_to_json(evaluations: dict, path: str, file_name: str) -> None:
    """
    Write the evaluations to a JSON file.

    Parameters:
    -----------
        evaluations: dict
            The evaluations to be written to the JSON file.
        path: str
            The path where the JSON file will be saved.
        file_name: str
            The name of the JSON file.
    """
    # Serializing json
    json_object = json.dumps(evaluations, indent=4)
    
    os.makedirs(os.path.dirname(path), exist_ok=True)  

    # Writing to sample.json
    with open(f"{path}{file_name}.json", "w+") as outfile:
        outfile.write(json_object)

def get_model(model_name: str, data_cfg: DictConfig, device: str = "cpu"):
    if model_name == "geneformer":
        configurer = GeneformerConfig(device=device, batch_size=5)
        return Geneformer(configurer)
    elif model_name == "scgpt":
        configurer = scGPTConfig(device=device)
        return scGPT(configurer)
    elif model_name == "uce":
        configurer = UCEConfig(device=device)
        return UCE(configurer)
    elif model_name == "scanorama":
        return Scanorama(data_cfg["batch_key"])
    else:
        raise ValueError(f"Model {model_name} not found.")

def run_classification_example(data: AnnData, probing_models: list[str], data_cfg: DictConfig, head_cfg: DictConfig, device: str = "cpu"):
    """
    This only runs with probing models. Meaning that a Classifier head is trained on top of the embeddings generated by a foundation model.

    Parameters
    ----------
    data : AnnData
        The data to be classified.
    probing_models : list[str]
        The models to be used for classification.
    data_cfg : DictConfig
        The configuration of the data. It must contain the keys "gene_names" and "label_key".
    head_cfg : DictConfig
        The configuration what model head to use and how it is configured.
    device : str
        The device to run the models on. Default is "cpu".
    """

    train_data = data[:int(len(data)*0.8)]
    eval_data = data[int(len(data)*0.8):]
    evaluations_all = {}
    
    for model_name in probing_models:
        model = get_model(model_name, data_cfg, device)
        model_c = Classifier().train_classifier_head(train_data,
                                                        model, 
                                                        SupportVectorMachine(**head_cfg), 
                                                        gene_names=data_cfg["gene_names"],
                                                        labels_column_name=data_cfg["label_key"],
                                                        test_size=0.1,
                                                        random_state=42) 
        evaluations = evaluate_classification([model_c], eval_data, data_cfg["label_key"])
        evaluations_all.update({model_name: evaluations})
        # save outputs
        model_c.trained_task_model.save(f"{data_cfg['base_dir']}/{data_cfg['name']}/cell_type_annotation/{model_name}/")        
        write_to_json(evaluations, f"{data_cfg['base_dir']}/{data_cfg['name']}/cell_type_annotation/", f"classification_evaluations_{model_name}")

        # memory management
        del model_c, model

def run_integration_example(data: AnnData, models: list[str], data_cfg: DictConfig, integration_cfg: DictConfig, device: str = "cpu"):
    """
    Run integration evaluation for the specified models.

    Parameters
    ----------
    data : AnnData
        The data to be integrated.
    models : list[str]
        The models to be used for integration.
    data_cfg : DictConfig
        The configuration of the data. 
    integration_cfg : DictConfig
        The configuration of the integration.
    device : str
        The device to run the models on. Default is "cpu".
    """

    for model_name in models:
        model = get_model(model_name, data_cfg, device)
        evaluations = evaluate_integration([(model, model_name)], data, data_cfg, integration_cfg)
        write_to_json(evaluations, f"{data_cfg['base_dir']}/{data_cfg['name']}/integration/", f"integration_evaluations_{model_name}")
        del model

@hydra.main(version_base=None, config_path=".", config_name="config")
def benchmark(cfg: DictConfig) -> None:

    cfg["device"] = "cuda"
    dataset = "yolksac"

    data_cfg = cfg["data"][dataset]
    head_cfg = cfg["svm"]
    integration_cfg = cfg["integration"]

    hf_dataset = load_dataset(data_cfg["path"], split="train[:10%]", trust_remote_code=True, download_mode="reuse_cache_if_exists")
    data = get_anndata_from_hf_dataset(hf_dataset)[:100]
    data.obs[data_cfg["label_key"]] = data.obs[data_cfg["label_key"]].astype("category")

    # set gene names. for example if the index is the ensemble gene id 
    # data.var_names = data.var["feature_name"]

    run_classification_example(data, ["geneformer", "scgpt"], data_cfg, head_cfg, device=cfg["device"])
    # run_integration_example(data, ["geneformer", "scgpt", "scanorama"], data_cfg, integration_cfg, device=cfg["device"])
    LOGGER.info("Benchmarking done.")

if __name__ == "__main__":
    benchmark()