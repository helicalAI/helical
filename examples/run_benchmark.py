from helical.benchmark.benchmark import evaluate_classification, evaluate_integration, get_alcs_evaluations
from helical.models.geneformer.model import Geneformer
from helical.models.geneformer.geneformer_config import GeneformerConfig
from helical.models.scgpt.model import scGPT
from helical.models.scgpt.scgpt_config import scGPTConfig
from helical.models.uce.model import UCE
from helical.models.uce.uce_config import UCEConfig
from helical.models.classification.svm import SupportVectorMachine
from helical.models.classification.classifier import Classifier
import anndata as ad
from omegaconf import DictConfig
import hydra
import json
import scanpy as sc
import scanpy.external as sce
import os
from anndata import AnnData
import logging
from numpy import ndarray

LOGGER = logging.getLogger(__name__)

class Original():
    """
    This class is used to not process the data at all and
    to return the original expression matrix as the embeddings.
    This can be used as a baseline.
    """
    def __init__(self):
        pass
    def process_data(self, anndata, _) -> AnnData:
        return anndata
    def get_embeddings(self, anndata) -> ndarray:
        return anndata.X

class Scanorama():
    def __init__(self, batch_key: str):
        self.key = batch_key

    def process_data(self, data, **kwargs) -> AnnData:
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
    elif model_name == "original":
        return Original()
    else:
        raise ValueError(f"Model {model_name} not found.")

def run_classification_example(data, helical_models, data_cfg, head_cfg: DictConfig, device: str = "cpu", alcs: bool = True):
    """
    This function shows one example how to run a classification evaluation.
    Different combinations can be used, which is shown with the comments.
    - The first comment loads a saved neural network with its label encoder.
    - The second comment loads a saved SVM model.
    Currently, the function trains a neural network classifier for the UCE embeddings and
    another NN for the scGPT embeddings. It saves the model for scGPT with the label
    encoder for potential later use.

    Parameters:
    -----------
        cfg: DictConfig
            The configuration for the classification evaluation.
    
    Returns:
    --------
        dict[str, dict[str, float]]
            The evaluations for the classification models.
    """

    train_data = data[:int(len(data)*0.8)]
    eval_data = data[int(len(data)*0.8):]
    evaluations_all = {}
    
    for model_name in helical_models:
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
        del model_c
        del model

    if alcs:
        LOGGER.info("Processing ALCS evaluation.")
        alcs_evaluation = get_alcs_evaluations(evaluations_all)
        write_to_json(alcs_evaluation, f"{data_cfg['base_dir']}/{data_cfg['name']}/alcs/", "results")

def run_integration_example(data, models, data_cfg, integration_cfg: DictConfig, device: str = "cpu"):

    for model_name in models:
        model = get_model(model_name, data_cfg, device)
        evaluations = evaluate_integration([(model, model_name)], data, data_cfg, integration_cfg)
        write_to_json(evaluations, f"{data_cfg['base_dir']}/{data_cfg['name']}/integration/", f"integration_evaluations_{model_name}")
        del model

@hydra.main(version_base=None, config_path=".", config_name="config")
def benchmark(cfg: DictConfig) -> None:

    cfg["device"] = "cpu"
    dataset = "pbmc"

    data_cfg = cfg["data"][dataset]
    head_cfg = cfg["svm"]
    integration_cfg = cfg["integration"]

    data = ad.read_h5ad(data_cfg["path"])[:100]
    data.obs[data_cfg["label_key"]] = data.obs[data_cfg["label_key"]].astype("category")

    # scanorama requires continuous batches
    data = data[data.obs[data_cfg["batch_key"]].sort_values().index]

    run_classification_example(data, ["geneformer", "uce", "scgpt", "original"], data_cfg, head_cfg, device=cfg["device"], alcs=cfg["integration"]["alcs"])
    run_integration_example(data, ["geneformer", "uce", "scgpt", "scanorama"], data_cfg, integration_cfg, device=cfg["device"])
    LOGGER.info("Benchmarking done.")

if __name__ == "__main__":
    benchmark()