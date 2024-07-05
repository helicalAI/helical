from helical.benchmark.benchmark import evaluate_classification, evaluate_integration
from helical.models.geneformer.model import Geneformer
from helical.models.scgpt.model import scGPT
from helical.models.uce.model import UCE
from helical.models.classification.neural_network import NeuralNetwork
from helical.models.classification.svm import SupportVectorMachine as SVM
from helical.models.classification.classifier import Classifier
from helical.services.downloader import Downloader
import anndata as ad
from omegaconf import DictConfig
import hydra
import json
from pathlib import Path
import scanpy as sc
import scanpy.external as sce

geneformer = Geneformer()
scgpt = scGPT()
uce = UCE()

def write_to_json(evaluations: dict, file_name: str) -> None:
    """
    Write the evaluations to a JSON file.

    Parameters:
    -----------
        evaluations: dict
            The evaluations to be written to the JSON file.
        file_name: str
            The name of the JSON file.
    """
    # Serializing json
    json_object = json.dumps(evaluations, indent=4)
    
    # Writing to sample.json
    with open(f"data/benchmark/{file_name}.json", "w") as outfile:
        outfile.write(json_object)

def run_classification_example(data: ad.AnnData, cfg: DictConfig) -> dict[str, dict[str, float]]:
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
        data: AnnData
            The data to be used for the classification evaluation.
        cfg: DictConfig
            The configuration for the classification evaluation.
    
    Returns:
    --------
        dict[str, dict[str, float]]
            The evaluations for the classification models.
    """
    train_data = data[:20]
    eval_data = data[20:25]
    
    # saved_nn_head = NeuralNetwork().load('my_model.h5', 'classes.npy')
    # scgpt_loaded_nn = Classifier().load_model(scgpt, saved_nn_head, "scgpt with saved NN")    
    
    # saved_svm_head = SVM().load('my_svm.pkl')
    # scgpt_loaded_svm = Classifier().load_model(scgpt, saved_svm_head, "scgpt with saved SVM")           

    uce_c = Classifier().train_classifier_head(train_data, uce, NeuralNetwork(**cfg["neural_network"]))
    scgpt_nn_c = Classifier().train_classifier_head(train_data, scgpt, NeuralNetwork(**cfg["neural_network"]))
    scgpt_nn_c.trained_task_model.save("cell_type_annotation/scgpt_w_nn")

    return evaluate_classification([scgpt_nn_c, uce_c], eval_data, "cell_type")

def run_integration_example(adata: ad.AnnData, cfg: DictConfig) -> dict[str, dict[str, float]]:
    """
    This function shows one example how to run an integration evaluation.
    We get the embeddings for the UCE and scGPT models and compare them to the
    scanorama integration.

    Parameters:
    -----------
        adata: AnnData
            The data to be used for the integration evaluation.
        cfg: DictConfig
            The configuration for the integration evaluation.
    
    Returns:
    --------
        dict[str, dict[str, float]]
            The evaluations for the integration models.
    """

    batch_0 = adata[adata.obs["batch"]==0]
    batch_1 = adata[adata.obs["batch"]==1]
    adata = ad.concat([batch_0[:20], batch_1[:20]])
    
    dataset = uce.process_data(adata)
    adata.obsm["X_uce"] = uce.get_embeddings(dataset)

    dataset = scgpt.process_data(adata)
    adata.obsm["X_scgpt"] = scgpt.get_embeddings(dataset)

    # data specific configurations
    cfg["data"]["batch_key"] = "batch"
    cfg["data"]["label_key"] = "str_labels"

    sc.pp.recipe_zheng17(adata)
    sc.pp.pca(adata)
    sce.pp.scanorama_integrate(adata, 'batch', verbose=1)

    return evaluate_integration(
            [
                ("scgpt", "X_scgpt"),
                ("uce", "X_uce"),
                ("scanorama", "X_scanorama")
            ], adata, cfg
        )


@hydra.main(version_base=None, config_path=".", config_name="config")
def benchmark(cfg: DictConfig) -> None:

    data = ad.read_h5ad("./10k_pbmcs_proc.h5ad")

    evaluations_c = run_classification_example(data, cfg)
    write_to_json(evaluations_c, "classification_evaluations")

    evaluations_i = run_integration_example(data, cfg)
    write_to_json(evaluations_i, "integration_evaluations")

if __name__ == "__main__":
    downloader = Downloader()
    downloader.download_via_link(Path("./10k_pbmcs_proc.h5ad"), "https://helicalpackage.blob.core.windows.net/helicalpackage/data/10k_pbmcs_proc.h5ad")
    benchmark()