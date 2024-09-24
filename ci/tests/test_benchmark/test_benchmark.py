from helical.benchmark.benchmark import evaluate_classification, evaluate_integration
from helical.models.classification.classifier import Classifier
from helical.models.classification.neural_network import NeuralNetwork
from helical.models.scgpt.model import scGPT
import anndata as ad
import numpy as np
import pandas as pd
from anndata import AnnData

scgpt = scGPT()
data = ad.read_h5ad("ci/tests/data/cell_type_sample.h5ad")
config = {
    "data": {
        "batch_key": "batch",
        "label_key": "str_labels",
        "gene_names": "index"
    },
    "integration": {
        "scib": {
            "isolated_labels_asw_": False,
            "silhouette_": True,
            "hvg_score_": False,
            "graph_conn_": True,
            "pcr_": True,
            "isolated_labels_f1_": False,
            "trajectory_": False,
            "nmi_": True,  # use the clustering bias to the best matching
            "ari_": True,  # use the clustering bias to the best matching
            "cell_cycle_": False,
            "kBET_": False,  # kBET return nan sometimes need to examine
            "ilisi_": False,
            "clisi_": False
        }
    }
}

generator = np.random.default_rng(seed=42)
data.obs["batch"] = generator.choice([0, 1], size=data.shape[0])
data.obs["str_labels"] = pd.Categorical(['type1', 'type2'] * (data.shape[0] // 2))
data.obsm["X_scgpt"] = np.zeros((data.shape[0], 10))

def assert_near_exact(x, y, diff=1e-5):
    assert abs(x - y) <= diff, f"{x} != {y} with error margin {diff}"

def test_evaluate_classification(mocker):
    head = NeuralNetwork()

    # Mocking the get_embeddings method by returning a zero matrix
    mocker.patch.object(scgpt, 'get_embeddings')
    scgpt.get_embeddings.return_value = np.zeros((data.shape[0], 10))
    
    # Mocking the training process by returning the untrained head
    mocker.patch.object(head, 'train')
    head.train.return_value = head

    # Mocking the prediction process by returning the untrained head
    mocker.patch.object(head, 'predict')
    head.predict.return_value = np.array(['type1', 'type2'] * (data.shape[0] // 2))

    scgpt_nn_c = Classifier().train_classifier_head(data, scgpt, head, gene_names = "gene_symbols", labels_column_name = "cell_type")
    evaluations = evaluate_classification([scgpt_nn_c, scgpt_nn_c], data, "cell_type")
    assert evaluations == {'scGPT with NeuralNetwork': {'Accuracy': 1.0, 'Precision': 1.0, 'F1': 1.0, 'Recall': 1.0}, 
                           'scGPT with NeuralNetwork': {'Accuracy': 1.0, 'Precision': 1.0, 'F1': 1.0, 'Recall': 1.0}}
    
def test_evaluate_integration(mocker):
    """
    Test that the integration evaluation function returns the expected results.
    We follow the approach from scGPT to evaluate the integration based on the bioloical conservation
    and the batch correction. The metrics we are interested in are:
    - ARI (Adjusted Rand Index)
    - NMI (Normalized Mutual Information)
    - ASW (Average Silhouette Width)
    - Graph Connectivity
    """

    # Mocking the get_embeddings method by returning a zero matrix
    mocker.patch.object(scgpt, 'get_embeddings')
    scgpt.get_embeddings.return_value = np.zeros((data.shape[0], 10))

    evaluations = evaluate_integration([(scgpt, "scgpt")], data, config["data"], config["integration"])
    
    # scgpt
    assert_near_exact(evaluations["scgpt"]["BATCH"]["ASW_batch"], 1.0)
    assert_near_exact(evaluations["scgpt"]["BATCH"]["Graph_Conn"], 1.0)
    
    assert_near_exact(evaluations["scgpt"]["BIO"]["ARI_cell"], 0.0)
    assert_near_exact(evaluations["scgpt"]["BIO"]["NMI_cell"], 0.2616480412956257)
    assert_near_exact(evaluations["scgpt"]["BIO"]["ASW_cell"], 0.5)

def test_integration_with_custom_model():
    class CustomWrapper():
        def process_data(self, data, **kwargs) -> AnnData:
            return data

        def get_embeddings(self, data: AnnData):
            return np.zeros((data.shape[0], 10))
        
        def train(self):
            pass # TODO

    custom_model = CustomWrapper()
    evaluations = evaluate_integration([(custom_model, "custom_model")], data, config["data"], config["integration"])
  
    # custom_model
    assert_near_exact(evaluations["custom_model"]["BATCH"]["ASW_batch"], 1.0)
    assert_near_exact(evaluations["custom_model"]["BATCH"]["Graph_Conn"], 1.0)

    assert_near_exact(evaluations["custom_model"]["BIO"]["ARI_cell"], 0.0)
    assert_near_exact(evaluations["custom_model"]["BIO"]["NMI_cell"], 0.2616480412956257)
    assert_near_exact(evaluations["custom_model"]["BIO"]["ASW_cell"], 0.5)
