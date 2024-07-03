from helical.benchmark.benchmark import evaluate_classification
from helical.models.classification.classifier import Classifier
from helical.models.classification.neural_network import NeuralNetwork
from helical.models.scgpt.model import scGPT
import anndata as ad
import numpy as np

def test_evaluate_classification(mocker):
    scgpt = scGPT()
    data = ad.read_h5ad("ci/tests/data/cell_type_sample.h5ad")
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

    scgpt_nn_c = Classifier().train_classifier_head(data, scgpt, head, gene_col_name = "gene_symbols", labels_column_name = "cell_type")
    evaluations = evaluate_classification([scgpt_nn_c, scgpt_nn_c], data, "cell_type")
    assert evaluations == {'scGPT with NeuralNetwork': {'Accuracy': 1.0, 'Precision': 1.0, 'F1': 1.0, 'Recall': 1.0}, 
                           'scGPT with NeuralNetwork': {'Accuracy': 1.0, 'Precision': 1.0, 'F1': 1.0, 'Recall': 1.0}}