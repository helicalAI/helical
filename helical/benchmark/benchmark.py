from helical.benchmark.tasks.classifier import Classifier
from anndata import AnnData
import logging
from numpy import ndarray
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score

LOGGER = logging.getLogger(__name__)

class Benchmark():
    def __init__(self):
        pass

    def evaluate_classification(self, models: list[Classifier], eval_anndata: AnnData) -> dict[str, dict[str, float]]:
        """Classification task training and evaluating a type of classification_model.
        This is done for each entry in the train_data and eval_data dictionaries.
        
        If for example train_data = {"Geneformer": geneformer_embeddings_train, "scGPT": scgpt_embeddings_train}
        and eval_data = {"Geneformer": geneformer_embeddings_eval, "scGPT": scgpt_embeddings_eval} and the classification_model is a NeuralNetwork,
        then two classification NNs will be trained on geneformer_embeddings_train and scgpt_embeddings_train.
        
        The predictions of these models will be evaluated on geneformer_embeddings_eval and scgpt_embeddings_eval.
    
        Parameters
        ----------
        classification_model : BaseTaskModel
            The type of classification model to use.
        train_labels : ndarray
            The training labels.
        eval_labels : ndarray
            The evaluation labels.

        Returns
        -------
        A dictionary containing the evaluations for each HelicalBaseModel provided in the initialization.

        """
        self.eval_anndata = eval_anndata
        evaluations = {}
        eval_labels = np.array(eval_anndata.obs["cell_type"].tolist())
        for model in models:
            LOGGER.info(f"Processing evaluation embeddings using {model.name}.")
            prediction = model.get_predictions(eval_anndata)
            evaluation = self.get_evaluations(prediction, eval_labels)
            evaluations.update({model.name: evaluation})
        return evaluations

    def get_evaluations(self, y_true: ndarray, y_pred: ndarray) -> dict[str, float]:
        """Based on the predictions and the ground truth, calculate evaluation metrics: accuracy, precision, f1, recall.
        For the evaluation labels, the same encoder used for the training labels is used.

        Parameters
        ----------
        predictions_dict : dict[str, ndarray]
            The predictions for each model.
        eval_labels : ndarray
            The ground truth labels for the evaluation dataset.

        Returns
        -------
        A dictionary containing the evaluations for each model.
        """
        evaluation = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average='macro'),
            "f1": f1_score(y_true, y_pred, average='macro'),
            "recall": recall_score(y_true, y_pred, average='macro'),
        }        
        return evaluation