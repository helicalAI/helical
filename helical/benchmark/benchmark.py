from helical.classification.classifier import Classifier
from anndata import AnnData
import logging
from numpy import ndarray
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score

LOGGER = logging.getLogger(__name__)

class Benchmark():
    def __init__(self):
        pass

    def evaluate_classification(self, models: list[Classifier], eval_anndata: AnnData, labels_column_name: str) -> dict[str, dict[str, float]]:
        """
        Evaluate the classification models using the evaluation dataset. 
        The evaluation dataset is used to calculate the accuracy, precision, f1, and recall scores.
    
        Parameters
        ----------
        models : list[Classifier]
            The list of models to evaluate.
        eval_anndata : AnnData
            The evaluation data.
        labels_column_name : str
            The name of the column in the obs attribute of the AnnData object that contains the labels.

        Raises
        ------
        TypeError
            If the column with the labels is not found in the evaluation data.

        Returns
        -------
        A dictionary containing the evaluations for each HelicalBaseModel provided in the initialization.

        """
        try:
            eval_labels = np.array(eval_anndata.obs[labels_column_name].tolist())
        except KeyError:
            message = f"Column {labels_column_name} not found in the evaluation data."
            LOGGER.error(message)
            raise TypeError(message)

        self.eval_anndata = eval_anndata
        evaluations = {}
        for model in models:
            LOGGER.info(f"Processing evaluation embeddings using {model.name}.")
            prediction = model.get_predictions(eval_anndata)
            evaluation = self.get_evaluations(prediction, eval_labels)
            evaluations.update({model.name: evaluation})
        return evaluations

    def get_evaluations(self, y_true: ndarray, y_pred: ndarray) -> dict[str, float]:
        """
        Based on the predictions and the ground truth, calculate evaluation metrics: accuracy, precision, f1, recall.

        Parameters
        ----------
        y_pred : ndarray
            The predicted labels.
        
        y_true : ndarray
            The ground truth labels.

        Returns
        -------
        A dictionary containing the evaluations.
        """
        evaluation = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average='macro'),
            "f1": f1_score(y_true, y_pred, average='macro'),
            "recall": recall_score(y_true, y_pred, average='macro'),
        }        
        return evaluation