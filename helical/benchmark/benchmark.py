from helical.models.classification.classifier import Classifier
from anndata import AnnData
import logging
from numpy import ndarray
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from scib.metrics import metrics
from omegaconf import DictConfig
from copy import deepcopy

LOGGER = logging.getLogger(__name__)

def evaluate_integration(data_list: list[tuple[str, str]], adata: AnnData, data_cfg: DictConfig, integration_cfg: DictConfig) -> dict[str, dict[str, float]]:
    """
    Evaluate the data integration of the anndata object using the scib metrics. 

    Parameters
    ----------
    data_list : list[tuple[str, AnnData, str]]
        A list of tuples containing:
        The name of the model that was used to generate the embeddings.
        The name of the obsm attribute that contains the embeddings.
    adata : AnnData
        The AnnData object that contains the embeddings.
    data_cfg : DictConfig
        The configuration of the data. It must contain the keys "batch_key" and "label_key".
    integration_cfg : DictConfig
        The configuration of the integration.

    Returns
    -------
    A dictionary containing the evaluations for each model specified in the tuple.

    """
    evaluations = {}
    for model, embed_obsm_name in data_list:
        LOGGER.info(f"Processing integration evaluation using...")

        # because scib library modifies the adata object, we need to deepcopy it for each model
        # otherwise, some evaluations will be identical and thus incorrect 
        evaluation = _get_integration_evaluations(deepcopy(adata),
                                                  deepcopy(adata),
                                                  data_cfg["batch_key"], 
                                                  data_cfg["label_key"], 
                                                  embed_obsm_name, 
                                                  **integration_cfg)
        evaluations.update({model: evaluation})
    return evaluations

def evaluate_classification(models: list[Classifier], eval_anndata: AnnData, labels_column_name: str) -> dict[str, dict[str, float]]:
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
    A dictionary containing the evaluations for each HelicalBaseFoundationModel provided in the initialization.

    """
    try:
        eval_labels = np.array(eval_anndata.obs[labels_column_name].tolist())
    except KeyError:
        message = f"Column {labels_column_name} not found in the evaluation data."
        LOGGER.error(message)
        raise TypeError(message)

    eval_anndata = eval_anndata
    evaluations = {}
    for model in models:
        LOGGER.info(f"Processing classification evaluation embeddings using {model.name}.")
        prediction = model.get_predictions(eval_anndata)
        evaluation = _get_classification_evaluations(prediction, eval_labels)
        evaluations.update({model.name: evaluation})
    return evaluations

def _get_integration_evaluations(adata: AnnData, 
                                 adata_int: AnnData,
                                 batch_key: str, 
                                 label_key: str, 
                                 embed_obsm_name: str, 
                                 **configs) -> dict[str, float]:
    """
    Wrapper for all metrics used in the scib study.
    Metrics are computed according to the provided configs.
    Have a look at the User Guide for more information: 
    https://scib.readthedocs.io/en/latest/user_guide.html
    
    Parameters
    ----------
    adata : AnnData
        Unintegrated, preprocessed anndata object
    adata_int : AnnData
        Integrated anndata object
    batch_key : str
        Name of the batch column in adata.obs and adata_int.obs
    label_key : str
        Name of the biological label (cell type) column in adata.obs and adata_int.obs
    embed_obsm_name : str
        The embedding representation in adata_int.obsm
        Used for:
            + silhouette scores (label ASW, batch ASW),
            + PC regression,
            + cell cycle conservation,
            + isolated label scores, and
            + kBET
    configs : DictConfig
        Configuration for the metrics calculation as key-value pairs, coming from the config file.
    """

    results = metrics(adata, adata_int, batch_key, label_key, embed = embed_obsm_name, **configs)
    result_dict = results[0].to_dict()

    # rearrange 
    evaluations = {
                    "BIO": 
                        {
                            "ARI_cell": result_dict["ARI_cluster/label"],
                            "NMI_cell": result_dict["NMI_cluster/label"],
                            "ASW_cell": result_dict["ASW_label"],
                            },
                    "BATCH": 
                        {
                            "ASW_batch": result_dict["ASW_label/batch"],
                            "Graph_Conn": result_dict["graph_conn"],
                            },
    }
    return evaluations


def _get_classification_evaluations(y_true: ndarray, y_pred: ndarray) -> dict[str, float]:
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
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average='macro'),
        "F1": f1_score(y_true, y_pred, average='macro'),
        "Recall": recall_score(y_true, y_pred, average='macro'),
    }        
    return evaluation