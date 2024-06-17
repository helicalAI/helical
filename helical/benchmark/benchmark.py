from helical.models.helical import HelicalBaseModel
from helical.benchmark.tasks.classifier import Classifier
from helical.benchmark.task_models.base_task_model import BaseTaskModel
from anndata import AnnData
import logging
from numpy import ndarray

LOGGER = logging.getLogger(__name__)

class Benchmark():
    def __init__(self, models: list[HelicalBaseModel], train_anndata: AnnData, eval_anndata: AnnData) -> None:

        self.train_anndata = train_anndata
        self.train_data = {}

        self.eval_anndata = eval_anndata
        self.eval_data = {}

        for model in models:
            if isinstance(model, HelicalBaseModel):
                LOGGER.info(f"Getting training embeddings with {model.__class__.__name__}.")
                dataset = model.process_data(train_anndata)
                self.train_data.update({model.__class__.__name__: model.get_embeddings(dataset)})

                LOGGER.info(f"Processing evaluation embeddings with {model.__class__.__name__}.")
                dataset = model.process_data(eval_anndata)
                self.eval_data.update({model.__class__.__name__: model.get_embeddings(dataset)})
    
            else:
                pass

    def classification(self, classification_model: BaseTaskModel, train_labels: ndarray, eval_labels: ndarray) -> dict[str, dict[str, float]]:
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
        self.classifier = Classifier(train_labels, classification_model)
        
        LOGGER.info(f"Training classification models heads of type '{classification_model.__class__.__name__}'.")
        self.classifier.train_task_models(x_train = self.train_data, test_size = 0.2, random_state = 42)
        
        LOGGER.info(f"Classification prediction with model heads of type '{classification_model.__class__.__name__}'.")
        predictions = self.classifier.get_predictions(x_eval = self.eval_data)

        LOGGER.info(f"Evaluating predictions of models '{classification_model.__class__.__name__}'.")
        evaluations = self.classifier.get_evaluations(predictions, eval_labels)
        return evaluations

            