from helical.models.helical import HelicalBaseModel
from helical.benchmark.tasks.classifier import Classifier
from helical.benchmark.task_models.base_task_model import BaseTaskModel
from anndata import AnnData
import numpy as np
import logging

LOGGER = logging.getLogger(__name__)

class Benchmark():
    def __init__(self, models: list[HelicalBaseModel], train_data: AnnData, eval_data: AnnData) -> None:

        self.train_data = train_data
        self.train_embeddings = {}

        self.eval_data = eval_data
        self.eval_embeddings = {}
        
        for model in models:
            LOGGER.info(f"Getting training embeddings with {model.__class__.__name__}")
            dataset = model.process_data(train_data)
            self.train_embeddings.update({model.__class__.__name__: model.get_embeddings(dataset)})

            LOGGER.info(f"Processing evaluation embeddings with {model.__class__.__name__}")
            dataset = model.process_data(eval_data)
            self.eval_embeddings.update({model.__class__.__name__: model.get_embeddings(dataset)})
    
    def classification(self, classification_model: BaseTaskModel) -> dict[str, dict[str, float]]:
        """Classification task training and evaluating a type of classification_model.
        This is done for each entry in the train_embeddings and eval_embeddings dictionaries.
        
        If for example train_embeddings = {"Geneformer": geneformer_embeddings_train, "scGPT": scgpt_embeddings_train}
        and eval_embeddings = {"Geneformer": geneformer_embeddings_eval, "scGPT": scgpt_embeddings_eval} and the classification_model is a NeuralNetwork,
        then two classification NNs will be trained on geneformer_embeddings_train and scgpt_embeddings_train.
        
        The predictions of these models will be evaluated on geneformer_embeddings_eval and scgpt_embeddings_eval.
    
        Parameters
        ----------
        classification_model : BaseTaskModel
            The type of classification model to use.

        Returns
        -------
        A dictionary containing the evaluations for each HelicalBaseModel provided in the initialization.

        """
        train_labels = np.array(self.train_data.obs["cell_type"].tolist())
        eval_labels = np.array(self.eval_data.obs["cell_type"].tolist())

        self.classifier = Classifier(train_labels, classification_model)
        
        LOGGER.info(f"Training classification models heads of type '{classification_model.__class__.__name__}' on training embeddings.")
        self.classifier.train_task_models(x_train_embeddings = self.train_embeddings)
        
        LOGGER.info(f"Classification prediction with model heads of type '{classification_model.__class__.__name__}' on evaluation embeddings.")
        predictions = self.classifier.get_predictions(x_eval_embeddings = self.eval_embeddings)

        LOGGER.info(f"Evaluating predictions of models '{classification_model.__class__.__name__}'.")
        evaluations = self.classifier.get_evaluations(predictions, eval_labels)
        return evaluations

            