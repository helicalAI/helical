from helical.models.classification.neural_network import NeuralNetwork
from helical.models.classification.classifier import Classifier 
from helical.models.scgpt.model import scGPT
import pytest 
import anndata as ad
import numpy as np

def test_load_model__ok_nn_head():
    """
    Happy path for Classifier.load_model.
    The load_model expects a base model and a trained task model.
    - The base model needs to follow the BaseModelProtocol.
        In this case, the base model is an instance of scGPT. As all HelicalBaseFoundationModels have a process_data and 'get_embeddings' method, this is fine.
    - The trained task model needs to follow the ClassificationModelProtocol.
        In this case, the trained task model is an instance of NeuralNetwork. As the NeuralNetwork class inherits from a BaseTaskModel which requires the 
        implementation of the 'predict' method.
    """
    scgpt = scGPT()
    saved_nn_head = NeuralNetwork().load('ci/tests/data/valid_nn_head/my_model.h5', 'ci/tests/data/valid_nn_head/classes.npy')
    name = "scGPT with saved NN"
    scgpt_loaded_nn = Classifier().load_model(scgpt, saved_nn_head, name)    
    assert isinstance(scgpt_loaded_nn, Classifier)
    assert scgpt_loaded_nn.base_model == scgpt
    assert scgpt_loaded_nn.trained_task_model == saved_nn_head
    assert scgpt_loaded_nn.name == name
    assert scgpt_loaded_nn.gene_names == "index"


def test_load_model__ok_custom_head():
    """
    Happy path for Classifier.load_model.
    If a user decides to use his own model head, he just needs to make sure it is following the ClassificationModelProtocol.
    """
    scgpt = scGPT()

    class CustomModel:
        def __init__(self):
            pass
        def load(self):
            pass
        def predict(self):
            pass

    custom_head = CustomModel()
    name = "scGPT with custom NN"
    scgpt_loaded_nn = Classifier().load_model(scgpt, custom_head, name)    
    assert isinstance(scgpt_loaded_nn, Classifier)
    assert scgpt_loaded_nn.base_model == scgpt
    assert scgpt_loaded_nn.trained_task_model == custom_head
    assert scgpt_loaded_nn.name == name
    assert scgpt_loaded_nn.gene_names == "index"

def test_load_model__nok_custom_head():
    """
    Unappy path for Classifier.load_model.
    If a user decides to use his own model head, he just needs to make sure it is following the ClassificationModelProtocol.
    If this is not the case, raise an error.
    """
    scgpt = scGPT()

    class CustomModelHead:
        def __init__(self):
            pass
        # missing predict method

    custom_head = CustomModelHead()
    name = "scGPT with custom NN"
    with pytest.raises(TypeError):
        Classifier().load_model(scgpt, custom_head, name)    

def test_load_model__ok_custom_base():
    """
    Happy path for Classifier.load_model.
    If a user decides to use his own model head, he just needs to make sure it is following the ClassificationModelProtocol.
    """

    class CustomBaseModel:
        def __init__(self):
            pass
        def get_embeddings(self):
            pass
        def process_data(self):
            pass
        def train(self):
            pass

    custom_base = CustomBaseModel()

    saved_nn_head = NeuralNetwork().load('ci/tests/data/valid_nn_head/my_model.h5', 'ci/tests/data/valid_nn_head/classes.npy')
    name = "custom base model with NN"
    custom_base_classifier = Classifier().load_model(custom_base, saved_nn_head, name) 
    assert isinstance(custom_base_classifier, Classifier)
    assert custom_base_classifier.base_model == custom_base
    assert custom_base_classifier.trained_task_model == saved_nn_head
    assert custom_base_classifier.name == name
    assert custom_base_classifier.gene_names == "index"

def test_load_model__nok_custom_base():
    """
    Happy path for Classifier.load_model.
    If a user decides to use his own model head, he just needs to make sure it is following the ClassificationModelProtocol.
    """

    class CustomBaseModel:
        def __init__(self):
            pass
        # missing get_embeddings and process_data method

    custom_base = CustomBaseModel()

    saved_nn_head = NeuralNetwork().load('ci/tests/data/valid_nn_head/my_model.h5', 'ci/tests/data/valid_nn_head/classes.npy')
    name = "custom base model with NN"
    with pytest.raises(TypeError):
        Classifier().load_model(custom_base, saved_nn_head, name)    
    
def test_train_classifier_head(mocker):
    """
    Happy path for Classifier.train_classifier_head.
    """
    scgpt = scGPT()
    data = ad.read_h5ad("ci/tests/data/cell_type_sample.h5ad")
    head = NeuralNetwork()

    # Mocking the get_embeddings method by returning a zero matrix
    mocker.patch.object(scgpt, 'get_embeddings')
    scgpt.get_embeddings.return_value = np.zeros((data.shape[0], 10))
    
    # Mocking the training process by returning the untrained head
    mocker.patch.object(head, 'train')
    head.train.return_value = head

    scgpt_nn_c = Classifier().train_classifier_head(data, scgpt, head, labels_column_name = "cell_type")
    assert isinstance(scgpt_nn_c, Classifier)
    assert scgpt_nn_c.base_model == scgpt
    assert scgpt_nn_c.name == f"{scgpt.__class__.__name__} with {head.__class__.__name__}"
    assert scgpt_nn_c.trained_task_model == head 
    assert scgpt_nn_c.gene_names == "index"      


def test_check_validity_for_training_wrong_column_label_name():
    """
    Checks if a TypeError is raised if the labels_column_name is not present in the obs attribute of the AnnData object.
    """
    scgpt = scGPT()
    data = ad.read_h5ad("ci/tests/data/cell_type_sample.h5ad")
    with pytest.raises(TypeError):
        Classifier()._check_validity_for_training(data, labels_column_name = "wrong_name", base_model = scgpt) 


def test_check_validity_for_training_wrong_base_model():
    """
    Checks if a TypeError is raised if the base model is not an instance of a class implementing 'BaseModelProtocol'.
    """
    class CustomBaseModel:
        def __init__(self):
            pass
        # missing get_embeddings and process_data method

    custom_base = CustomBaseModel()
    data = ad.read_h5ad("ci/tests/data/cell_type_sample.h5ad")
    with pytest.raises(TypeError):
        Classifier()._check_validity_for_training(data, labels_column_name = "cell_type", base_model = custom_base) 


def test_get_predictions_w_no_base_model(mocker):
    """
    Happy path for Classifier.get_predictions having no base model.
    This is the case where the trained_task_model is a standalone classifier (handle anndata as input and predict classes as output).
    """

    base_model = None
    data = ad.read_h5ad("ci/tests/data/cell_type_sample.h5ad")
    standalone = NeuralNetwork().load('ci/tests/data/valid_nn_head/my_model.h5', 'ci/tests/data/valid_nn_head/classes.npy')

    # Mocking the prediction process by returning the untrained head
    mocker.patch.object(standalone, 'predict')
    standalone.predict.return_value = np.array(['type1', 'type2'] * (data.shape[0] // 2))

    name = "None as base with custom class as standalone classifier"
    c = Classifier().load_model(base_model, standalone, name)    
    c.get_predictions(data)
    assert isinstance(c, Classifier)
    assert c.base_model == None
    assert c.name == name
    assert c.trained_task_model == standalone   
    assert c.gene_names == "index"    