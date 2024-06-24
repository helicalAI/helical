from helical.benchmark.benchmark import Benchmark
from helical.models.geneformer.model import Geneformer
from helical.models.scgpt.model import scGPT
from helical.models.uce.model import UCE
from helical.classification.neural_network import NeuralNetwork
from helical.classification.svm import SupportVectorMachine as SVM
from helical.classification.classifier import Classifier
import anndata as ad
from omegaconf import DictConfig
import hydra
import json

@hydra.main(version_base=None, config_path=".", config_name="config")
def benchmark(cfg: DictConfig) -> None:
    # geneformer = Geneformer()
    # scgpt = scGPT()
    uce = UCE()

    data = ad.read_h5ad("./examples/10k_pbmcs_proc.h5ad")
    train_data = data[:20]
    eval_data = data[20:25]

    # saved_nn_head = NeuralNetwork().load('my_model.h5', 'classes.npy')
    # scgpt_loaded_nn = Classifier().load_model(scgpt, saved_nn_head, "scgpt with saved NN")    
    
    # saved_svm_head = SVM().load('my_svm.pkl')
    # scgpt_loaded_svm = Classifier().load_model(scgpt, saved_svm_head, "scgpt with saved SVM")           

    # uce_c = Classifier().train_classifier_head(train_data, uce, NeuralNetwork(**cfg["neural_network"]))
    # scgpt_nn_c = Classifier().train_classifier_head(train_data, scgpt, NeuralNetwork(**cfg["neural_network"]))           

    saved_nn_head = NeuralNetwork().load('nn.h5', 'classes.npy')
    uce_c = Classifier().load_model(uce, saved_nn_head, "UCE with saved NN")   

    bench = Benchmark()
    evaluations = bench.evaluate_classification([uce_c], eval_data, "cell_type")
    
    # Serializing json
    json_object = json.dumps(evaluations, indent=4)
    
    # Writing to sample.json
    with open("helical/benchmark/results.json", "w") as outfile:
        outfile.write(json_object)

if __name__ == "__main__":
    benchmark()