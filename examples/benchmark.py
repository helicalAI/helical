from helical.benchmark.benchmark import Benchmark
from helical.models.geneformer.model import Geneformer
from helical.models.scgpt.model import scGPT
from helical.benchmark.classification.neural_network import NeuralNetwork
from helical.benchmark.classification.svm import SupportVectorMachine as SVM
import anndata as ad
from omegaconf import DictConfig
import hydra
from helical.benchmark.classification.classifier import Classifier

@hydra.main(version_base=None, config_path=".", config_name="config")
def benchmark(cfg: DictConfig) -> None:
    geneformer = Geneformer()
    scgpt = scGPT()

    data = ad.read_h5ad("./examples/10k_pbmcs_proc.h5ad")
    train_data = data[:30]
    eval_data = data[30:35]

    # saved_nn_head = NeuralNetwork().load('my_model.h5', 'classes.npy')
    # scgpt_loaded_nn = Classifier().load_model(scgpt, saved_nn_head, "scgpt with saved NN")    
    
    # saved_svm_head = SVM().load('my_svm.pkl')
    # scgpt_loaded_svm = Classifier().load_model(scgpt, saved_svm_head, "scgpt with saved SVM")           

    gene_c = Classifier().train_classifier_head(train_data, geneformer, NeuralNetwork(**cfg["neural_network"]), 0.2, 42)
    scgpt_nn_c = Classifier().train_classifier_head(train_data, scgpt, NeuralNetwork(**cfg["neural_network"]))           

    bench = Benchmark()
    evaluations = bench.evaluate_classification([gene_c, scgpt_nn_c], eval_data, "cell_type")
    print(evaluations)

if __name__ == "__main__":
    benchmark()