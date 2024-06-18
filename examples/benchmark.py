from helical.benchmark.benchmark import Benchmark
from helical.models.geneformer.model import Geneformer
from helical.models.scgpt.model import scGPT
from helical.benchmark.task_models.neural_network import NeuralNetwork
from helical.benchmark.task_models.svm import SupportVectorMachine as SVM
import anndata as ad
from omegaconf import DictConfig
import hydra
from helical.benchmark.tasks.classifier import Classifier

@hydra.main(version_base=None, config_path=".", config_name="config")
def benchmark(cfg: DictConfig) -> None:
    geneformer = Geneformer()
    scgpt = scGPT()

    data = ad.read_h5ad("./examples/10k_pbmcs_proc.h5ad")
    train_data = data[:30]
    eval_data = data[30:35]

    # head = tf.keras.models.load_model('my_model.h5')
    # scgpt_loaded_c = Classifier().load_custom_model(scgpt, head, "my_model")    
    gene_c = Classifier().train_classifier(geneformer, train_data, NeuralNetwork(**cfg["neural_network"]))
    scgpt_nn_c = Classifier().train_classifier(scgpt, train_data, NeuralNetwork(**cfg["neural_network"]))           
    scgpt_svm_c = Classifier().train_classifier(scgpt, train_data, SVM(**cfg["svm"]))           

    bench = Benchmark()
    evaluations = bench.evaluate_classification([gene_c, scgpt_nn_c, scgpt_svm_c], eval_data)
    print(evaluations)

if __name__ == "__main__":
    benchmark()