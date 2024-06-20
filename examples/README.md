# Helical Examples

The `examples` folder contains a `run_models` and a `notebooks` folder.

## Run Models
We show that each supported Helical model can be included in a uniform manner.
```
from helical.models.scgpt.model import scGPT, scGPTConfig

scgpt_config = scGPTConfig(batch_size=10)
scgpt = scGPT(configurer = scgpt_config)
```
For specific configurations, such as `batch_size`, a model can be provided with its own configuration (`scGPTConfig` in this case).
Processing the data and getting the embeddings is uniform across models too:
```
adata = ad.read_h5ad("./10k_pbmcs_proc.h5ad")
data = scgpt.process_data(adata)
embeddings = scgpt.get_embeddings(data)
```

## Notebooks
To do something useful with these embeddings, we provide a number of use case examples in the `notebooks` folder.

One such example is the [Cell-Type-Annotation](./notebooks/Cell-Type-Annotation.ipynb) notebook. An scGPT model is used to get embeddings of a gene expression profile which are then used as inputs to a smaller neural network, predicting the cell type.

That notebook explains the procedure step-by-step in much detail. A more modular and automated procedure can be found in the [benchmark.py](benchmark.py) script.

## Benchmark
To compare different models against each other, we built a benchmarking infrastructre.
A simple example is shown below:
```
from helical.benchmark.benchmark import Benchmark
from helical.models.scgpt.model import scGPT
from helical.classification.neural_network import NeuralNetwork
from helical.classification.classifier import Classifier
import anndata as ad
from omegaconf import DictConfig
import hydra

@hydra.main(version_base=None, config_path=".", config_name="config")
def benchmark(cfg: DictConfig) -> None:
    scgpt = scGPT()

    data = ad.read_h5ad("./examples/10k_pbmcs_proc.h5ad")
    train_data = data[:3000]
    eval_data = data[3000:3500]
    
    scgpt_nn_c = Classifier().train_classifier_head(
        train_anndata = train_data, 
        base_model = scgpt, 
        head = NeuralNetwork(**cfg["neural_network"]),
        labels_column_name = "cell_type", 
        test_size = 0.2, 
        random_state = 42)        

    bench = Benchmark()
    evaluations = bench.evaluate_classification([scgpt_nn_c], eval_data, "cell_type")
    print(evaluations)

if __name__ == "__main__":
    benchmark()
```
We use [Hydra](https://hydra.cc/) to pass configurations to our models. In this example, a neural network is used as a classification `head` but other models (such as SVM) can be found in the [classification folder](../helical/classification/). It is also possible to use your own head, your own base model or even your own standalone classifier. This is shown in the picture below: 

TODO include


