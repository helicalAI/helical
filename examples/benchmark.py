from helical.benchmark.benchmark import Benchmark
from helical.models.geneformer.model import Geneformer
from helical.models.scgpt.model import scGPT
from helical.benchmark.task_models.neural_network import NeuralNetwork

import anndata as ad
import numpy as np

geneformer = Geneformer()
scgpt = scGPT()

data = ad.read_h5ad("./examples/10k_pbmcs_proc.h5ad")
train_data = data[:10]
eval_data = data[10:20]
bench = Benchmark([geneformer, scgpt], train_data, eval_data)

train_labels = np.array(train_data.obs["cell_type"].tolist())
eval_labels = np.array(eval_data.obs["cell_type"].tolist())
evaluations = bench.classification(NeuralNetwork((512,), 3), train_labels, eval_labels)

print(evaluations)