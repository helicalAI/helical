from helical.benchmark import Benchmark
from helical.models.geneformer.model import Geneformer
from helical.models.scgpt.model import scGPT
import anndata as ad
geneformer = Geneformer()
scgpt = scGPT()
data = ad.read_h5ad("./examples/10k_pbmcs_proc.h5ad")

test = Benchmark([geneformer, scgpt], data[:10])