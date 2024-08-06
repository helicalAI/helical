from helical.models.scgpt.model import scGPT, scGPTConfig
import anndata as ad
from helical.benchmark.sccaf import SCCAF_assessment
from helical.models.classification.classifier import Classifier
from helical.benchmark.benchmark import evaluate_classification
from numpy import ndarray
from anndata import AnnData
from helical.models.classification.svm import SupportVectorMachine

scgpt_config = scGPTConfig(batch_size=10, device="cuda")
scgpt = scGPT(configurer = scgpt_config)
adata = ad.read_h5ad("./10k_pbmcs_proc.h5ad")

class DummyIdentity():
    def __init__(self):
        pass
    def process_data(self, anndata, gene_names) -> AnnData:
        return anndata
    def get_embeddings(self, anndata) -> ndarray:
        return anndata.X

before_c = Classifier().train_classifier_head(train_anndata = adata[:1000], base_model = DummyIdentity(), head = SupportVectorMachine())
after_c = Classifier().train_classifier_head(train_anndata = adata[:1000], base_model = scgpt, head = SupportVectorMachine())
evaluations = evaluate_classification(models = [before_c, after_c], eval_anndata = adata[1000:1500], labels_column_name = "cell_type")
print(evaluations)

