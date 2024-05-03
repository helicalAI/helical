from helical.models.scgpt.model import scGPT, scGPTConfig
from helical.models.geneformer.model import Geneformer, GeneformerConfig
from helical.models.uce.model import UCE, UCEConfig
from helical.services.downloader import Downloader
import anndata as ad
from pathlib import Path

downloader = Downloader()
downloader.download_via_link(Path("./10k_pbmcs_proc.h5ad"), "https://helicalpublicdata.blob.core.windows.net/helicalpackage/data/10k_pbmcs_proc.h5ad")

# SCGPT
model_config = scGPTConfig(batch_size=10)
scgpt = scGPT(model_config=model_config)
adata = ad.read_h5ad("10k_pbmcs_proc.h5ad")
scgpt.process_data(adata[:100])
embeddings = scgpt.get_embeddings()
print(f"scGPT embeddings shape: {embeddings.shape}")

# Geneformer
model_config=GeneformerConfig(batch_size=10)
geneformer = Geneformer(model_config=model_config)
ann_data = ad.read_h5ad("10k_pbmcs_proc.h5ad")
dataset = geneformer.process_data(ann_data[:100])
embeddings = geneformer.get_embeddings(dataset)
print(f"Geneformer embeddings shape: {embeddings.shape}")

# UCE
model_config=UCEConfig(batch_size=10)
uce = UCE(model_config=model_config)
ann_data = ad.read_h5ad("./data/10k_pbmcs_proc.h5ad")
data_loader = uce.process_data(ann_data[:100])
embeddings = uce.get_embeddings(data_loader)
print(f"UCE embeddings shape: {embeddings.shape}")
