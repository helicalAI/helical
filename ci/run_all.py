# from helical.models.scgpt.model import scGPT, scGPTConfig
from helical.models.geneformer.model import Geneformer, GeneformerConfig
from helical.models.scgpt.model import scGPTConfig, scGPT
from helical.models.uce.model import UCE, UCEConfig
from helical.services.downloader import Downloader
import anndata as ad
from pathlib import Path
import os

downloader = Downloader()
downloader.display = False
downloader.download_via_link(Path("./10k_pbmcs_proc.h5ad"), "https://helicalpackage.blob.core.windows.net/helicalpackage/data/10k_pbmcs_proc.h5ad")

downloader.download_via_name("scgpt/scGPT_CP/vocab.json")
downloader.download_via_name("scgpt/scGPT_CP/best_model.pt")
scgpt_model_dir = Path(os.path.join(downloader.CACHE_DIR_HELICAL,'scgpt/scGPT_CP'))

downloader.download_via_name("geneformer/gene_median_dictionary.pkl")
downloader.download_via_name("geneformer/human_gene_to_ensemble_id.pkl")
downloader.download_via_name("geneformer/token_dictionary.pkl")
downloader.download_via_name("geneformer/geneformer-12L-30M/config.json")
downloader.download_via_name("geneformer/geneformer-12L-30M/pytorch_model.bin")
downloader.download_via_name("geneformer/geneformer-12L-30M/training_args.bin")
geneformer_model_dir = Path(os.path.join(downloader.CACHE_DIR_HELICAL,'geneformer'))

downloader.download_via_name("uce/4layer_model.torch")
downloader.download_via_name("uce/all_tokens.torch")
downloader.download_via_name("uce/species_chrom.csv")
downloader.download_via_name("uce/species_offsets.pkl")
downloader.download_via_name("uce/protein_embeddings/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt")
downloader.download_via_name("uce/protein_embeddings/Macaca_fascicularis.Macaca_fascicularis_6.0.gene_symbol_to_embedding_ESM2.pt")
uce_model_dir = Path(os.path.join(downloader.CACHE_DIR_HELICAL,'uce'))

model_config = scGPTConfig(batch_size=10)
scgpt = scGPT(model_dir = scgpt_model_dir, model_config = model_config)
adata = ad.read_h5ad("10k_pbmcs_proc.h5ad")
data = scgpt.process_data(adata[:10])
embeddings = scgpt.get_embeddings(data)
print(f"scGPT embeddings shape: {embeddings.shape}")

# Geneformer
model_config=GeneformerConfig(batch_size=10)
geneformer = Geneformer(model_dir = geneformer_model_dir, model_config=model_config)
ann_data = ad.read_h5ad("10k_pbmcs_proc.h5ad")
dataset = geneformer.process_data(ann_data[:5])
embeddings = geneformer.get_embeddings(dataset)
print(f"Geneformer embeddings shape: {embeddings.shape}")

# UCE
model_config=UCEConfig(batch_size=10)
uce = UCE(model_config=model_config)
ann_data = ad.read_h5ad("10k_pbmcs_proc.h5ad")
data_loader = uce.process_data(ann_data[:5])
embeddings = uce.get_embeddings(data_loader)
print(f"UCE embeddings shape: {embeddings.shape}")
