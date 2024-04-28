from helical.models.uce.model import UCE
import anndata as ad

uce = UCE("./data/uce/4layer_model.torch")
ann_data = ad.read_h5ad("./data/10k_pbmcs_proc.h5ad")
data_loader = uce.process_data(ann_data[:100], "./data/config.json")
embeddings = uce.get_embeddings(data_loader)

print(embeddings.shape)