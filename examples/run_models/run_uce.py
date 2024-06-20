from helical.models.uce.model import UCE, UCEConfig
import anndata as ad

configurer=UCEConfig(batch_size=10)
uce = UCE(configurer=configurer)
ann_data = ad.read_h5ad("./10k_pbmcs_proc.h5ad")
data_loader = uce.process_data(ann_data[:10])
embeddings = uce.get_embeddings(data_loader)

print(embeddings.shape)