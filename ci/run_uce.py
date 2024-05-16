from helical.models.uce.model import UCE, UCEConfig
from helical.services.downloader import Downloader
import anndata as ad
from pathlib import Path
import os




def main():
    downloader = Downloader()
    downloader.display = False
    downloader.download_via_name("uce/4layer_model.torch")
    downloader.download_via_name("uce/all_tokens.torch")
    downloader.download_via_name("uce/species_chrom.csv")
    downloader.download_via_name("uce/species_offsets.pkl")
    downloader.download_via_name("uce/protein_embeddings/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt")
    downloader.download_via_name("uce/protein_embeddings/Macaca_fascicularis.Macaca_fascicularis_6.0.gene_symbol_to_embedding_ESM2.pt")
    uce_model_dir = Path(os.path.join(downloader.CACHE_DIR_HELICAL,'uce'))


    #UCE
    print(f"Loading UCE",flush=True)
    model_config=UCEConfig(batch_size=1)
    uce = UCE(model_dir=uce_model_dir,model_config=model_config)
    print(f"Loading UCE Done",flush=True)
    ann_data = ad.read_h5ad("10k_pbmcs_proc.h5ad")
    print(f"Processing Data",flush=True)
    data_loader = uce.process_data(ann_data[:5])
    print(f"Processing Data Done",flush=True)
    embeddings = uce.get_embeddings(data_loader)
    print(f"UCE embeddings shape: {embeddings.shape}",flush=True)

    return True



if __name__ == "__main__":
    main()






