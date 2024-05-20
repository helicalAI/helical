from helical.services.downloader import Downloader
from pathlib import Path

def main():
    downloader = Downloader()
    downloader.display = False

    downloader.download_via_name("uce/4layer_model.torch")
    downloader.download_via_name("uce/all_tokens.torch")
    downloader.download_via_name("uce/species_chrom.csv")
    downloader.download_via_name("uce/species_offsets.pkl")
    downloader.download_via_name("uce/protein_embeddings/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt")
    downloader.download_via_name("uce/protein_embeddings/Macaca_fascicularis.Macaca_fascicularis_6.0.gene_symbol_to_embedding_ESM2.pt")
    
    downloader.download_via_name("scgpt/scGPT_CP/vocab.json")
    downloader.download_via_name("scgpt/scGPT_CP/best_model.pt")

    downloader.download_via_name("geneformer/gene_median_dictionary.pkl")
    downloader.download_via_name("geneformer/human_gene_to_ensemble_id.pkl")
    downloader.download_via_name("geneformer/token_dictionary.pkl")
    downloader.download_via_name("geneformer/geneformer-12L-30M/config.json")
    downloader.download_via_name("geneformer/geneformer-12L-30M/pytorch_model.bin")
    downloader.download_via_name("geneformer/geneformer-12L-30M/training_args.bin")

    downloader.download_via_name("hyena_dna/hyenadna-tiny-1k-seqlen.ckpt")
    downloader.download_via_name("hyena_dna/hyenadna-tiny-1k-seqlen-d256.ckpt")

    downloader.download_via_link(Path("./10k_pbmcs_proc.h5ad"), "https://helicalpackage.blob.core.windows.net/helicalpackage/data/10k_pbmcs_proc.h5ad")
    return True

if __name__ == "__main__":
    main()
