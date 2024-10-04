from helical.utils.downloader import Downloader
from pathlib import Path
import logging
LOGGER = logging.getLogger(__name__)

def download_geneformer_models():
    downloader = Downloader()
    versions = ['v1', 'v2']

    # We can decide to download more models by simply adding the model names from the full list as reported in geneformer_config.py
    version_models_dict = {
        "v1": ["gf-12L-30M-i2048"],
        "v2": ["gf-12L-95M-i4096"]
    }
    
    for version in versions:
        # Download common files for each version
        common_files = [
            f"geneformer/{version}/gene_median_dictionary.pkl",
            f"geneformer/{version}/token_dictionary.pkl",
            f"geneformer/{version}/ensembl_mapping_dict.pkl",
        ]
        for file in common_files:
            downloader.download_via_name(file)
        
        # Get all model directories
        model_dirs = version_models_dict[version]
        
        for model_name in model_dirs:
            model_files = [
                f"geneformer/{version}/{model_name}/config.json",
                f"geneformer/{version}/{model_name}/training_args.bin",
            ]
            
            # Add version-specific files
            if version == 'v2':
                model_files.extend([
                    f"geneformer/{version}/{model_name}/generation_config.json",
                    f"geneformer/{version}/{model_name}/model.safetensors",
                ])
            else:
                model_files.append(f"geneformer/{version}/{model_name}/pytorch_model.bin")
            
            # Download all files for the current model
            for file in model_files:
                downloader.download_via_name(file)
    
    LOGGER.info("All Geneformer models and files have been downloaded.")

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

    download_geneformer_models()

    downloader.download_via_name("hyena_dna/hyenadna-tiny-1k-seqlen.ckpt")
    downloader.download_via_name("hyena_dna/hyenadna-tiny-1k-seqlen-d256.ckpt")

    downloader.download_via_link(Path("yolksac_human.h5ad"), "https://huggingface.co/datasets/helical-ai/yolksac_human/resolve/main/data/17_04_24_YolkSacRaw_F158_WE_annots.h5ad?download=true")
    return True

if __name__ == "__main__":
    main()
