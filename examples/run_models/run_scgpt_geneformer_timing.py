from helical.models.geneformer import Geneformer, GeneformerConfig
from helical.models.scgpt import scGPT, scGPTConfig
import hydra
from omegaconf import DictConfig
import anndata as ad
from datasets import load_dataset
from helical.utils import get_anndata_from_hf_dataset
import time
import numpy as np


def run_geneformer(cfg: DictConfig, ann_data: ad.AnnData):
    geneformer_config = GeneformerConfig(**cfg)
    geneformer = Geneformer(configurer=geneformer_config)

    num = geneformer.model.num_parameters(only_trainable=False)
    print(f"Number of parameters: {num:_}".replace("_", " "))

    # either load via huggingface
    # hf_dataset = load_dataset(
    #     "helical-ai/yolksac_human",
    #     split="train[:5%]",
    #     trust_remote_code=True,
    #     download_mode="reuse_cache_if_exists",
    # )
    # ann_data = get_anndata_from_hf_dataset(hf_dataset)

    # or load directly

    start_time = time.time()
    dataset = geneformer.process_data(ann_data[:])
    end_time = time.time()
    print(f"Data processing time: {end_time - start_time:.4f} seconds")
    start_time = time.time()
    embeddings, input_genes = geneformer.get_embeddings(dataset, output_genes=True)
    end_time = time.time()
    print(
        f"Get embeddings time (output_genes=True): {end_time - start_time:.4f} seconds"
    )
    # np.save("geneformer_embeddings.npy", embeddings)
    print(embeddings)
    # print(embeddings)
    # embeddings, attention_weights = geneformer.get_embeddings(
    #     dataset, output_attentions=True
    # )


def run_scgpt(cfg: DictConfig, ann_data: ad.AnnData):
    scgpt_config = scGPTConfig(**cfg)
    scgpt = scGPT(configurer=scgpt_config)

    # either load via huggingface
    # hf_dataset = load_dataset(
    #     "helical-ai/yolksac_human",
    #     split="train[:5%]",
    #     trust_remote_code=True,
    #     download_mode="reuse_cache_if_exists",
    # )
    # ann_data = get_anndata_from_hf_dataset(hf_dataset)

    # or load directly
    ann_data = ad.read_h5ad("./yolksac_human.h5ad")
    start_time = time.time()
    data = scgpt.process_data(ann_data[:])
    end_time = time.time()
    print(f"Data processing time: {end_time - start_time:.4f} seconds")
    start_time = time.time()
    embeddings, input_genes = scgpt.get_embeddings(data, output_genes=True)

    end_time = time.time()
    print(
        f"Get embeddings time (output_genes=True): {end_time - start_time:.4f} seconds"
    )
    # np.save("scgpt_embeddings.npy", embeddings)
    print(embeddings)
    # embeddings, attn_weights = scgpt.get_embeddings(data, output_attentions=True)


if __name__ == "__main__":

    ## Load Specific Hydra Config
    ann_data = ad.read_h5ad("./yolksac_human.h5ad")
    hydra.initialize(config_path="configs", job_name="geneformer_config")
    cfg_geneformer = hydra.compose(
        config_name="geneformer_config", overrides=["device=cuda"]
    )
    start_time = time.time()
    run_geneformer(cfg_geneformer, ann_data)
    end_time = time.time()
    print(f"Geneformer run time: {end_time - start_time:.4f} seconds")
    print("\n\n\n")
    # hydra.initialize(config_path="configs", job_name="scgpt_config")
    cfg_scgpt = hydra.compose(config_name="scgpt_config", overrides=["device=cuda"])
    start_time = time.time()
    run_scgpt(cfg_scgpt, ann_data)
    end_time = time.time()
    print(f"scGPT run time: {end_time - start_time:.4f} seconds")
