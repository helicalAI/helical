import os
import json
from omegaconf import OmegaConf
from helical.constants.paths import CACHE_DIR_HELICAL
from typing import Literal

class TranscriptFormerConfig:
    def __init__(self, 
        model_name: Literal["tf_sapiens", "tf_metazoa", "tf_exemplar"] = "tf_sapiens",
        inference_config: dict = {
            "_target_": "helical.models.transcriptformer.data.dataclasses.InferenceConfig",
            "batch_size": 8,
            "output_keys": ["embeddings"],
            "obs_keys": ["all"],
                "data_files": [None],
                "output_path": "./inference_results",
                "load_checkpoint": None,
                "pretrained_embedding": None,
                "precision": "16-mixed"
            },
            data_config: dict = {
                "_target_": "helical.models.transcriptformer.data.dataclasses.DataConfig",
                "gene_col_name": "ensembl_id",
                "clip_counts": 30,
                "filter_to_vocabs": True,
                "filter_outliers": 0.0,
                "normalize_to_scale": 0,
                "sort_genes": False,
                "randomize_genes": False,
                "min_expressed_genes": 0
            }
    ):
        config = OmegaConf.create({
            "model": {
                "inference_config": inference_config,
                "data_config": data_config
            }
        })

        if model_name not in ["tf_sapiens", "tf_metazoa", "tf_exemplar"]:
            raise ValueError(f"Model name {model_name} not supported. Only tf_sapiens, tf_metazoa, and tf_exemplar are supported.")

        cache_config_path = os.path.join(CACHE_DIR_HELICAL, "transcriptformer", model_name, "config.json")
        with open(cache_config_path) as f:
            cache_config_dict = json.load(f)
        cache_config = OmegaConf.create(cache_config_dict)

        # Merge the cache config with the main config
        self.config = OmegaConf.merge(cache_config, config)

        self.config.model.inference_config.load_checkpoint = os.path.join(CACHE_DIR_HELICAL, "transcriptformer", model_name, "model_weights.pt")

        # Set the auxiliary vocabulary paths to None 
        self.config.model.data_config.aux_vocab_path = os.path.join(CACHE_DIR_HELICAL, "transcriptformer", model_name, "vocabs")
        self.config.model.data_config.aux_cols = "assay"
        self.config.model.data_config.esm2_mappings_path = os.path.join(CACHE_DIR_HELICAL, "transcriptformer", model_name, "vocabs")

        self.list_of_files_to_download = [
            f"transcriptformer/{model_name}/config.json",
            f"transcriptformer/{model_name}/model_weights.pt",
            f"transcriptformer/{model_name}/vocabs/assay_vocab.json",
            f"transcriptformer/{model_name}/vocabs/homo_sapiens_gene.h5",
        ]
