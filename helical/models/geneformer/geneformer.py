from helical.models.helical import HelicalBaseModel
from helical.constants.enums import LoggingType, LoggingLevel
import logging
import os
from pathlib import Path
import numpy as np
from anndata import AnnData
import pickle
from torch.utils.data import DataLoader
from transformers import BertForMaskedLM
from helical.models.geneformer.geneformer_utils import get_embs,quant_layers
from helical.models.geneformer.geneformer_tokenizer import TranscriptomeTokenizer
from datasets import Dataset

class Geneformer(HelicalBaseModel):
    
    def __init__(self, logging_type = LoggingType.CONSOLE, level = LoggingLevel.INFO) -> None:
        super().__init__(logging_type, level)
        self.log = logging.getLogger("Geneformer-Model")

    def get_model(self, model_config, data_config, files_config, accelerator=None):
        self.model_config = model_config
        self.data_config = data_config
        self.files_config = files_config
        self.device = model_config['device']

        self.model =  BertForMaskedLM.from_pretrained(model_config['model_directory'], output_hidden_states=True, output_attentions=False)
        self.model.eval()#.to("cuda:0")
        self.model = self.model.to(self.device)

        self.layer_to_quant = quant_layers(self.model) + model_config['emb_layer']
        self.accelerator = accelerator
        self.emb_mode = model_config["emb_mode"]
        self.forward_batch_size = model_config["batch_size"]
        
        if accelerator is not None:
           self.model = accelerator.prepare(self.model)
        
        # load token dictionary (Ensembl IDs:token)
        with open(files_config['token_dictionary_file'], "rb") as f:
            self.gene_token_dict = pickle.load(f)

        self.token_gene_dict = {v: k for k, v in self.gene_token_dict.items()}
        self.pad_token_id = self.gene_token_dict.get("<pad>")

        self.tk = TranscriptomeTokenizer({"cell_type": "cell_type"}, nproc=4,gene_median_file=files_config['gene_median_file'], token_dictionary_file=files_config['token_dictionary_file'],)
        return self.model

    def process_data(self, data: AnnData, species="human") -> DataLoader:    
        tokenized_cells, cell_metadata =  self.tk.tokenize_anndata(data)
        tokenized_dataset = self.tk.create_dataset(tokenized_cells, cell_metadata, use_generator=False)
        output_dir = "/tmp"
        output_prefix = "tmp"
        output_path = (Path(output_dir) / output_prefix).with_suffix(".dataset")
        tokenized_dataset.save_to_disk(output_path)
        return tokenized_dataset
        
    def run(self, dataset: Dataset) -> np.array:
        self.log.info(f"Inference started")
        embeddings = get_embs(
            self.model,
            dataset,
            self.emb_mode,
            self.layer_to_quant,
            self.pad_token_id,
            self.forward_batch_size,
            self.device
        )
        return embeddings.cpu().detach().numpy()

    def get_embeddings(self, dataset:Dataset) -> np.array:
        embeddings = self.run(dataset)
        return embeddings