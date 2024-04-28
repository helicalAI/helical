from helical.models.helical import HelicalBaseModel
from helical.constants.enums import LoggingType, LoggingLevel
import logging
from pathlib import Path
import numpy as np
from anndata import AnnData
import pickle
from torch.utils.data import DataLoader
from transformers import BertForMaskedLM
from helical.models.geneformer.geneformer_utils import get_embs,quant_layers
from helical.models.geneformer.geneformer_tokenizer import TranscriptomeTokenizer
from datasets import Dataset
import json
from accelerate import Accelerator
from typing import Union
import pickle as pkl

class Geneformer(HelicalBaseModel):
    
    def __init__(self, 
                 model_dir,
                 use_accelerator=True, 
                 logging_type = LoggingType.CONSOLE, 
                 level = LoggingLevel.INFO) -> None:
        
        super().__init__(logging_type, level)
        self.log = logging.getLogger("Geneformer-Model")

        # load model configs via model_dir input
        self.model_dir = Path(model_dir)
        with open(self.model_dir / "args.json", "r") as f:
            model_config = json.load(f)

        self.model_config = model_config
        self.device = model_config['device']

        self.model =  BertForMaskedLM.from_pretrained(self.model_dir / model_config['model_name'], output_hidden_states=True, output_attentions=False)
        self.model.eval()#.to("cuda:0")
        self.model = self.model.to(self.device)

        self.layer_to_quant = quant_layers(self.model) + model_config['emb_layer']
        self.emb_mode = model_config["emb_mode"]
        self.forward_batch_size = model_config["batch_size"]
        
        if use_accelerator:
            self.accelerator = Accelerator(project_dir=self.model_dir, cpu=self.model_config["accelerator"]["cpu"])
            self.model = self.accelerator.prepare(self.model)
        else:
            self.accelerator = None

    def process_data(self, data: AnnData, data_config_path: Union[str, Path]) -> DataLoader:    

        with open(data_config_path) as f:
            config = json.load(f)
        self.data_config = config

        files_config = {
            "mapping_path": self.model_dir / "human_gene_to_ensemble_id.pkl",
            "gene_median_path": self.model_dir / "gene_median_dictionary.pkl",
            "token_path": self.model_dir / "token_dictionary.pkl"
        }

        mappings = pkl.load(open(files_config["mapping_path"], 'rb'))
        data.var['ensembl_id'] = data.var['gene_symbols'].apply(lambda x: mappings.get(x,{"id":None})['id'])

        # load token dictionary (Ensembl IDs:token)
        with open(files_config["token_path"], "rb") as f:
            self.gene_token_dict = pickle.load(f)

        self.token_gene_dict = {v: k for k, v in self.gene_token_dict.items()}
        self.pad_token_id = self.gene_token_dict.get("<pad>")

        self.tk = TranscriptomeTokenizer({"cell_type": "cell_type"}, 
                                         nproc=self.data_config["geneformer"]["nproc"], 
                                         gene_median_file = files_config["gene_median_path"], 
                                         token_dictionary_file = files_config["token_path"])

        tokenized_cells, cell_metadata =  self.tk.tokenize_anndata(data)
        tokenized_dataset = self.tk.create_dataset(tokenized_cells, cell_metadata, use_generator=False)
        
        output_path = self.data_config["geneformer"].get("tokenized_dataset_output_path")
        if output_path:
            output_path = Path(output_path).with_suffix(".dataset")
            tokenized_dataset.save_to_disk(output_path)
        return tokenized_dataset

    def get_embeddings(self, dataset: Dataset) -> np.array:
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