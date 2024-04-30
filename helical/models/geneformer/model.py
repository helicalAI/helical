from helical.models.helical import HelicalBaseModel
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
from typing import Union
import pickle as pkl

class Geneformer(HelicalBaseModel):
    
    def __init__(self, model_dir, model_args_path: Path = Path(__file__).parent.resolve() / "args.json", use_accelerator=True) -> None:

        """Initializes the Geneformer class

        Parameters
        ----------
        model_dir : str
            The path to the model directory 
        model_args_path : Path, optional
            The path to the model arguments file
        use_accelerator : bool, default=True
            Whether to use the accelerator class from Huggingface

        Returns
        -------
        None
        """
        
        super().__init__(model_dir, model_args_path)
        self.log = logging.getLogger("Geneformer-Model")
        self.device = self.model_config['device']

        self.model =  BertForMaskedLM.from_pretrained(self.model_dir / self.model_config['model_name'], output_hidden_states=True, output_attentions=False)
        self.model.eval()#.to("cuda:0")
        self.model = self.model.to(self.device)

        self.layer_to_quant = quant_layers(self.model) + self.model_config['emb_layer']
        self.emb_mode = self.model_config["emb_mode"]
        self.forward_batch_size = self.model_config["batch_size"]
        
    def process_data(self, data: AnnData, data_config_path: Union[str, Path],save_to_disk=False) -> Dataset:    
        """Processes the data for the UCE model

        Parameters 
        ----------
        data : AnnData
            The AnnData object containing the data to be processed
        data_config_path : Union[str, Path]
            The path to the data configuration file
        save_to_disk : bool, default=False
            Whether to save the tokenized dataset to disk

        Returns
        -------
        Dataset
            The tokenized dataset in the form of a Hugginface Dataset object
        """

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
        if output_path and save_to_disk:
            output_path = Path(output_path).with_suffix(".dataset")
            tokenized_dataset.save_to_disk(output_path)
        return tokenized_dataset

    def get_embeddings(self, dataset: Dataset) -> np.array:
        """Gets the gene embeddings from the Geneformer model   

        Parameters
        ----------
        dataset : Dataset
            The tokenized dataset containing the processed data

        Returns
        -------
        np.array
            The gene embeddings in the form of a numpy array
        """
        
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