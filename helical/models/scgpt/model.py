import os
import scanpy as sc
from helical.models.base_models import HelicalRNAModel
from helical.models.scgpt.scgpt_config import scGPTConfig
import numpy as np
from anndata import AnnData
import logging
from typing import Literal, Optional
from accelerate import Accelerator
from helical.models.scgpt.scgpt_utils import load_model
from helical.models.scgpt.dataset import Dataset
from helical.services.downloader import Downloader
from helical.models.scgpt.data_collator import DataCollator
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, SequentialSampler
from transformers import get_scheduler
from torch import optim
from torch.nn.modules import loss
import torch
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class scGPT(HelicalRNAModel):
    """scGPT Model. 
        The scGPT Model is a transformer-based model that can be used to extract gene embeddings from single-cell RNA-seq data.
        Currently we load the continous pre-training model from the scGPT repository as default model which works best on zero-shot tasks.


        Example
        -------
        >>> from helical.models import scGPT,scGPTConfig
        >>> import anndata as ad
        >>> scgpt_config=scGPTConfig(batch_size=10)
        >>> scgpt = scGPT(configurer=scgpt_config)
        >>> ann_data = ad.read_h5ad("./data/10k_pbmcs_proc.h5ad")
        >>> dataset = scgpt.process_data(ann_data[:100])
        >>> embeddings = scgpt.get_embeddings(dataset)


        Parameters
        ----------
        configurer : scGPTConfig, optional, default = configurer
            The model configuration.

        Returns
        -------
        None

        Notes
        -----
        We use the implementation from this `repository <https://github.com/bowang-lab/scGPT>`_ , which comes from the original authors. You can find the description of the method in this `paper <https://www.nature.com/articles/s41592-024-02201-0>`_.
        """
    configurer = scGPTConfig()

    def __init__(self, configurer: scGPTConfig = configurer) -> None:
          
        super().__init__()
        self.config = configurer.config
        
        downloader = Downloader()
        for file in self.config["list_of_files_to_download"]:
            downloader.download_via_name(file)

        self.model, self.vocab = load_model(self.config)
        
        if self.config["accelerator"]:
            self.accelerator = Accelerator(project_dir=self.config["model_path"].parent)
            self.model = self.accelerator.prepare(self.model)
        else:
            self.accelerator = None
        LOGGER.info(f"Model finished initializing.")
        
    def get_embeddings(self, dataset: Dataset) -> np.array:
        """Gets the gene embeddings

        Parameters 
        ----------
        dataset: Dataset
            The processed dataset to get the embeddings from.

        Returns
        -------
        np.array
            The gene embeddings in the form of a numpy array
        """
        LOGGER.info(f"Inference started:")

        try:
            use_batch_labels = dataset.batch_ids is not None
        except:
            use_batch_labels = False
            
        collator = DataCollator(
            do_padding=True,
            pad_token_id=self.vocab[self.config["pad_token"]],
            pad_value=self.config["pad_value"],
            do_mlm=False,
            do_binning=True,
            max_length=1200,
            sampling=True,
            keep_first_n_tokens=1,
        )
        data_loader = DataLoader(
            dataset,
            batch_size=self.config["batch_size"],
            sampler=SequentialSampler(dataset),
            collate_fn=collator,
            drop_last=False,
            pin_memory=True,
        )

        device = next(self.model.parameters()).device

        cell_embeddings = np.zeros(
            (len(dataset), self.config["embsize"]), dtype=np.float32
        )
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True): #torch.autocast(device_type=str(device),enabled=True): # torch.cuda.amp.autocast(enabled=True):
            count = 0
            for data_dict in tqdm(data_loader, desc="Embedding cells"):
                input_gene_ids = data_dict["gene"].to(device)
                src_key_padding_mask = input_gene_ids.eq(
                    self.vocab[self.config["pad_token"]]
                )
                embeddings = self.model._encode(
                    input_gene_ids,
                    data_dict["expr"].to(device),
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=data_dict["batch_labels"].to(device)
                    if use_batch_labels
                    else None,
                )

                embeddings = embeddings[:, 0, :]  # get the <cls> position embedding
                embeddings = embeddings.cpu().numpy()
                cell_embeddings[count : count + len(embeddings)] = embeddings
                count += len(embeddings)
        cell_embeddings = cell_embeddings / np.linalg.norm(
            cell_embeddings, axis=1, keepdims=True
        )

        # TODO?
        # return_new_adata (bool): Whether to return a new AnnData object. If False, will
        #     add the cell embeddings to a new :attr:`adata.obsm` with key "X_scGPT".
        # if return_new_adata:
        #     obs_df = adata.obs[obs_to_save] if obs_to_save is not None else None
        #     return sc.AnnData(X=cell_embeddings, obs=obs_df, dtype="float32")
        
        return cell_embeddings
    
    def process_data(self,
                     adata: AnnData, 
                     gene_names: str = "index", 
                     fine_tuning: bool = False,
                     n_top_genes: int = 1800, 
                     flavor: Literal["seurat", "cell_ranger", "seurat_v3", "seurat_v3_paper"] = "seurat_v3",
                     use_batch_labels: bool = False
    ) -> Dataset:
        """Processes the data for the scGPT model

        Parameters 
        ----------
        data : AnnData
            The AnnData object containing the data to be processed. 
            The Anndata requires the expression counts as the data matrix and the column with the gene symbols is defined by the argument gene_names.
        gene_names: str, optional, default = "index"
            The column in adata.var that contains the gene names. Default is to use the index column.
        fine_tuning: bool, optional, default = False
            If you intend to use the data to fine-tune the model on a downstream task, set this to True.
        n_top_genes: int, optional, default = 1800
           Only taken into account if you use the dataset for fine-tuning the model. Number of highly-variable genes to keep. Mandatory if flavor='seurat_v3'.
        flavor: Literal["seurat", "cell_ranger", "seurat_v3", "seurat_v3_paper"], optional, default = "seurat_v3",
            Only taken into account if you use the dataset for fine-tuning the model.
            Choose the flavor for identifying highly variable genes. 
            For the dispersion based methods in their default workflows, 
            Seurat passes the cutoffs whereas Cell Ranger passes n_top_genes.
        use_batch_labels: Bool, default = False
            Whether to use batch labels. Defaults to False.

        Returns
        -------
        Dataset
            The processed dataset.
        """
 
        self.ensure_data_validity(adata, gene_names, use_batch_labels)
        self.gene_names = gene_names
        if fine_tuning:
            # Preprocess the dataset and select `N_HVG` highly variable genes for downstream analysis.
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)

            # highly variable genes
            sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor=flavor)
            adata = adata[:, adata.var['highly_variable']]

        # filtering
        adata.var["id_in_vocab"] = [ self.vocab[gene] if gene in self.vocab else -1 for gene in adata.var[self.gene_names] ]
        LOGGER.info(f"Filtering out {np.sum(adata.var['id_in_vocab'] < 0)} genes to a total of {np.sum(adata.var['id_in_vocab'] >= 0)} genes with an id in the scGPT vocabulary.")
        adata = adata[:, adata.var["id_in_vocab"] >= 0]

        # Binning will be applied after tokenization. A possible way to do is to use the unified way of binning in the data collator.

        self.vocab.set_default_index(self.vocab["<pad>"])
        genes = adata.var[self.gene_names].tolist()
        gene_ids = np.array(self.vocab(genes), dtype=int)
        count_matrix = (adata.X if isinstance(adata.X, np.ndarray) else adata.X.A)

        # gene vocabulary ids
        if gene_ids is None:
            gene_ids = np.array(adata.var["id_in_vocab"])
        assert np.all(gene_ids >= 0)

        if use_batch_labels:
            batch_ids = np.array(adata.obs["batch_id"].tolist())

        dataset = Dataset(
            count_matrix, gene_ids, self.vocab, self.config, batch_ids if use_batch_labels else None
        )
        return dataset


    def ensure_data_validity(self, adata: AnnData, gene_names: str, use_batch_labels: bool) -> None:
        """Checks if the data is eligible for processing by the scGPT model  

        Parameters
        ----------
        data : AnnData
            The AnnData object containing the data to be validated. 
        gene_names : str
            The name of the column containing gene names.
        use_batch_labels : str
            Wheter to use batch labels.

        Raises
        ------
        KeyError
            If the data is missing column names.
        """
        self.ensure_rna_data_validity(adata, gene_names)

        if use_batch_labels:
            if not "batch_id" in adata.obs:
                message = "Data must have the 'obs' key 'batch_id' to be processed by the scGPT model."
                LOGGER.error(message)
                raise KeyError(message)

    def fine_tune(
            self,
            fine_tune_head: torch.nn.Module,
            train_input_data: Dataset, 
            labels,     
            validation_input_data = None,
            optimizer: optim = optim.AdamW,
            optimizer_params: dict = {'lr': 0.0001}, 
            loss_function: loss = loss.CrossEntropyLoss(), 
            epochs: int = 1,
            freeze_layers: int = 0,
            validation_dataset: Optional[Dataset] = None,
            lr_scheduler_params: Optional[dict] = None):
        """Fine-tunes the Geneformer model for classification tasks. 

        Parameters
        ----------

        train_dataset : Dataset
            A helical processed dataset for fine-tuning
        optimizer : torch.optim, default = torch.optim.AdamW
            The optimizer to be used for training.
        optimizer_params : dict
            The optimizer parameters to be used for the optimizer specified. This list should NOT include model parameters.
            e.g. optimizer_params = {'lr': 0.0001}
        loss_function : torch.nn.modules.loss, default = torch.nn.modules.loss.CrossEntropyLoss()
            The loss function to be used.
        label : str, optional, default = "cell_types"
            The column in the dataset containing the training labels. These should be stored as unique per class integers.
        epochs : int, optional, default = 10
            The number of epochs to train the model
        freeze_layers : int, optional, default = 0
            The number of layers to freeze.
        validation_dataset : Dataset, default = None
            A helical processed dataset for per epoch validation. If this is not specified, no validation will be performed.
        lr_scheduler_params : dict, default = None
            The learning rate scheduler parameters for the transformers get_scheduler method. The optimizer will be taken from the optimizer input and should not be included in the learning scheduler parameters. If not specified, no scheduler will be used.
            e.g. lr_scheduler_params = { 'name': 'linear', 'num_warmup_steps': 0, 'num_training_steps': 5 }

        Returns
        -------
        BertForSequenceClassification
            The fine-tuned model. Original model is a huggingface BertForMaskedLM model. By using BertForSequenceClassification, it allows for an automatic head to be added to the model for classification tasks.
        """
        device = next(self.model.parameters()).device
        class scGPTFineTuningModel(torch.nn.Module):
            def __init__(self, helical_model, fine_tuning_head):
                super(scGPTFineTuningModel, self).__init__()
                self.helical_model = helical_model # Ensure no overwriting of the original model
                self.fine_tuning_head = fine_tuning_head

            def forward(self, input_gene_ids, data_dict, src_key_padding_mask, use_batch_labels, device):
                embeddings = self.helical_model._encode(
                    input_gene_ids,
                    data_dict["expr"].to(device),
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=data_dict["batch_labels"].to(device)
                    if use_batch_labels
                    else None,
                )
                avg_pool_seq = embeddings[:, 0, :]
                output = self.fine_tuning_head(avg_pool_seq)
                return output
        try:
            use_batch_labels = train_input_data.batch_ids is not None
        except:
            use_batch_labels = False
                
        collator = DataCollator(
            do_padding=True,
            pad_token_id=self.vocab[self.config["pad_token"]],
            pad_value=self.config["pad_value"],
            do_mlm=False,
            do_binning=True,
            max_length=1200,
            sampling=True,
            keep_first_n_tokens=1,
        )

        data_loader = DataLoader(
            train_input_data,
            batch_size=self.config["batch_size"],
            sampler=SequentialSampler(train_input_data),
            collate_fn=collator,
            drop_last=False,
            pin_memory=True,
        )

        model = scGPTFineTuningModel(self.model, fine_tune_head).to(device)
        model.train()

        optimizer = optimizer(model.parameters(), **optimizer_params)

        lr_scheduler = None
        if lr_scheduler_params is not None: 
            lr_scheduler = get_scheduler(optimizer=optimizer, **lr_scheduler_params)

        with torch.cuda.amp.autocast(enabled=True): #torch.autocast(device_type=str(device),enabled=True): # torch.cuda.amp.autocast(enabled=True):
            for j in range(epochs):
                batch_count = 0
                batch_loss = 0.0
                batches_processed = 0
                training_loop = tqdm(data_loader, desc="Fine-Tuning")
                for data_dict in training_loop:
                    input_gene_ids = data_dict["gene"].to(device)
                    src_key_padding_mask = input_gene_ids.eq(
                        self.vocab[self.config["pad_token"]]
                    )
                    output = model(input_gene_ids, data_dict, src_key_padding_mask, use_batch_labels, device)
                    cell_types = torch.tensor(labels[batch_count: batch_count + self.config["batch_size"]], device=device)
                    batch_count += self.config["batch_size"]
                    loss = loss_function(output, cell_types)
                    loss.backward()
                    batch_loss += loss.item()
                    batches_processed += 1
                    optimizer.step()
                    optimizer.zero_grad()

                    training_loop.set_postfix({"loss": batch_loss/batches_processed})
                    training_loop.set_description(f"Fine-Tuning: epoch {j+1}/{epochs}")

                if lr_scheduler is not None:
                    lr_scheduler.step()

                if validation_input_data is not None:
                    testing_loop = tqdm(data_loader, desc="Fine-Tuning Validation")
                    accuracy = 0.0
                    count = 0.0
                    validation_batch_count = 0
                    for validation_data_dict in testing_loop:
                        input_gene_ids = validation_data_dict["gene"].to(device)
                        src_key_padding_mask = input_gene_ids.eq(
                            self.vocab[self.config["pad_token"]]
                        )
                        output = model(input_gene_ids, validation_data_dict, src_key_padding_mask, use_batch_labels, device)
                        cell_types = torch.tensor(labels[validation_batch_count: validation_batch_count + self.config["batch_size"]], device=device)
                        validation_batch_count += self.config["batch_size"]
                        accuracy += accuracy_score(cell_types.cpu(), torch.argmax(output, dim=1).cpu())
                        count += 1.0
                        testing_loop.set_postfix({"accuracy": accuracy/count})


        return model
