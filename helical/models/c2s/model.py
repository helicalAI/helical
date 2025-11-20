'''

This package implements the Cell2Sen model from the open-source weights available on huggingface and the Gemma 2-2b/27b Model.
We add custom code that follows the original method outlined in the C2S-Scale paper. 
The model supports pre-processing and generating embeddings/perturbations.
See the tutorial notebook for usage. 

'''

import torch
import anndata
import scanpy as sc
import numpy as np
from tqdm import tqdm
from helical.models.base_models import HelicalBaseFoundationModel
# from helical.utils.downloader import Downloader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sklearn.linear_model import LinearRegression
from datasets import Dataset
from .config import Cell2SenConfig, PERTURBATION_PROMPT, EMBEDDING_PROMPT
import logging
import torch._dynamo

LOGGER = logging.getLogger(__name__)

class Cell2Sen(HelicalBaseFoundationModel):
    """
    Cell2Sen Model.

    The Cell2Sen Model is a transformer/Gemma-based model that can be used to generate cell sentences from gene expression data.

    Example
    -------
    ```python
    from helical.models.cell2sen import Cell2Sen, Cell2SenConfig
    import anndata as ad

    config = Cell2SenConfig(batch_size=16)
    cell2sen = Cell2Sen(configurer=config)

    # Process your data
    dataloader = cell2sen.process_data(adata)

    # Get embeddings
    embeddings = cell2sen.get_embeddings(dataloader)
    print("State embeddings shape:", embeddings.shape)
    ```

    Parameters
    ----------
    configurer : Cell2SenConfig, optional, default=None
        The model configuration. If None, uses default Cell2SenConfig.

    """

    def __init__(self, configurer: Cell2SenConfig = None) -> None:
        super().__init__()

        if configurer is None:
            self.config = Cell2SenConfig().config
        else:
            self.config = configurer.config

        # downloader = Downloader()
        # for file in self.config["list_of_files_to_download"]:
        #     downloader.download_via_name(file)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.config["dtype"] == "bfloat16":
            self.torch_dtype = torch.bfloat16
        elif self.config["dtype"] == "float32":
            self.torch_dtype = torch.float32
        else:
            raise ValueError(f"Dtype {self.config['dtype']} not supported. Please choose from 'bfloat16' or 'float32'.")

        if self.torch_dtype == torch.bfloat16 and self.device == "cpu":
            LOGGER.warning("Bfloat16 is not supported on CPU. Defaulting to 'float32' instead.")
            self.torch_dtype = torch.float32
    
        if self.config["use_quantization"]:
            self.bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.torch_dtype
            )
        else:
            self.bnb_config = None
       
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config["hf_model_path"],
            torch_dtype=self.torch_dtype, 
            cache_dir=self.config["model_path"],
            quantization_config=self.bnb_config,
            ).to(self.device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["hf_model_path"], cache_dir=self.config["model_path"])
        self.model.eval()
         
        self.batch_size = self.config['batch_size']
        self.max_new_tokens = self.config['max_new_tokens']
        self.organism = self.config['organism']
        self.perturbation_column = self.config['perturbation_column']
        self.return_fit = self.config['return_fit']
        
        LOGGER.info("Successfully loaded model")

    def process_data(
        self, 
        adata: anndata.AnnData, 
        max_genes: int = None,
    ):
        """
        Process anndata to create a HuggingFace Dataset with cell sentences and fit parameters.
        
        Parameters:
        -----------
        anndata : AnnData
            Annotated data object with gene expression
        max_genes : int, optional
            Maximum number of genes to process per cell in descending expression order
        Returns:
        --------
        dataset : Dataset
            HuggingFace Dataset with fields: cell_sentence, fit_parameters, organism, perturbations
        """

        LOGGER.info("Processing data")

        # standard log-normalization, enables accurate expression reconstruction
        anndata = adata.copy()
        sc.pp.normalize_total(anndata, target_sum=1e4)
        sc.pp.log1p(anndata, base=10)
   
        X = anndata.X    
        if hasattr(X, 'toarray'):
            X = X.toarray()


        # gene names corresponding to each cell in order
        # anndata.X[i, j] is the expression of the j-th gene in the i-th cell
        cell_sentences = []

        # Collect ranks and corresponding expression as training data for reconstruction model
        ranks_to_fit = []
        expr_to_fit = []

        if self.organism is None:
            if 'organism' in anndata.uns:
                self.organism = anndata.uns['organism']
            elif 'organism' in anndata.obs.columns:
                # If organism varies per cell, use first one or most common
                self.organism = anndata.obs['organism'].iloc[0] if len(anndata.obs['organism'].unique()) == 1 else anndata.obs['organism'].mode()[0]
            elif 'species' in anndata.uns:
                self.organism = anndata.uns['species']
            elif 'species' in anndata.obs.columns:
                self.organism = anndata.obs['species'].iloc[0] if len(anndata.obs['species'].unique()) == 1 else anndata.obs['species'].mode()[0]
            else:
                self.organism = "unknown"  # Default if not found

        # Process each cell
        progress_bar = tqdm(total=X.shape[0], desc="Processing cells")
        for cell_idx in range(X.shape[0]):
            gene_names = anndata.var_names.values
            cell_expr = X[cell_idx, :]
            
            # Rank nonzero genes by expression (highest = rank 1)
            non_zero_mask = cell_expr > 0 
            if non_zero_mask.sum() == 0:
                LOGGER.warning(f"No genes expressed above zero in cell {cell_idx}. Skipping.")
                progress_bar.update(1)
                continue
              
            cell_expr = cell_expr[non_zero_mask]
            gene_names = gene_names[non_zero_mask]

            ranked_indices = np.argsort(cell_expr)[::-1]
            expr_values = cell_expr[ranked_indices]  # Expression values in descending order
            gene_names = gene_names[ranked_indices]  # Gene names in descending order by expression

            # Cut at max_genes if desired
            if max_genes:
                if len(gene_names) > max_genes:
                    gene_names = gene_names[:max_genes]
                    expr_values = expr_values[:max_genes]

            if self.return_fit:
                ranks_to_fit.extend(np.arange(1, len(gene_names) + 1))
                expr_to_fit.extend(expr_values)
               
            cell_sentence = " ".join(gene_names)            
            cell_sentences.append(cell_sentence)
            progress_bar.update(1)


        if self.return_fit:
            log_ranks_to_fit = np.log10(ranks_to_fit)
            expr_to_fit = np.array(expr_to_fit)
            
            # Fit linear model to predict log-normalized expression from log rank: expr(g) = slope * log(rank(g)) = intercept
            model = LinearRegression()
            model.fit(log_ranks_to_fit.reshape(-1, 1), np.array(expr_to_fit))
            slope, intercept = model.coef_[0], model.intercept_
            r_squared = model.score(log_ranks_to_fit.reshape(-1, 1), expr_to_fit)

            fit_parameters = {"slope": float(slope), "intercept": float(intercept), "r_squared": float(r_squared)}

        else:
            fit_parameters = None

        progress_bar.close()

        if self.perturbation_column is not None:
            perturbations = anndata.obs[self.perturbation_column].values.tolist()
            if len(perturbations) != len(cell_sentences):
                raise ValueError(f"Number of perturbations ({len(perturbations)}) does not match number of cells ({len(cell_sentences)})")
        else:
            perturbations = [None] * len(cell_sentences)
        
        dataset = Dataset.from_dict({
            'cell_sentence': cell_sentences,
            'fit_parameters': [fit_parameters] * len(cell_sentences),
            'organism': [self.organism] * len(cell_sentences),
            'perturbations': perturbations
        })

        LOGGER.info("Successfully processed data")
        
        return dataset

    def get_embeddings(
        self, 
        dataset: Dataset, 
        output_attentions: bool = False
        ):
        """
        Extract embeddings from cell sentences in a HuggingFace Dataset using the last hidden layer of Gemma.
        
        Parameters:
        -----------
        dataset : Dataset
            HuggingFace Dataset with 'cell_sentence' and 'organism' fields  
        
        output_attentions : bool, optional
            Whether to output the attention maps from the model. If set to True, the attention maps will be returned along with the embeddings.
            If set to False, only the embeddings will be returned. **Note**: This will increase the memory usage of the model significantly, so use it only if you need the attention maps.
        Returns:
        --------
        embeddings : np.ndarray
            Embeddings of shape (num_sentences, hidden_size)
        """

        LOGGER.info("Extracting embeddings from dataset")

        sentences_list = dataset['cell_sentence']
        organisms_list = dataset['organism']
        
        all_embeddings = []
        all_attentions = []

        progress_bar = tqdm(total=len(sentences_list), desc="Processing embeddings")
        for i in range(0, len(sentences_list), self.batch_size):
            batch_sentences = sentences_list[i:i + self.batch_size]
            batch_organisms = organisms_list[i:i + self.batch_size]
            
            prompts = [
                EMBEDDING_PROMPT.format(organism=org, cell_sentence=cs)
                for org, cs in zip(batch_organisms, batch_sentences)
            ]

            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=False,
                # truncation=True,
                # max_length=max_length
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(
                    **inputs, 
                    output_hidden_states=True,
                    output_attentions=output_attentions
                )
                last_hidden = outputs.hidden_states[-1]  # Shape: (batch_size, seq_len, hidden_size)
                attention_mask = inputs['attention_mask'].float() # Shape: (batch_size, seq_len, 1)

                # Sum embeddings over sequence length, masking out padding tokens
                sum_embeddings = torch.sum(last_hidden * attention_mask.unsqueeze(-1), dim=1)
                sum_mask = torch.clamp(attention_mask.sum(dim=1, keepdim=True), min=1e-9)
                batch_embeddings = sum_embeddings / sum_mask

                if output_attentions:
                    # outputs.attentions is a tuple of tensors, one per layer
                    # Each tensor has shape (batch_size, num_heads, seq_length, seq_length)
                    # Initialize all_attentions_per_layer on first batch
                    if len(all_attentions) == 0:
                        num_layers = len(outputs.attentions)
                        all_attentions = [[] for _ in range(num_layers)]
                    
                    # Append attention maps from each layer to the corresponding list
                    for layer_idx, attn in enumerate(outputs.attentions):
                        all_attentions[layer_idx].append(attn.float().cpu().numpy())
                del outputs

            all_embeddings.append(batch_embeddings.float().cpu().numpy())
            progress_bar.update(len(batch_sentences))
        progress_bar.close()
        LOGGER.info("Successfully extracted embeddings")

        if output_attentions:
            # Concatenate attention maps per layer across batches
            # Each element in stacked_attentions has shape (total_batch_size, num_heads, seq_length, seq_length)
            stacked_attentions = tuple(
                np.concatenate(all_attentions[layer_idx], axis=0)
                for layer_idx in range(len(all_attentions))
            )
            return np.concatenate(all_embeddings, axis=0), stacked_attentions
        else:
            return np.concatenate(all_embeddings, axis=0)

    def get_perturbations(
        self, 
        dataset: Dataset, 
        perturbations_list: list[str] = None, 
        ):
        """
        Generate perturbed cell sentences using the model.
        
        Parameters:
        -----------
        dataset : Dataset
            HuggingFace Dataset with 'cell_sentence' and 'perturbations' fields
        
        perturbations_list : list[str], optional
            List of perturbations to apply to the cells. If None, uses the perturbations from the dataset.
            If provided, overrides the perturbations in the dataset. E.g. ["pert1", "pert2", "pert3", ...]

        Returns:
        --------
        perturbed_dataset : Dataset
            HuggingFace Dataset with 'cell_sentence' and 'perturbations' fields and a new column 'perturbed_cell_sentence'

        perturbed_sentences : list
            List of perturbed cell sentences (strings)
        """

        LOGGER.info("Generating perturbed cell sentences")

        sentences_list = dataset['cell_sentence']
        organisms_list = dataset['organism']
        if perturbations_list is None:
            perturbations_list = dataset['perturbations']
        else:
            if len(perturbations_list) != len(sentences_list):
                raise ValueError(f"perturbations_list length ({len(perturbations_list)}) must match dataset length ({len(sentences_list)})")
        
        # Handle None perturbations - skip those entries or use empty string
        valid_indices = [i for i, p in enumerate(perturbations_list) if p is not None]
        if len(valid_indices) == 0:
                raise ValueError("No valid perturbations found in dataset. All perturbations are None.")
                    
        valid_sentences = [sentences_list[i] for i in valid_indices]
        valid_perturbations = [perturbations_list[i] for i in valid_indices]
        valid_organisms = [organisms_list[i] for i in valid_indices]    
        all_perturbed = []
        # Process in batches
        progress_bar = tqdm(total=len(valid_sentences), desc="Processing valid perturbations")
        for i in range(0, len(valid_sentences), self.batch_size):
            batch_cells = valid_sentences[i:i + self.batch_size]
            batch_perturbs = valid_perturbations[i:i + self.batch_size]
            batch_organisms = valid_organisms[i:i + self.batch_size]

            prompts = [
                PERTURBATION_PROMPT.format(
                    organism=org,
                    perturbation=pert,  # Changed from perturbation_in_words
                    cell_sentence=cs
                )
                for org, pert, cs in zip(batch_organisms, batch_perturbs, batch_cells)
            ]
            
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=False,
                # truncation=True,
                # max_length=max_length
            ).to(self.device)
            
            with torch.no_grad():
                if self.config["use_quantization"]:
                    # Disable torch.compile entirely for quantized models
                    # suppress_errors alone isn't sufficient - we need to disable compilation
                    original_disable = torch._dynamo.config.disable
                    original_suppress = torch._dynamo.config.suppress_errors
                    torch._dynamo.config.disable = True
                    torch._dynamo.config.suppress_errors = True
                    try:
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=self.max_new_tokens,
                            do_sample=False,
                            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
                        )
                    finally:
                        torch._dynamo.config.disable = original_disable
                        torch._dynamo.config.suppress_errors = original_suppress
                else:
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
                    )
            
            input_lengths = inputs['attention_mask'].sum(dim=1)
            batch_perturbed = []
            
            for j, output in enumerate(outputs):
                # Extract only the generated tokens (skip the prompt)
                input_length = input_lengths[j].item()
                generated_tokens = output[input_length:]  # Only generated part
                decoded = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                batch_perturbed.append(decoded.strip())
            
            all_perturbed.extend(batch_perturbed)
            progress_bar.update(len(batch_cells))
        progress_bar.close()
        # Create result list with None for entries without perturbations
        perturbed_sentences = [None] * len(sentences_list)
        for idx, perturbed in zip(valid_indices, all_perturbed):
            perturbed_sentences[idx] = perturbed
        
        dataset = dataset.add_column('perturbed_cell_sentence', perturbed_sentences)

        LOGGER.info("Successfully generated perturbed cell sentences")

        return dataset, perturbed_sentences