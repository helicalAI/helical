from helical.constants.paths import CACHE_DIR_HELICAL
from pathlib import Path

EMBEDDING_PROMPT = """
You are given a list of genes in descending order of expression levels in a {organism} cell. \n
Genes: {cell_sentence} \n
Using this information, predict the cell type. 
"""

#EMBEDDING_PROMPT = """
#You are given a list of genes in descending order of expression levels in a {organism} cell. \n
#Genes: {cell_sentence} \n
#Using this information, predict the wether the cell comes from an ALS or pathological normal patient.
#"""

PERTURBATION_PROMPT = """
You are given a list of genes in descending order of expression levels in a {organism} control cell. \n
Using this information, predict the perturbed gene profile with the perturbation: {perturbation}.
Control Cell Sentence: {cell_sentence} \n
Perturbed Cell Sentence:
"""

class Cell2SenConfig:
    """
    Configuration class for the Cell2Sen Model.

    Parameters
    ----------
    batch_size: int = 16
        int: Number of samples to process in each batch during model operations. Default is 16.
    
    organism: str = None
        The organism from which the cell data is derived (e.g., 'human', 'mouse').

    perturbation_column: str = None
        Column name in the input data that specifies the perturbation applied to cells.

    max_new_tokens: int = 200
        Maximum number of new tokens that the model can generate for prediction. Default is 200.
        One gene is roughly 4 tokens. 

    return_fit: bool = False
        Whether to return model fit parameters in outputs. Default is False. This fits a linear model (y=mx+c) to the gene rank and expression values in log10-transformed space
        and can be used to map between expression values and gene ranks. The paper shows this is well captured by a linear model. The fit parameters are returned in the `fit_parameters` field.
    
    dtype: str = "bfloat16"
        Data type for the model. Default is "bfloat16". 
    
    model_size: str = "2B"
        Size of the model. Default is "2B".
        Choices are "2B" or "27B".
    
    use_quantization: bool = False
        Whether to use 4-bit quantization. Default is False.

    seed: int = 42
        Random seed for reproducibility. Default is 42.

    """
    def __init__(
        self,
        batch_size: int = 16,
        organism: str = None,
        perturbation_column: str = None,
        max_genes: int = None,
        max_new_tokens: int = 200,
        return_fit: bool = False,
        dtype: str = "bfloat16", 
        model_size: str = "2B",
        use_quantization: bool = False,
        seed: int = 42,
    ):

        if model_size == "2B":
            model_path = Path(CACHE_DIR_HELICAL, "c2s_model_2B")
            hf_model_path = "vandijklab/C2S-Scale-Gemma-2-2B"
            # list_of_files_to_download = [
            #     "c2s_model/config.json",
            #     "c2s_model/generation_config.json",
            #     "c2s_model/model-00001-of-00002.safetensors",
            #     "c2s_model/model-00002-of-00002.safetensors",
            #     "c2s_model/model.safetensors.index.json",
            #     "c2s_model/special_tokens_map.json",
            #     "c2s_model/tokenizer_config.json",
            #     "c2s_model/tokenizer.json",
            # ]
        elif model_size == "27B":
            model_path = Path(CACHE_DIR_HELICAL, "c2s_model_27B")
            hf_model_path = "vandijklab/C2S-Scale-Gemma-2-27B"
            # list_of_files_to_download = [
            #     "c2s_model_27B/config.json",
            #     "c2s_model_27B/generation_config.json",
            #     "c2s_model_27B/model-00001-of-00012.safetensors",
            #     "c2s_model_27B/model-00002-of-00012.safetensors",
            #     "c2s_model_27B/model-00003-of-00012.safetensors",
            #     "c2s_model_27B/model-00004-of-00012.safetensors",
            #     "c2s_model_27B/model-00005-of-00012.safetensors",
            #     "c2s_model_27B/model-00006-of-00012.safetensors",
            #     "c2s_model_27B/model-00007-of-00012.safetensors",
            #     "c2s_model_27B/model-00008-of-00012.safetensors",
            #     "c2s_model_27B/model-00009-of-00012.safetensors",
            #     "c2s_model_27B/model-00010-of-00012.safetensors",
            #     "c2s_model_27B/model-00011-of-00012.safetensors",
            #     "c2s_model_27B/model-00012-of-00012.safetensors",
            #     "c2s_model_27B/model.safetensors.index.json",
            #     "c2s_model_27B/special_tokens_map.json",
            #     "c2s_model_27B/tokenizer_config.json",
            #     "c2s_model_27B/tokenizer.json",
            # ]
        else:
            raise ValueError(f"Model size {model_size} not supported. Please choose from '2B' or '27B'.")

        self.config = {
            "hf_model_path": hf_model_path,
            # "list_of_files_to_download": list_of_files_to_download,
            "model_path": model_path,
            "batch_size": batch_size,
            "organism": organism,
            "perturbation_column": perturbation_column,
            "max_genes": max_genes,
            "max_new_tokens": max_new_tokens,
            "return_fit": return_fit,
            "use_quantization": use_quantization,
            "seed": seed,
            "dtype": dtype,
            "model_size": model_size,
        }