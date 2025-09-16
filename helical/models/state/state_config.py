from helical.constants.paths import CACHE_DIR_HELICAL
import os


class StateConfig:
    """
    Configuration class for the State Model.

    Parameters
    ----------
    output_path : str, optional, default="prediction.h5ad"
        Path where the prediction results will be saved as an h5ad file
    checkpoint_name : str, optional, default="final.ckpt"
        Name of the model checkpoint file to load
    perturb_dir : str, optional, default=os.path.join(CACHE_DIR_HELICAL, "state", "state_transition")
        Directory path where perturbation-related model files are stored
    embed_dir : str, optional, default=os.path.join(CACHE_DIR_HELICAL, "state", "state_embed")
        Directory path where embedding-related model files are stored
    batch_size : int, optional, default=16
        The batch size for inference
    batch_col : str, optional, default="batch_var"
        Column name in the data that contains batch information
    pert_col : str, optional, default="target_gene"
        Column name in the data that contains perturbation/target gene information
    control_pert : str, optional, default="non-targeting"
        Label used to identify control/non-targeting perturbations
    embed_key : str, optional, default=None
        Key to access embeddings in the data object
    celltype_col : str, optional, default=None
        Column name in the data that contains cell type information
    celltypes : str, optional, default=None
        Specific cell types to focus on (comma-separated string)
    max_set_len : int, optional, default=None
        Maximum length for gene sets or perturbation sets
    tsv : str, optional, default=None
        Path to a TSV file containing additional data or metadata
    seed : int, optional, default=42
        Random seed for reproducibility

    Returns
    -------
    StateConfig
        The State configuration object

    Notes
    -----
    This configuration contains all the parameters needed to configure the State Embedding 
    model and State Transition model for perturbation prediction. The configuration
    includes paths for model files to download and model parameters.

    """
    def __init__(
        self,
        output_path: str = "prediction.h5ad",
        checkpoint_name: str = "final.ckpt",
        perturb_dir: str = os.path.join(CACHE_DIR_HELICAL, "state", "state_transition"),
        embed_dir: str = os.path.join(CACHE_DIR_HELICAL, "state", "state_embed"),
        batch_size: int = 16,
        batch_col: str = "batch_var",
        pert_col: str = "target_gene",
        control_pert: str = "non-targeting",
        embed_key: str = None,
        celltype_col: str = None,
        celltypes: str = None,
        max_set_len: int = None,
        tsv: str = None,
        seed: int = 42,
    ):

        self.config = {
            "batch_size": batch_size,
            "perturb_dir": perturb_dir,
            "embed_dir": embed_dir,
            "checkpoint_name": checkpoint_name,
            "output_path": output_path,
            "control_pert": control_pert,
            "embed_key": embed_key,
            "pert_col": pert_col,
            "celltype_col": celltype_col,
            "celltypes": celltypes,
            "batch_col": batch_col,
            "seed": seed,
            "max_set_len": max_set_len,
            "tsv": tsv,
            "embed_files_to_download": [
                "state/state_embed/protein_embeddings.pt",
                "state/state_embed/config.yaml",
                "state/state_embed/se600m_model_weights.pt",
            ],
            "perturbation_files_to_download": [
                "state/state_transition/pert_onehot_map.pt",
                "state/state_transition/batch_onehot_map.pkl",
                "state/state_transition/ST_all.pt",
            ],
        }
