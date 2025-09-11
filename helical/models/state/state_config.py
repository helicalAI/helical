from helical.constants.paths import CACHE_DIR_HELICAL
import os


class stateConfig:
    def __init__(
        self,
        output: str = "competition/prediction.h5ad",
        checkpoint_name: str = "final.ckpt",
        model_dir: str = os.path.join(CACHE_DIR_HELICAL, "state", "state_transition"),
        model_config: str = os.path.join(
            CACHE_DIR_HELICAL, "state", "state_transition", "config.yaml"
        ),
        model_name: str = "best_model",
        pert_col: str = "target_gene",
        embed_key: str = None,
        celltype_col: str = None,
        celltypes: str = None,
        batch_col: str = "batch_var",
        control_pert: str = "non-targeting",
        seed: int = 42,
        use_perturbation_embeddings: bool = True,
        max_set_len: int = None,
        tsv: str = None,
        batch_size: int = 16,
        freeze_backbone: bool = True,
    ):

        self.config = {
            "embed": {
                "batch_size": batch_size,
                "cache_dir": os.path.join(CACHE_DIR_HELICAL, "state", "state_embed"),
                # files downloaded from remote server - do NOT edit unless you have your own configurations/weights
                "list_of_files_to_download": [
                    "state/state_embed/protein_embeddings.pt",
                    "state/state_embed/config.yaml",
                    "state/state_embed/se600m_model_weights.pt",
                ],
            },
            "finetune": {
                "batch_size": batch_size,
                "model_dir": model_dir,
                # "model_path": model_path,
                "model_config": model_config,
                "freeze_backbone": freeze_backbone,
                "checkpoint_name": checkpoint_name,
                "control_pert": control_pert,
                "pert_col": pert_col,
                "batch_col": batch_col,
                "use_perturbation_embeddings": use_perturbation_embeddings,
                "celltype_col": celltype_col,
            },
            "perturb": {
                "list_of_files_to_download": [
                    "state/state_transition/config.yaml",
                    "state/state_transition/pert_onehot_map.pt",
                    "state/state_transition/batch_onehot_map.pkl",
                    "state/state_transition/var_dims.pkl",
                    "state/state_transition/cell_type_onehot_map.pkl",
                    "state/state_transition/data_module.torch",
                    "state/state_transition/final.ckpt",
                ],
                "checkpoint_name": checkpoint_name,
                "output": output,
                "model_dir": model_dir,
                "model_config": model_config,
                "control_pert": control_pert,
                "embed_key": embed_key,
                "pert_col": pert_col,
                "celltype_col": celltype_col,
                "celltypes": celltypes,
                "batch_col": batch_col,
                "seed": seed,
                "max_set_len": max_set_len,
                "tsv": tsv,
            },
        }
