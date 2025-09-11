from helical.constants.paths import CACHE_DIR_HELICAL
import os


class stateConfig:
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
        freeze_backbone: bool = True,
        seed: int = 42,
    ):

        self.config = {
            "batch_size": batch_size,
            "freeze_backbone": freeze_backbone,
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
                "state/state_transition/config.yaml",
                "state/state_transition/pert_onehot_map.pt",
                "state/state_transition/batch_onehot_map.pkl",
                "state/state_transition/var_dims.pkl",
                "state/state_transition/cell_type_onehot_map.pkl",
                "state/state_transition/data_module.torch",
                "state/state_transition/final.ckpt",
            ],
        }
