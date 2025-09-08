from helical.constants.paths import CACHE_DIR_HELICAL
from pathlib import Path
import os

class stateConfig:
    def __init__(
        self,
        output: str = "competition/prediction.h5ad",
        checkpoint: str = "checkpoints/final.ckpt",
        # pert_col: str = "drugname_drugconc",
        pert_col: str = "target_gene",
        embed_key: str = None,
        celltype_col: str = None,
        celltypes: str = None,
        batch_col: str = None,
        control_pert: str = "non-targeting",
        seed: int = 42,
        max_set_len: int = None,
        tsv: str = None,
        batch_size: int = 16, 
        hf_model_dir: Path = Path(CACHE_DIR_HELICAL, "state/embed_files"),
        model_dir: Path = Path(CACHE_DIR_HELICAL, "state/transition_files"),
        model_config: str = Path(CACHE_DIR_HELICAL, "state/transition_files", "config.yaml"),
        embed_repo_id: str = "arcinstitute/SE-600M",
        embed_checkpoint: str = "se600m_epoch16.ckpt",
        perturb_repo_id: str = "arcinstitute/ST-Tahoe",
        model_name: str = "best_model",
        freeze_backbone: bool = True,
        checkpoint_name: str = "final.ckpt",
    ):

        model_path = Path(CACHE_DIR_HELICAL, "state/state_finetune", f"{model_name}.pt")
        os.makedirs(os.path.join(CACHE_DIR_HELICAL, "state/state_finetune"), exist_ok=True)

        self.config = {      
            "embed": {
                "repo_id": embed_repo_id,
                "filename": embed_checkpoint,
                "cache_dir": hf_model_dir,
            },
            "list_of_files_to_download": ["https://huggingface.co/arcinstitute/ST-Tahoe"],
            "finetune": {
                "batch_size": batch_size,
                "model_dir": model_dir,
                "model_path": model_path,
                "model_config": model_config,
                "freeze_backbone": freeze_backbone,
            },
            "perturb": {
                "output": output,
                "repo_id": perturb_repo_id,
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
                "save": True,
                "checkpoint_name": checkpoint_name,
            },
        }
