from typing import Optional, Literal
from helical.constants.paths import CACHE_DIR_HELICAL
from pathlib import Path
import os

class stateConfig:
    def __init__(
        self,
        output: str = "competition/prediction.h5ad",
        model_dir: str = "competition/first_run",
        checkpoint: str = "competition/first_run/checkpoints/final.ckpt",
        pert_col: str = "drugname_drugconc",
        model_config: str = "/home/rasched/home/pr_repo/helical/helical/models/state/model_configs.yaml",
        embed_key: str = None,
        celltype_col: str = None,
        celltypes: str = None,
        batch_col: str = None,
        control_pert: str = "non-targeting",
        seed: int = 42,
        max_set_len: int = None,
        tsv: str = None,
        batch_size: int = 16,
        hf_model_dir: Path = Path(CACHE_DIR_HELICAL, "state"),
        embed_repo_id: str = "arcinstitute/SE-600M",
        embed_checkpoint: str = "se600m_epoch16.ckpt",
        perturb_repo_id: str = "arcinstitute/ST-Tahoe",
        head: Literal["classification"] = "classification",
        model_name: str = "best_model",
    ):

        model_path = Path(CACHE_DIR_HELICAL, "state/state_CP", f"{model_name}.pt")
        self.config = {
            "model_path": model_path,
            
            "finetune": {
                "batch_size": batch_size,
                "head": head,
                # "checkpoint_path": os.path.join(perturb_model_dir, perturb_checkpoint),
            },
            "embed": {
                "repo_id": embed_repo_id,
                "filename": embed_checkpoint,
                "model_dir": hf_model_dir,
                "output_path": os.path.join(hf_model_dir, "SE-600M_MODEL"),
            },
            "perturb": {
                "output": output,
                "model_config": model_config,
                "repo_id": perturb_repo_id,
                "model_dir": model_dir,
                "checkpoint": checkpoint,
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
                "hf_model_dir": os.path.join(hf_model_dir, "ST-Tahoe_MODEL"),
            },
        }
