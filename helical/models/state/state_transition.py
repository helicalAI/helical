from typing import Dict, Optional, List
import os
import pickle
import warnings
import numpy as np
import scanpy as sc
import torch
from tqdm import tqdm

from helical.models.state._perturb_utils.state_transition_model import (
    StateTransitionPerturbationModel,
)

from helical.models.state.state_config import stateConfig
from helical.models.state._perturb_utils.utils import (
    to_dense,
    argmax_index_from_any,
    pad_adata_with_tsv,
    prepare_batch,
    cfg_setup_inference,
)
from huggingface_hub import hf_hub_download, snapshot_download
from helical.models.base_models import HelicalBaseFoundationModel
import yaml

# this code to do inference on new data using the transition model
class stateTransitionModel(HelicalBaseFoundationModel):
    def __init__(self, configurer: stateConfig = None) -> None:
        super().__init__()
        if configurer is None:
            configurer = stateConfig()
        
        # ckpt_path = hf_hub_download(
        #     repo_id=self.config["repo_id"],
        #     filename=self.config["filename"],
        #     local_dir=self.config["model_dir"],  # your target folder
        #     local_dir_use_symlinks=False,  # ensures an actual copy, not a symlink
        # )
        self.config = configurer.config["perturb"]

        # local_dir = snapshot_download(
        #     repo_id=self.config["repo_id"],
        #     local_dir=self.config["model_dir"],
        #     local_dir_use_symlinks=False,
        # )

        with open(self.config["model_config"], "r") as f:
            self.model_config = yaml.safe_load(f)
        
        # with open(os.path.join(self.config["model_dir"], "config.yaml"), "r") as f:
        #     self.model_config = yaml.safe_load(f)

        # self.config, control_pert = cfg_setup_inference(self.config)
        with open(os.path.join(self.config["model_dir"], "var_dims.pkl"), "rb") as f:
            var_dims = pickle.load(f)

        self.pert_dim = var_dims.get("pert_dim")
        self.batch_dim = var_dims.get("batch_dim", None)
        # mappings
        self.pert_onehot_map_path = os.path.join(
            self.config["model_dir"], "pert_onehot_map.pt"
        )
        self.batch_onehot_map_path = os.path.join(
            self.config["model_dir"], "batch_onehot_map.pkl"
        )

        self.checkpoint_path = self.config["checkpoint"]
        print(f"Using checkpoint: {self.checkpoint_path}")

        self.model = StateTransitionPerturbationModel.load_from_checkpoint(
            self.checkpoint_path, 
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device

        self.cell_set_len = (
            self.config["max_set_len"]
            if self.config["max_set_len"] is not None
            else getattr(self.model, "cell_sentence_len", 256)
        )
        self.uses_batch_encoder = getattr(self.model, "batch_encoder", None) is not None
        self.output_space = getattr(self.model, "output_space", self.model_config.get("data", {}).get("kwargs", {}).get("output_space", "gene"))

        print(f"Model device: {self.device}")
        print(f"Model cell_set_len (max sequence length): {self.cell_set_len}")
        print(f"Model uses batch encoder: {bool(self.uses_batch_encoder)}")
        print(f"Model output space: {self.output_space}")

    def process_data(self, adata: sc.AnnData):        

        control_pert = self.config["control_pert"]
        if control_pert is None:
            try:
                control_pert = self.model_config["data"]["kwargs"]["control_pert"]
            except Exception:
                control_pert = None

        if control_pert is None and self.config["pert_col"] == "drugname_drugconc":
            control_pert = "[('DMSO_TF', 0.0, 'uM')]"

        if control_pert is None:
            control_pert = "non-targeting"
        
        self.config["control_pert"] = control_pert

        # choose cell type column
        if self.config["celltype_col"] is None:
            ct_from_cfg = None
            try:
                ct_from_cfg = self.model_config["data"]["kwargs"].get("cell_type_key", None)
            except Exception:
                pass

            guess = self.pick_first_present(
                adata,
                candidates=(
                    [
                        ct_from_cfg,
                        "cell_type",
                        "celltype",
                        "cellType",
                        "ctype",
                        "celltype_col",
                    ]
                    if ct_from_cfg
                    else ["cell_type", "celltype", "cellType", "ctype", "celltype_col"]
                ),
            )
            self.config["celltype_col"] = guess

            celltype_col = self.config["celltype_col"]
            if celltype_col:
                print(f"Grouping by cell type column: {celltype_col}")
            else:
                print("Grouping by cell type column: not found; no grouping")

        # choose batch column
        if self.config["batch_col"] is None:
            try:
                self.config["batch_col"] = self.model_config["data"]["kwargs"].get(
                    "batch_col", None
                )
            except Exception:
                self.config["batch_col"] = None

        if self.config["tsv"]:
            print(f"==> TSV padding mode: loading {self.config['tsv']}")
            pad_rng = np.random.RandomState(self.config["seed"])

            adata = pad_adata_with_tsv(
                adata=adata,
                tsv_path=self.config["tsv"],
                pert_col=self.config["pert_col"],
                control_pert=control_pert,
                rng=pad_rng,
                quiet=False,
            )

        # optional filter by cell types
        if self.config["celltype_col"] and self.config["celltypes"]:
            keep_cts = [ct.strip() for ct in self.config["celltypes"].split(",")]
            if self.config["celltype_col"] not in adata.obs:
                raise ValueError(
                    f"Column '{self.config['celltype_col']}' not in adata.obs"
                )

            n0 = adata.n_obs
            adata = adata[adata.obs[self.config["celltype_col"]].isin(keep_cts)].copy()

            print(
                f"Filtered to {adata.n_obs} cells (from {n0}) "
                f"for cell types: {keep_cts}"
            )

        # select features: embeddings or genes
        if self.config["embed_key"] is None:
            self.X_in = to_dense(adata.X)  # [N, E_in]
            self.writes_to = (".X", None)  # write predictions to .X
        else:
            if self.config["embed_key"] not in adata.obsm:
                raise KeyError(
                    f"Embedding key '{self.config['embed_key']}' "
                    f"not found in adata.obsm"
                )

            # [N, E_in]
            self.X_in = np.asarray(adata.obsm[self.config["embed_key"]])
            # write predictions to obsm[embed_key]
            self.writes_to = (".obsm", self.config["embed_key"])

        if self.config["embed_key"] is None:
            print("Using adata.X as input features")
        else:
            print(
                f"Using adata.obsm[{self.config['embed_key']}] "
                f"as input features: shape {self.X_in.shape}"
            )

        # pick pert names; ensure they are strings
        if self.config["pert_col"] not in adata.obs:
            raise KeyError(
                f"Perturbation column '{self.config['pert_col']}' "
                f"not found in adata.obs"
            )

        self.pert_names_all = adata.obs[self.config["pert_col"]].astype(str).values

        # derive batch indices (per-token integers) if needed
        self.batch_indices_all: Optional[np.ndarray] = None
        
        batch_onehot_map = None
        if os.path.exists(self.batch_onehot_map_path):
            with open(self.batch_onehot_map_path, "rb") as f:
                batch_onehot_map = pickle.load(f)

        if self.uses_batch_encoder:
            # locate batch column
            batch_col = self.config["batch_col"]
            if batch_col is None:
                candidates = [
                    "gem_group",
                    "gemgroup",
                    "batch",
                    "donor",
                    "plate",
                    "experiment",
                    "lane",
                    "batch_id",
                ]

                batch_col = next((c for c in candidates if c in adata.obs), None)

            if batch_col is not None and batch_col in adata.obs:
                raw_labels = adata.obs[batch_col].astype(str).values
                if batch_onehot_map is None:
                    warnings.warn(
                        f"Model has a batch encoder, "
                        f"but '{self.batch_onehot_map_path}' not found. "
                        "Batch info will be ignored; predictions may degrade."
                    )
                    self.uses_batch_encoder = False
                else:
                    # Convert labels to indices using saved map
                    label_to_idx: Dict[str, int] = {}
                    for k, v in batch_onehot_map.items():
                        key = str(k)
                        idx = argmax_index_from_any(v, expected_dim=self.batch_dim)

                        if idx is not None:
                            label_to_idx[key] = idx
                    idxs = np.zeros(len(raw_labels), dtype=np.int64)
                    misses = 0
                    for i, lab in enumerate(raw_labels):
                        if lab in label_to_idx:
                            idxs[i] = label_to_idx[lab]
                        else:
                            misses += 1
                            idxs[i] = 0  # fallback to zero
                    if misses:
                        print(
                            f"Warning: {misses} / {len(raw_labels)} "
                            f"batch labels not found in saved mapping;"
                            f"using index 0 as fallback."
                        )
                    self.batch_indices_all = idxs
            else:
                print(
                    """Batch encoder present, but no batch column found;
                        proceeding without batch indices."""
                )
                self.uses_batch_encoder = False

        return adata

    def get_embeddings(self, adata: sc.AnnData):

        rng = np.random.RandomState(self.config["seed"])

        # Identify control vs non-control
        ctl_mask = self.pert_names_all == str(self.config["control_pert"])
        n_controls = int(ctl_mask.sum())
        n_total = adata.n_obs
        n_nonctl = n_total - n_controls
        print(
            f"Cells: total={n_total}, control={n_controls}, " f"non-control={n_nonctl}"
        )

        # Where we will write predictions (initialize with originals;
        # we overwrite all rows, including controls)
        if self.writes_to[0] == ".X":
            sim_X = self.X_in.copy()
            out_target = "X"
        else:
            sim_obsm = self.X_in.copy()
            out_target = f"obsm['{self.writes_to[1]}']"

        # Group labels for set-to-set behavior
        if self.config["celltype_col"] and self.config["celltype_col"] in adata.obs:
            group_labels = adata.obs[self.config["celltype_col"]].astype(str).values
            unique_groups = np.unique(group_labels)
        else:
            group_labels = np.array(["__ALL__"] * n_total)
            unique_groups = np.array(["__ALL__"])

        # Control pools (group-specific with fallback to global)
        all_control_indices = np.where(ctl_mask)[0]

        def group_control_indices(group_name: str) -> np.ndarray:
            if group_name == "__ALL__":
                return all_control_indices
            grp_mask = group_labels == group_name
            grp_ctl = np.where(grp_mask & ctl_mask)[0]
            return grp_ctl if len(grp_ctl) > 0 else all_control_indices

        pert_onehot_map: Dict[str, torch.Tensor] = torch.load(
            self.pert_onehot_map_path, weights_only=False
        )

        # default pert vector when unmapped label shows up
        if self.config["control_pert"] in pert_onehot_map:
            default_pert_vec = pert_onehot_map[self.config["control_pert"]].float().clone()
        else:
            default_pert_vec = torch.zeros(self.pert_dim, dtype=torch.float32)
            if self.pert_dim and self.pert_dim > 0:
                default_pert_vec[0] = 1.0

        print(
            "Running virtual experiment (homogeneous per-perturbation "
            "forward passes; controls included)..."
        )

        with torch.no_grad():
            for g in unique_groups:
                grp_idx = np.where(group_labels == g)[0]
                if len(grp_idx) == 0:
                    continue

                # control pool for this group (fallback to global if empty)
                grp_ctrl_pool = group_control_indices(g)
                if len(grp_ctrl_pool) == 0:
                    print(
                        f"Group '{g}': no control cells available anywhere; "
                        f"leaving rows unchanged."
                    )
                    continue

                # --- iterate by perturbation so
                # each forward pass is homogeneous ---

                grp_perts = np.unique(self.pert_names_all[grp_idx])
                POSTFIX_WIDTH = 30
                pbar = tqdm(
                    grp_perts,
                    desc=f"Group {g}",
                    # r_bar already has n/total, time, rate, and postfix
                    bar_format="{l_bar}{bar}{r_bar}",
                    dynamic_ncols=True,
                    disable=False,
                )

                for p in pbar:
                    current_postfix = f"Pert: {p}"
                    pbar.set_postfix_str(
                        f"{current_postfix:<{POSTFIX_WIDTH}.{POSTFIX_WIDTH}}"
                    )

                    idxs = grp_idx[self.pert_names_all[grp_idx] == p]
                    if len(idxs) == 0:
                        continue

                    # one-hot vector for this perturbation (repeat across window)
                    vec = pert_onehot_map.get(p, None)
                    if vec is None:
                        vec = default_pert_vec
                        print(
                            f"  (group {g}) pert '{p}' not in mapping; "
                            f"using control fallback one-hot."
                        )

                    start = 0
                    while start < len(idxs):
                        end = min(start + self.cell_set_len, len(idxs))
                        idx_window = idxs[start:end]
                        win_size = len(idx_window)

                        # 1) Sample matched control basals (with replacement)
                        sampled_ctrl_idx = rng.choice(
                            grp_ctrl_pool, size=win_size, replace=True
                        )

                        ctrl_basal = self.X_in[sampled_ctrl_idx, :]  # [win, E_in]

                        # 2) Build homogeneous pert one-hots
                        pert_oh = (
                            vec.float().unsqueeze(0).repeat(win_size, 1)
                        )  # [win, pert_dim]

                        # 3) Batch indices (optional)
                        if (
                            self.uses_batch_encoder
                            and self.batch_indices_all is not None
                        ):
                            bi = torch.tensor(
                                self.batch_indices_all[idx_window], dtype=torch.long
                            )  # [win]
                        else:
                            bi = None

                        # 4) Forward pass (homogeneous pert in this window)
                        batch = prepare_batch(
                            ctrl_basal_np=ctrl_basal,
                            pert_onehots=pert_oh,
                            batch_indices=bi,
                            pert_names=[p] * win_size,
                            device=self.device,
                        )
                        batch_out = self.model.predict_step(
                            batch, batch_idx=0, padded=False
                        )

                        # 5) Choose output to write
                        if (
                            self.writes_to[0] == ".X"
                            and ("pert_cell_counts_preds" in batch_out)
                            and (batch_out["pert_cell_counts_preds"] is not None)
                        ):
                            preds = (
                                batch_out["pert_cell_counts_preds"]
                                .detach()
                                .cpu()
                                .numpy()
                                .astype(np.float32)
                            )  # [win, G]
                        else:
                            preds = (
                                batch_out["preds"]
                                .detach()
                                .cpu()
                                .numpy()
                                .astype(np.float32)
                            )  # [win, D]

                        # 6) Write predictions for these rows (controls included)
                        if self.writes_to[0] == ".X":
                            if preds.shape[1] == sim_X.shape[1]:
                                sim_X[idx_window, :] = preds
                            else:
                                print(
                                    f"Dimension mismatch for X (got {preds.shape[1]} vs {sim_X.shape[1]}). "
                                    f"Falling back to adata.obsm['X_state_pred']."
                                )
                                if "X_state_pred" not in adata.obsm:
                                    adata.obsm["X_state_pred"] = np.zeros(
                                        (n_total, preds.shape[1]), dtype=np.float32
                                    )
                                adata.obsm["X_state_pred"][idx_window, :] = preds
                                out_target = "obsm['X_state_pred']"
                        else:
                            if preds.shape[1] == sim_obsm.shape[1]:
                                sim_obsm[idx_window, :] = preds
                            else:
                                side_key = f"{self.writes_to[1]}_pred"
                                print(
                                    f"Dimension mismatch for obsm['{self.writes_to[1]}'] "
                                    f"(got {preds.shape[1]} vs {sim_obsm.shape[1]}). "
                                    f"Writing to adata.obsm['{side_key}'] instead."
                                )
                                if side_key not in adata.obsm:
                                    adata.obsm[side_key] = np.zeros(
                                        (n_total, preds.shape[1]), dtype=np.float32
                                    )
                                adata.obsm[side_key][idx_window, :] = preds
                                out_target = f"obsm['{side_key}']"

                        start = end  # next window

        if self.writes_to[0] == ".X":
            if out_target == "X":
                adata.X = sim_X
        else:
            if out_target == f"obsm['{self.writes_to[1]}']":
                adata.obsm[self.writes_to[1]] = sim_obsm

        output_path = self.config["output"] or self.config["adata"].replace(
            ".h5ad", "_simulated.h5ad"
        )

        if self.config["save"]:
            adata.write_h5ad(output_path)

        print("\n=== Inference complete ===")
        print(f"Input cells:         {n_total}")
        print(f"Controls simulated:  {n_controls}")
        print(f"Treated simulated:   {n_nonctl}")
        print(f"Wrote predictions to adata.{out_target}")
        print(f"Saved:               {output_path}")

        return adata

    def pick_first_present(
        self, d: "sc.AnnData", candidates: List[str]
    ) -> Optional[str]:

        for c in candidates:
            if c in d.obs:
                return c
        return None
