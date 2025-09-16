from typing import Dict, Optional, List
import os
import pickle
import warnings
import numpy as np
import scanpy as sc
import torch
from tqdm import tqdm
from .model_dir.perturb_utils.state_transition_model import (
    StateTransitionPerturbationModel,
)
from .state_config import StateConfig
from .model_dir.perturb_utils.utils import (
    to_dense,
    argmax_index_from_any,
    pad_adata_with_tsv,
    prepare_batch,
)
from helical.models.base_models import HelicalBaseFoundationModel
from helical.utils.downloader import Downloader
import logging

LOGGER = logging.getLogger(__name__)

# this class is used to do inference on new data using the transition model
class StateTransitionModel(HelicalBaseFoundationModel):
    def __init__(self, configurer: StateConfig = None) -> None:
        super().__init__()
        if configurer is None:
            configurer = StateConfig()
        self.config = configurer.config

        downloader = Downloader()
        for file in self.config["perturbation_files_to_download"]:
            downloader.download_via_name(file)

        self.pert_onehot_map_path, self.batch_onehot_map_path, self.checkpoint_path = [
            os.path.join(self.config["perturb_dir"], f) for f in ["pert_onehot_map.pt", "batch_onehot_map.pkl", "ST_all.pt"]
        ]

        model_configs = torch.load(self.checkpoint_path)

        self.input_params = model_configs["params"]
        self.pert_dim = self.input_params.get("pert_dim")
        self.batch_dim = self.input_params.get("batch_dim", None)
        self.model = StateTransitionPerturbationModel(**self.input_params)
        self.model.load_state_dict(model_configs["state_dict"])
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.cell_set_len = (
            self.config["max_set_len"]
            if self.config["max_set_len"] is not None
            else getattr(self.model, "cell_sentence_len", 256)
        )
        self.uses_batch_encoder = getattr(self.model, "batch_encoder", None) is not None
        
        self.output_space = getattr(
            self.model,
            "output_space",
            self.input_params.get("output_space", "gene"),
        )

        LOGGER.info(f"Checkpoint: {self.checkpoint_path}, Device: {self.device}")
        LOGGER.info(f"Cell set length (max sequence length): {self.cell_set_len}")
        LOGGER.info(f"Batch encoder: {bool(self.uses_batch_encoder)}")
        LOGGER.info(f"Output space: {self.output_space}")

    def process_data(self, adata: sc.AnnData):

        control_pert = self.config["control_pert"]

        if control_pert is None:
            try:
                control_pert = self.input_params["control_pert"]
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
                ct_from_cfg = self.input_params.get(
                    "cell_type_key", None
                )
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
                LOGGER.info(f"Grouping by cell type column: {celltype_col}")
            else:
                LOGGER.info("Grouping by cell type column: not found; no grouping")

        # choose batch column
        if self.config["batch_col"] is None:
            try:
                self.config["batch_col"] = self.input_params.get("batch_col", None)
            except Exception:
                self.config["batch_col"] = None

        if self.config["tsv"]:
            LOGGER.info(f"==> TSV padding mode: loading {self.config['tsv']}")
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

            LOGGER.info(
                f"Filtered to {adata.n_obs} cells (from {n0}) "
                f"for cell types: {keep_cts}"
            )

        # select features: embeddings or genes
        if self.config["embed_key"] is None:
            # using raw X as input features --> G dims
            self.X_in = to_dense(adata.X)  # [N, E_in]
            LOGGER.info("Using adata.X as input features")
        else:
            if self.config["embed_key"] not in adata.obsm:
                raise KeyError(
                    f"Embedding key '{self.config['embed_key']}' "
                    f"not found in adata.obsm"
                )
            # using embeddings as input features --> D dims
            self.X_in = np.asarray(adata.obsm[self.config["embed_key"]])
            LOGGER.info(
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
                        LOGGER.info(
                            f"Warning: {misses} / {len(raw_labels)} "
                            f"batch labels not found in saved mapping;"
                            f"using index 0 as fallback."
                        )
                    self.batch_indices_all = idxs
            else:
                LOGGER.info(
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
        LOGGER.info(
            f"Cells: total={n_total}, control={n_controls}, " f"non-control={n_nonctl}"
        )

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
            default_pert_vec = (
                pert_onehot_map[self.config["control_pert"]].float().clone()
            )
        else:
            default_pert_vec = torch.zeros(self.pert_dim, dtype=torch.float32)
            if self.pert_dim and self.pert_dim > 0:
                default_pert_vec[0] = 1.0

        LOGGER.info(
            "Running virtual experiment (homogeneous per-perturbation "
            "forward passes; controls included)..."
        )
        
        embeddings = None
        with torch.no_grad():
            for g in unique_groups:
                grp_idx = np.where(group_labels == g)[0]
                if len(grp_idx) == 0:
                    continue

                # control pool for this group (fallback to global if empty)
                grp_ctrl_pool = group_control_indices(g)
                if len(grp_ctrl_pool) == 0:
                    LOGGER.info(
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
                        LOGGER.info(
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
                            # self.writes_to[0] == ".X"
                            self.config["embed_key"] is None
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

                        if embeddings is None:
                            embeddings = np.zeros((n_total, preds.shape[1]), dtype=np.float32)
                        embeddings[idx_window, :] = preds
                        start = end  # next window

        key = "perturbed_embeds" if self.config["embed_key"] is None else f"{self.config['embed_key']}_perturbed_embeds"
        adata.obsm[key] = embeddings
        adata.write_h5ad(self.config["output_path"])
        
        LOGGER.info(f"--Complete--\nInput cells: {n_total}, Control simulated: {n_controls}, Treated simulated: {n_nonctl}")
        LOGGER.info(f"Wrote predictions to adata.obsm.{key} in output file")

        return embeddings

    def pick_first_present(self, d: "sc.AnnData", candidates: List[str]) -> Optional[str]:
        for c in candidates:
            if c in d.obs:
                return c
        return None
