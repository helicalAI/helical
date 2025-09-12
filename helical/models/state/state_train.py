import logging
import os
import pickle
from os.path import exists, join
import lightning.pytorch as pl
import torch
from cell_load.data_modules import PerturbationDataModule
from cell_load.utils.modules import get_datamodule



from .model_dir.vcc_eval._evaluator import MetricsEvaluator
from .model_dir.vcc_eval.utils import split_anndata_on_celltype




import anndata
import numpy as np
import pandas as pd
from tqdm import tqdm

from .model_dir.perturb_utils.utils import (
    get_checkpoint_callbacks,
    RobustCSVLogger,
)
from .model_dir.perturb_utils.state_transition_model import (
    StateTransitionPerturbationModel,
)

from helical.models.state import trainConfig

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
torch.set_float32_matmul_precision("medium")
torch.multiprocessing.set_sharing_strategy("file_system")


# this code to do full training for the VCC dataset
class stateTransitionTrainModel:
    def __init__(
        self,
        configurer: trainConfig = trainConfig(),
    ):

        self.cfg = configurer.config
        self.run_output_dir = join(self.cfg["output_dir"], self.cfg["name"])
        os.makedirs(self.run_output_dir, exist_ok=True)
        pl.seed_everything(self.cfg["training"]["train_seed"])

        if self.cfg["data"]["kwargs"]["pert_col"] == "drugname_drugconc":
            self.cfg["data"]["kwargs"]["control_pert"] = "[('DMSO_TF', 0.0, 'uM')]"

        sentence_len = self.cfg["model"]["kwargs"]["transformer_backbone_kwargs"][
            "max_position_embeddings"
        ]

        self.data_module: PerturbationDataModule = get_datamodule(
            self.cfg["data"]["name"],
            self.cfg["data"]["kwargs"],
            batch_size=self.cfg["training"]["batch_size"],
            cell_sentence_len=sentence_len,
        )

        # we setup with None for var dims to work and then call the correct setup later for each stage
        self.data_module.setup(stage="fit")

        var_dims = self.data_module.get_var_dims()
        self.var_dims = var_dims
        self.gene_dim = (
            var_dims.get("hvg_dim", 2000)
            if self.cfg["data"]["kwargs"]["output_space"] == "gene"
            else var_dims.get("gene_dim", 2000)
        )
        latent_dim = var_dims["output_dim"]  # same as model.output_dim
        self.hidden_dims = self.cfg["model"]["kwargs"].get(
            "decoder_hidden_dims", [1024, 1024, 512]
        )

        decoder_cfg = dict(
            latent_dim=latent_dim,
            gene_dim=self.gene_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.cfg["model"]["kwargs"].get("decoder_dropout", 0.1),
            residual_decoder=self.cfg["model"]["kwargs"].get("residual_decoder", False),
        )

        # tuck it into the kwargs that will reach the LightningModule
        self.cfg["model"]["kwargs"]["decoder_cfg"] = decoder_cfg

        # create_model
        self.model_config = self.cfg["model"]["kwargs"]
        training_config = self.cfg["training"]
        data_config = self.cfg["data"]["kwargs"]
        module_config = {**self.model_config, **training_config}

        module_config["embed_key"] = data_config["embed_key"]
        module_config["output_space"] = data_config["output_space"]
        module_config["gene_names"] = var_dims["gene_names"]
        module_config["batch_size"] = training_config["batch_size"]
        module_config["control_pert"] = data_config.get("control_pert", "non-targeting")

        self.model = StateTransitionPerturbationModel(
            input_dim=self.var_dims["input_dim"],
            gene_dim=self.gene_dim,
            hvg_dim=self.var_dims["hvg_dim"],
            output_dim=self.var_dims["output_dim"],
            pert_dim=self.var_dims["pert_dim"],
            batch_dim=self.var_dims["batch_dim"],
            **module_config,
        )

        params_count = sum(
            p.numel() * p.element_size() for p in self.model.parameters()
        )
        print(
            f"Model created. Estimated params size: {params_count/ 1024**3:.2f} GB and {params_count} parameters"
        )

    def train(self):
        self.data_module.setup(stage="fit")
        # Save the onehot maps as pickle files instead of storing in config
        cell_type_onehot_map_path = join(
            self.run_output_dir, "cell_type_onehot_map.pkl"
        )
        pert_onehot_map_path = join(self.run_output_dir, "pert_onehot_map.pt")
        batch_onehot_map_path = join(self.run_output_dir, "batch_onehot_map.pkl")
        var_dims_path = join(self.run_output_dir, "var_dims.pkl")

        with open(cell_type_onehot_map_path, "wb") as f:
            pickle.dump(self.data_module.cell_type_onehot_map, f)

        torch.save(self.data_module.pert_onehot_map, pert_onehot_map_path)

        with open(batch_onehot_map_path, "wb") as f:
            pickle.dump(self.data_module.batch_onehot_map, f)

        with open(var_dims_path, "wb") as f:
            pickle.dump(self.var_dims, f)

        loggers = []
        csv_logger = RobustCSVLogger(
            save_dir=self.cfg["output_dir"], name=self.cfg["name"], version=0
        )
        loggers.append(csv_logger)

        # Set up callbacks
        callbacks = get_checkpoint_callbacks(
            self.cfg["output_dir"],
            self.cfg["name"],
            self.cfg["training"]["val_freq"],
            self.cfg["training"].get("ckpt_every_n_steps", 4000),
        )

        LOGGER.info("Loggers and callbacks set up.")

        trainer_kwargs = dict(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            max_steps=self.cfg["training"]["max_steps"],
            max_epochs=self.cfg["training"]["max_epochs"],
            check_val_every_n_epoch=None,
            val_check_interval=self.cfg["training"]["val_freq"],
            logger=loggers,
            plugins=[],
            callbacks=callbacks,
            gradient_clip_val=None,
        )
        trainer = pl.Trainer(**trainer_kwargs)
        print("Trainer built successfully")
        # Load checkpoint if exists
        checkpoint_path = os.path.join(self.run_output_dir, self.cfg["checkpoint_name"])
        if not exists(checkpoint_path):
            checkpoint_path = None
        else:
            logging.info(f"!! Resuming training from {checkpoint_path} !!")
        LOGGER.info("Starting trainer fit.")

        trainer.fit(
            self.model,
            datamodule=self.data_module,
            ckpt_path=checkpoint_path,
        )

        print("Training completed, saving final checkpoint...")

        checkpoint_path = join(self.run_output_dir, self.cfg["checkpoint_name"])
        if not exists(checkpoint_path):
            trainer.save_checkpoint(checkpoint_path)

    def predict(self):

        self.data_module.setup(stage="test")
        test_loader = self.data_module.test_dataloader()

        checkpoint_path = os.path.join(self.run_output_dir, self.cfg["checkpoint_name"])

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Could not find checkpoint at {checkpoint_path}.\nSpecify a correct checkpoint filename."
            )
        LOGGER.info("Loading model from %s", checkpoint_path)

        model_init_kwargs = {
            "input_dim": self.var_dims["input_dim"],
            "hidden_dim": self.model_config["hidden_dim"],
            "gene_dim": self.var_dims["gene_dim"],
            "hvg_dim": self.var_dims["hvg_dim"],
            "output_dim": self.var_dims["output_dim"],
            "pert_dim": self.var_dims["pert_dim"],
            **self.model_config,
        }

        self.model = StateTransitionPerturbationModel.load_from_checkpoint(
            checkpoint_path, **model_init_kwargs
        )
        self.model.eval()
        LOGGER.info("Model loaded successfully.")
        num_cells = test_loader.batch_sampler.tot_num
        output_dim = self.var_dims["output_dim"]
        LOGGER.info("Generating predictions on test set using manual loop...")

        final_preds = np.empty((num_cells, output_dim), dtype=np.float32)
        final_reals = np.empty((num_cells, output_dim), dtype=np.float32)

        store_raw_expression = (
            self.data_module.embed_key is not None
            and self.data_module.embed_key != "X_hvg"
            and self.cfg["data"]["kwargs"]["output_space"] == "gene"
        ) or (
            self.data_module.embed_key is not None
            and self.cfg["data"]["kwargs"]["output_space"] == "all"
        )

        final_X_hvg = None
        final_pert_cell_counts_preds = None
        if store_raw_expression:
            # Preallocate matrices of shape (num_cells, gene_dim) for decoded predictions.
            if self.cfg["data"]["kwargs"]["output_space"] == "gene":
                final_X_hvg = np.empty((num_cells, self.hvg_dim), dtype=np.float32)
                final_pert_cell_counts_preds = np.empty(
                    (num_cells, self.hvg_dim), dtype=np.float32
                )
            if self.cfg["data"]["kwargs"]["output_space"] == "all":
                final_X_hvg = np.empty((num_cells, self.gene_dim), dtype=np.float32)
                final_pert_cell_counts_preds = np.empty(
                    (num_cells, self.gene_dim), dtype=np.float32
                )

        current_idx = 0

        # Initialize aggregation variables directly
        all_pert_names = []
        all_celltypes = []
        all_gem_groups = []
        all_pert_barcodes = []
        all_ctrl_barcodes = []
        device = next(self.model.parameters()).device

        with torch.no_grad():
            for batch_idx, batch in enumerate(
                tqdm(test_loader, desc="Predicting", unit="batch")
            ):
                # Move each tensor in the batch to the model's device
                batch = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }

                # Get predictions
                batch_preds = self.model.predict_step(batch, batch_idx, padded=False)

                # Extract metadata and data directly from batch_preds
                # Handle pert_name
                if isinstance(batch_preds["pert_name"], list):
                    all_pert_names.extend(batch_preds["pert_name"])
                else:
                    all_pert_names.append(batch_preds["pert_name"])

                if "pert_cell_barcode" in batch_preds:
                    if isinstance(batch_preds["pert_cell_barcode"], list):
                        all_pert_barcodes.extend(batch_preds["pert_cell_barcode"])
                        all_ctrl_barcodes.extend(batch_preds["ctrl_cell_barcode"])
                    else:
                        all_pert_barcodes.append(batch_preds["pert_cell_barcode"])
                        all_ctrl_barcodes.append(batch_preds["ctrl_cell_barcode"])

                # Handle celltype_name
                if isinstance(batch_preds["celltype_name"], list):
                    all_celltypes.extend(batch_preds["celltype_name"])
                else:
                    all_celltypes.append(batch_preds["celltype_name"])

                # Handle gem_group
                if isinstance(batch_preds["batch"], list):
                    all_gem_groups.extend([str(x) for x in batch_preds["batch"]])
                elif isinstance(batch_preds["batch"], torch.Tensor):
                    all_gem_groups.extend(
                        [str(x) for x in batch_preds["batch"].cpu().numpy()]
                    )
                else:
                    all_gem_groups.append(str(batch_preds["batch"]))

                batch_pred_np = batch_preds["preds"].cpu().numpy().astype(np.float32)
                batch_real_np = (
                    batch_preds["pert_cell_emb"].cpu().numpy().astype(np.float32)
                )
                batch_size = batch_pred_np.shape[0]
                final_preds[current_idx : current_idx + batch_size, :] = batch_pred_np
                final_reals[current_idx : current_idx + batch_size, :] = batch_real_np
                current_idx += batch_size

                # Handle X_hvg for HVG space ground truth
                if final_X_hvg is not None:
                    batch_real_gene_np = (
                        batch_preds["pert_cell_counts"].cpu().numpy().astype(np.float32)
                    )
                    final_X_hvg[current_idx - batch_size : current_idx, :] = (
                        batch_real_gene_np
                    )

                # Handle decoded gene predictions if available
                if final_pert_cell_counts_preds is not None:
                    batch_gene_pred_np = (
                        batch_preds["pert_cell_counts_preds"]
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )
                    final_pert_cell_counts_preds[
                        current_idx - batch_size : current_idx, :
                    ] = batch_gene_pred_np

        LOGGER.info("Creating anndatas from predictions from manual loop...")

        # Build pandas DataFrame for obs and var
        df_dict = {
            self.data_module.pert_col: all_pert_names,
            self.data_module.cell_type_key: all_celltypes,
            self.data_module.batch_col: all_gem_groups,
        }

        if len(all_pert_barcodes) > 0:
            df_dict["pert_cell_barcode"] = all_pert_barcodes
            df_dict["ctrl_cell_barcode"] = all_ctrl_barcodes

        obs = pd.DataFrame(df_dict)

        gene_names = self.var_dims["gene_names"]
        var = pd.DataFrame({"gene_names": gene_names})

        if final_X_hvg is not None:
            if len(gene_names) != final_pert_cell_counts_preds.shape[1]:
                gene_names = np.load(
                    "/large_storage/ctc/userspace/aadduri/datasets/tahoe_19k_to_2k_names.npy",
                    allow_pickle=True,
                )
                var = pd.DataFrame({"gene_names": gene_names})

            # Create adata for predictions - using the decoded gene expression values
            adata_pred = anndata.AnnData(
                X=final_pert_cell_counts_preds, obs=obs, var=var
            )
            # Create adata for real - using the true gene expression values
            adata_real = anndata.AnnData(X=final_X_hvg, obs=obs, var=var)

            # add the embedding predictions
            adata_pred.obsm[self.data_module.embed_key] = final_preds
            adata_real.obsm[self.data_module.embed_key] = final_reals
            LOGGER.info(
                f"Added predicted embeddings to adata.obsm['{self.data_module.embed_key}']"
            )
        else:
            # if len(gene_names) != final_preds.shape[1]:
            #     gene_names = np.load(
            #         "/large_storage/ctc/userspace/aadduri/datasets/tahoe_19k_to_2k_names.npy", allow_pickle=True
            #     )
            #     var = pd.DataFrame({"gene_names": gene_names})

            # Create adata for predictions - model was trained on gene expression space already
            # adata_pred = anndata.AnnData(X=final_preds, obs=obs, var=var)
            adata_pred = anndata.AnnData(X=final_preds, obs=obs)
            # Create adata for real - using the true gene expression values
            # adata_real = anndata.AnnData(X=final_reals, obs=obs, var=var)
            adata_real = anndata.AnnData(X=final_reals, obs=obs)

        # Save the AnnData objects
        results_dir = os.path.join(self.cfg["output_dir"])

        os.makedirs(results_dir, exist_ok=True)
        adata_pred_path = os.path.join(results_dir, "adata_pred.h5ad")
        adata_real_path = os.path.join(results_dir, "adata_real.h5ad")

        adata_pred.write_h5ad(adata_pred_path)
        adata_real.write_h5ad(adata_real_path)

        LOGGER.info(f"Saved adata_pred to {adata_pred_path}")
        LOGGER.info(f"Saved adata_real to {adata_real_path}")

        if not self.cfg["predict_only"]:
            # 6. Compute metrics using cell-eval
            LOGGER.info("Computing metrics using cell-eval...")

            control_pert = self.data_module.get_control_pert()

            ct_split_real = split_anndata_on_celltype(
                adata=adata_real, celltype_col=self.data_module.cell_type_key
            )
            ct_split_pred = split_anndata_on_celltype(
                adata=adata_pred, celltype_col=self.data_module.cell_type_key
            )

            assert len(ct_split_real) == len(
                ct_split_pred
            ), f"Number of celltypes in real and pred anndata must match: {len(ct_split_real)} != {len(ct_split_pred)}"

            pdex_kwargs = dict(exp_post_agg=True, is_log1p=True)
            for ct in ct_split_real.keys():
                real_ct = ct_split_real[ct]
                pred_ct = ct_split_pred[ct]

                evaluator = MetricsEvaluator(
                    adata_pred=pred_ct,
                    adata_real=real_ct,
                    control_pert=control_pert,
                    pert_col=self.data_module.pert_col,
                    outdir=results_dir,
                    prefix=ct,
                    pdex_kwargs=pdex_kwargs,
                    batch_size=2048,
                )

                evaluator.compute(
                    profile=self.cfg["profile"],
                    metric_configs=(
                        {
                            "discrimination_score": (
                                {
                                    "embed_key": self.data_module.embed_key,
                                }
                                if self.data_module.embed_key
                                and self.data_module.embed_key != "X_hvg"
                                else {}
                            ),
                            "pearson_edistance": (
                                {
                                    "embed_key": self.data_module.embed_key,
                                    "n_jobs": -1,  # set to all available cores
                                }
                                if self.data_module.embed_key
                                and self.data_module.embed_key != "X_hvg"
                                else {
                                    "n_jobs": -1,
                                }
                            ),
                        }
                        if self.data_module.embed_key
                        and self.data_module.embed_key != "X_hvg"
                        else {}
                    ),
                    skip_metrics=["pearson_edistance", "clustering_agreement"],
                )
        return
