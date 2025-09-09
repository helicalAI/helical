from helical.models.scgpt import (
    stateConfig, 
    stateEmbeddingsModel, 
    trainingConfig, 
    stateTransitionTrainModel, 
    stateTransitionModel
    )
import hydra
from omegaconf import DictConfig
import anndata as ad
from datasets import load_dataset
from helical.utils import get_anndata_from_hf_dataset
import scanpy as sc
from helical.models.state import vcc_eval

@hydra.main(version_base=None, config_path="configs", config_name="state_config")
def run_embeddings(cfg: DictConfig):

    state_config = stateConfig()
    state_model = stateEmbeddingsModel(configurer=state_config)

    ann_data = sc.read_h5ad("./competition_support_set/competition_val_template.h5ad")
    ann_data = ann_data[:10].copy()

    ann_data = state_model.process_data(ann_data)
    embeddings = state_model.get_embeddings(ann_data)

    print(embeddings.shape)
    return


def run_training(cfg: DictConfig):

    train_config = trainingConfig(
        output_dir="competition",
        name="first_run",
        toml_config_path="competition_support_set/starter.toml",
        checkpoint_name="final.ckpt",
        max_steps=40000,
        max_epochs=1,
        ckpt_every_n_steps=20000,
        num_workers=4,
        batch_col="batch_var",
        pert_col="target_gene",
        cell_type_key="cell_type",
        control_pert="non-targeting",
        perturbation_features_file="competition_support_set/ESM2_pert_features.pt"
        )

    state_train = stateTransitionTrainModel(configurer=train_config)
    state_train.train() 
    state_train.predict() 
    return



def run_inference(cfg: DictConfig):

    state_config = stateConfig(
        output = "competition/prediction.h5ad",
        model_dir = "competition/first_run",
        model_config = "configs/config.yaml",
        pert_col = "target_gene",
    )

    adata = sc.read_h5ad("competition_support_set/competition_val_template.h5ad")

    state_transition = stateTransitionModel(configurer=state_config)
    adata = state_transition.process_data(adata)
    embeds = state_transition.get_embeddings(adata)
    return

def run_vcc_eval(cfg: DictConfig):
    # default configs for competition dataset
    EXPECTED_GENE_DIM = 18080
    MAX_CELL_DIM = 100000
    DEFAULT_PERT_COL = "target_gene"
    DEFAULT_CTRL = "non-targeting"
    DEFAULT_COUNTS_COL = "n_cells"
    DEFAULT_CELLTYPE_COL = "celltype"
    DEFAULT_NTC_NAME = "non-targeting"

    configs = {
        # path to the prediction file
        "input": "competition/prediction.h5ad",
        # path to the gene names file
        "genes": "competition_support_set/gene_names.csv",
        # path to the output file - if None will be created with default naming
        "output": None,
        "pert_col": DEFAULT_PERT_COL,
        "celltype_col": None,
        "ntc_name": DEFAULT_NTC_NAME,
        "output_pert_col": DEFAULT_PERT_COL,
        "output_celltype_col": DEFAULT_CELLTYPE_COL,
        "encoding": 32,
        "allow_discrete": False,
        "expected_gene_dim": EXPECTED_GENE_DIM,
        "max_cell_dim": MAX_CELL_DIM,
    }

    # this creates a submission file in the output directory which can be uploaded to the challenge leaderboard
    vcc_eval(configs)
    return 

if __name__ == "__main__":
    run_embeddings()
    run_training()
    run_inference()
    run_vcc_eval()
