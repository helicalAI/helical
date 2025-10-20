import pandas as pd
import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity


def get_embedding_cfg(cfg):
    return cfg["embeddings"][cfg["embeddings"]["current"]]


def get_dataset_cfg(cfg):
    return cfg["dataset"][cfg["dataset"]["current"]]


def get_precision_config(device_type="cuda"):
    """
    Single source of truth for precision configuration.

    Args:
        device_type: Device type ('cuda' or 'cpu')

    Returns:
        torch.dtype: The precision to use for autocast and model operations.
                    Returns torch.bfloat16 for CUDA, torch.float32 for CPU.
    """
    if device_type == "cuda":
        return torch.bfloat16
    else:
        return torch.float32


def compute_pearson_delta(pred, true, ctrl, ctrl_true):
    """
    pred, true, ctrl, ctrl_true are numpy arrays of shape [n_cells, n_genes],
    or you can pass means if you prefer.

    We'll compute the correlation of (pred.mean - ctrl.mean) with (true.mean - ctrl_true.mean).
    """
    return pearsonr(
        pred.mean(axis=0) - ctrl.mean(axis=0),
        true.mean(axis=0) - ctrl_true.mean(axis=0),
    )[0]


def compute_perturbation_ranking_score(
    adata_pred, adata_real, pert_col="gene", ctrl_pert="non-targeting"
):
    """
    1) compute mean perturbation effect for each perturbation in pred and real
    2) measure how well the real perturbation matches the predicted one by rank
    returns the mean normalized rank of the true perturbation
    """
    # Step 1: compute mean effects
    mean_real_effect = _compute_mean_perturbation_effect(
        adata_real, pert_col, ctrl_pert
    )
    mean_pred_effect = _compute_mean_perturbation_effect(
        adata_pred, pert_col, ctrl_pert
    )
    all_perts = mean_real_effect.index.values

    ranks = []
    for pert in all_perts:
        real_effect = mean_real_effect.loc[pert].values.reshape(1, -1)
        pred_effects = mean_pred_effect.values

        # Compute pairwise cosine similarities
        similarities = cosine_similarity(real_effect, pred_effects).flatten()
        # where is the true row? (the same index in `all_perts`)
        true_pert_index = np.where(all_perts == pert)[0][0]

        # Sort by descending similarity
        sorted_indices = np.argsort(similarities)[::-1]
        # rank of the correct one:
        rank_of_true_pert = np.where(sorted_indices == true_pert_index)[0][0]
        ranks.append(rank_of_true_pert)

    mean_rank = np.mean(ranks) / len(all_perts)
    return mean_rank


def _compute_mean_perturbation_effect(
    adata, pert_col="gene", ctrl_pert="non-targeting"
):
    """
    Helper to compute the mean effect (abs difference from control) for each perturbation.
    Actually we do the absolute difference from control row.
    """
    # shape: adata.X is (#cells, #genes)
    X = adata.X
    if not isinstance(X, np.ndarray):
        X = X.toarray()
    df = pd.DataFrame(X)
    df[pert_col] = adata.obs[pert_col].values
    mean_df = df.groupby(pert_col).mean(numeric_only=True)
    # difference from control
    return np.abs(mean_df - mean_df.loc[ctrl_pert])


def compute_gene_overlap_cross_pert(
    DE_pred, DE_true, control_pert="non-targeting", k=50
):
    all_overlaps = {}
    for c_gene in DE_pred.index:
        if c_gene == control_pert:
            continue
        all_overlaps[c_gene] = (
            len(
                set(DE_true.loc[c_gene].values).intersection(
                    set(DE_pred.loc[c_gene].values)
                )
            )
            / k
        )

    return all_overlaps


def get_shapes_dict(dataset_path, filter_by_species=None):
    datasets_df = pd.read_csv(dataset_path)

    if filter_by_species is not None:
        datasets_df = datasets_df[datasets_df["species"] == filter_by_species]

    sorted_dataset_names = sorted(datasets_df["names"])
    datasets_df = datasets_df.drop_duplicates()  ## TODO: there should be no duplicates

    de_data_available = "groupid_for_de" in datasets_df.columns

    shapes_dict = {}
    dataset_path_map = {}
    dataset_group_map = {}  # Name of the obs column to be used for retrieing DE scrores

    shapes_dict["dev_immune_mouse"] = (443697, 4786)
    shapes_dict["dev_immune_human"] = (34009, 5566)
    shapes_dict["intestinal_tract_human"] = (69668, 5192)
    shapes_dict["gtex_human"] = (18511, 7109)
    shapes_dict["gut_endoderm_mouse"] = (113043, 6806)
    shapes_dict["luca"] = (249591, 7196)
    shapes_dict.update(
        {
            "madissoon_novel_lung": (190728, 8000),
            "flores_cerebellum_human": (20232, 8000),
            "osuch_gut_human": (272310, 8000),
            "msk_ovarian_human": (929690, 8000),
            "htan_vmuc_dis_epi_human": (65084, 8000),
            "htan_vmuc_val_epi_human": (57564, 8000),
            "htan_vmuc_non_epi_human": (9099, 8000),
            "hao_pbmc_3p_human": (161764, 8000),
            "hao_pbmc_5p_human": (49147, 8000),
            "gao_tumors_human": (36111, 8000),
            "swabrick_breast_human": (92427, 8000),
            "wu_cryo_tumors_human": (105662, 8000),
            "cell_line_het_human": (53513, 8000),
            "bi_allen_metastasis_human": (27787, 8000),
            "zheng68k_human": (68579, 8000),
            "zheng68k_12k_human": (68579, 12000),
            "mouse_embryo_ct": (153597, 12000),
            "regev_gtex_heart": (36574, 8000),
            "tabula_sapiens_heart": (11505, 8000),
            "10k_pbmcs": (11990, 12000),
            "epo_ido": (35834, 12000),
            "tabula_sapiens_kidney": (9641, 8000),
            "tabula_microcebus_kidney": (14592, 8000),
            "tabula_muris_kidney": (2781, 8000),
            "tabula_muris_senis_kidney": (19610, 8000),
            "immune_human": (33506, 8000),
        }
    )

    shapes_dict["zyl_sanes_glaucoma_pig"] = (5901, 6819)
    shapes_dict["parkinsons_macaF"] = (1062, 5103)

    for row in datasets_df.iterrows():
        ngenes = row[1].num_genes
        ncells = row[1].num_cells
        name = row[1].names
        dataset_path_map[name] = row[1].path
        if de_data_available:
            dataset_group_map[name] = datasets_df.set_index("names").loc[name][
                "groupid_for_de"
            ]
        else:
            # This is for backward compatibility with old datasets CSV
            dataset_group_map[name] = "leiden"

        if not np.isnan(ngenes):
            shapes_dict[name] = (int(ncells), int(ngenes))
        else:
            shapes_dict[name] = (int(ncells), 8000)

    return (
        datasets_df,
        sorted_dataset_names,
        shapes_dict,
        dataset_path_map,
        dataset_group_map,
    )
