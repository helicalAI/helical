import json
import logging
import os
import numpy as np
import torch

from helical.models.transcriptformer.utils.utils import load_embeddings

SPECIAL_TOKENS = ["unknown", "[START]", "[END]", "[RD]", "[CELL]", "[PAD]", "[MASK]"]


def construct_gene_embeddings(
    pretrained_embedding_paths: list,
    special_tokens: list,
    random_seed: int = 42,  # Add random seed parameter with default value
) -> tuple[dict, np.ndarray]:
    """Construct gene embeddings from pretrained embeddings.

    Args:
        pretrained_embedding_paths (list): List of paths to pretrained embeddings.
        special_tokens (list): List of special tokens.
        random_seed (int, optional): Random seed for special token embeddings. Defaults to 42.

    Returns
    -------
        tuple: A tuple containing:
            - gene_vocab_dict (dict): A dictionary mapping gene names to indices.
            - embeddings (np.ndarray):
    """
    gene_names = []
    embeddings = []
    for file in pretrained_embedding_paths:
        gene_emb = load_embeddings(file)
        gene_names += list(gene_emb.keys())
        embeddings += list(gene_emb.values())

    # Check for duplicates in gene names
    seen_genes = {}
    deduped_names = []
    deduped_embeddings = []
    for i, gene in enumerate(gene_names):
        if gene in seen_genes:
            logging.warning(f"Duplicate gene name found: {gene}, removing duplicate")
        else:
            deduped_names.append(gene)
            deduped_embeddings.append(embeddings[i])
            seen_genes[gene] = True

    gene_vocab_dict = build_gene_vocab_from_list(deduped_names, special_tokens)
    emb_dim = embeddings[0].shape[0]

    # Set random seed before generating special token embeddings
    rng = np.random.RandomState(random_seed)
    special_tokens_emb = rng.rand(len(special_tokens), emb_dim)

    return gene_vocab_dict, np.concatenate(
        [special_tokens_emb, np.array(deduped_embeddings)], axis=0
    )


def load_vocabs_and_embeddings(cfg):
    if cfg.model.data_config.aux_cols is not None:
        aux_cols = cfg.model.data_config.aux_cols.split(",")
    else:
        aux_cols = None

    # Load vocabularies
    aux_vocab = (
        open_vocabs(cfg.model.data_config.aux_vocab_path, aux_cols)
        if aux_cols is not None
        else None
    )

    # Load embeddings
    logging.info(
        f"Loading ESM2 mappings from {cfg.model.data_config.esm2_mappings_path}"
    )
    emb_files = []
    for file in cfg.model.data_config.esm2_mappings:
        emb_files.append(os.path.join(cfg.model.data_config.esm2_mappings_path, file))

    gene_vocab, emb_matrix = construct_gene_embeddings(
        emb_files,
        cfg.model.data_config.special_tokens,
    )

    emb_matrix = torch.tensor(emb_matrix)
    assert emb_matrix.shape[0] == len(
        gene_vocab
    ), f"Embeddings matrix has wrong shape, {emb_matrix.shape[0]} != {len(gene_vocab)}"

    return (gene_vocab, aux_vocab), emb_matrix


def build_gene_vocab_from_list(gene_names, special_tokens=None):
    logging.info("Building gene vocabulary")
    if special_tokens is None:
        special_tokens = SPECIAL_TOKENS
    gene_vocab = {token: idx for idx, token in enumerate(special_tokens)}
    gene_ids = range(len(gene_vocab), len(gene_names) + len(gene_vocab))
    gene_vocab.update(dict(zip(gene_names, gene_ids, strict=False)))
    return gene_vocab


def open_vocabs(path, cols_to_load=None, verbose=True):
    if isinstance(cols_to_load, str):
        cols_to_load = [cols_to_load]
    vocabs = {}
    for file_name in os.listdir(path):
        if file_name.endswith("_vocab.json"):
            variable_name = file_name.split("_vocab.json")[0]
            if cols_to_load is not None and variable_name not in cols_to_load:
                continue
            if verbose:
                logging.info(
                    f"Loading vocabulary file: {os.path.join(path, variable_name)}"
                )
            with open(os.path.join(path, file_name)) as f:
                vocabs[variable_name] = json.load(f)
    if len(vocabs) == 0:
        raise ValueError("No vocabularies found in the specified path")
    elif cols_to_load == ["gene"]:
        return list(vocabs.values()).pop()
    return vocabs
