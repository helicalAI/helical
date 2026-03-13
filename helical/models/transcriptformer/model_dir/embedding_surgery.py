import torch

from helical.models.transcriptformer.model_dir.model import Transcriptformer
from helical.models.transcriptformer.tokenizer.vocab import (
    SPECIAL_TOKENS,
    construct_gene_embeddings,
)


def change_embedding_layer(
    model: Transcriptformer,
    pretrained_embedding_paths: list[str],
    special_tokens: list | None = None,
):
    """
    Change the embedding layer of a model to a new embedding layer.

    Args:
        model (torch.nn.Module): The model to change the embedding layer of.
        pretrained_embedding_paths (list[str]): The paths to the pretrained embedding files.
        special_tokens (Optional[list]): The special tokens to add to the model.


    Returns
    -------
        torch.nn.Module: The model with the new embedding layer.
    """
    if special_tokens is None:
        special_tokens = SPECIAL_TOKENS

    old_special_tokens = [i for i in special_tokens if i in model.gene_vocab.vocab_dict]

    # Extract the rows of the embedding matrix that correspond to the special tokens
    old_special_token_indices = torch.tensor(
        [model.gene_vocab.vocab_dict[token] for token in old_special_tokens],
        dtype=torch.int64,
    )
    # Move model to GPU - needed if model is not re-loaded from checkpoint
    model = model.to("cuda")
    old_special_token_embeddings = model.gene_embeddings.embedding(
        old_special_token_indices.to("cuda")
    )

    # Read the new embeddings from the files
    gene_vocab, new_embedding_matrix = construct_gene_embeddings(
        pretrained_embedding_paths,
        [token for token in special_tokens if token not in old_special_tokens],
    )

    # Build a special-token embedding matrix that preserves the original indices.
    # The training vocab may assign special tokens to indices that differ from
    # the order they appear in the SPECIAL_TOKENS list, so we must place each
    # token's embedding at its original row rather than re-numbering them.
    n_special = len(old_special_tokens)
    special_emb_matrix = torch.zeros(
        n_special, old_special_token_embeddings.shape[1], device="cuda"
    )
    for i, token in enumerate(old_special_tokens):
        orig_idx = model.gene_vocab.vocab_dict[token]
        special_emb_matrix[orig_idx] = old_special_token_embeddings[i]

    # Concatenate the (correctly ordered) special token embeddings with the new gene embeddings
    new_embedding_matrix = torch.cat(
        [special_emb_matrix, torch.Tensor(new_embedding_matrix).to("cuda")],
        dim=0,
    )

    # Update the vocab indices of the model
    gene_vocab = {
        gene: idx + n_special for gene, idx in gene_vocab.items()
    }

    # Restore special tokens at their original indices from the training vocab
    gene_vocab.update({token: model.gene_vocab.vocab_dict[token] for token in old_special_tokens})

    # Create a new embedding layer
    new_embedding = torch.nn.Embedding(
        len(gene_vocab),
        new_embedding_matrix.shape[1],
        padding_idx=model.gene_vocab.pad_idx,
    )

    # Set device to be the same as the model's embedding layer
    new_embedding = new_embedding.to(model.gene_embeddings.embedding.weight.device)

    # Assign the weights to the new embedding layer, from gene_vocab
    new_embedding.weight.data.copy_(new_embedding_matrix)

    # Replace the old embedding layer with the new one
    model.gene_embeddings.embedding = new_embedding

    # Update the gene_vocab
    model.gene_vocab.vocab_dict = gene_vocab
    model.gene_vocab.embedding_matrix = new_embedding_matrix
    model.token_to_gene_dict = {v: k for k, v in gene_vocab.items()}

    return model, gene_vocab
