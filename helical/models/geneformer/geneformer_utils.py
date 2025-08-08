import pickle as pkl
import requests
import json
import pickle as pkl

# imports
import logging
import torch
from tqdm.auto import trange
import re
import torch
from transformers import BertForMaskedLM
import pandas as pd
import numpy as np


logger = logging.getLogger(__name__)


def load_mappings(gene_symbols):
    server = "https://rest.ensembl.org"
    ext = "/lookup/symbol/homo_sapiens"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    # r = requests.post(server+ext, headers=headers, data='{ "symbols" : ["BRCA2", "BRAF" ] }')

    gene_id_to_ensemble = dict()

    for i in range(0, len(gene_symbols), 1000):
        # print(i+1000,"/",len(test.var['gene_symbols']))
        symbols = {"symbols": gene_symbols[i : i + 1000].tolist()}
        r = requests.post(server + ext, headers=headers, data=json.dumps(symbols))
        decoded = r.json()
        gene_id_to_ensemble.update(decoded)
        # print(repr(decoded))

    pkl.dump(gene_id_to_ensemble, open("./human_gene_to_ensemble_id.pkl", "wb"))
    return gene_id_to_ensemble


def _compute_embeddings_depending_on_mode(
    embeddings: torch.tensor,
    data_dict: dict,
    emb_mode: str,
    cls_present: bool,
    eos_present: bool,
    token_to_ensembl_dict: dict,
):
    """
    Compute the different embeddings for each emb_mode

    Parameters
    -----------
    embeddings: torch.tensor
        The embedding batch output by the model.
    data_dict: dict
        The minibatch data dictionary used an input to the model.
    emb_mode: str
        The mode in which the embeddings are to be computed.
    cls_present: bool
        Whether the <cls> token is present in the token dictionary.
    eos_present: bool
        Whether the <eos> token is present in the token dictionary.
    token_to_ensembl_dict: dict
        The token to ensemble dictionary from the .
    """
    if emb_mode == "cell":
        length = data_dict["length"]
        if cls_present:
            embeddings = embeddings[:, 1:, :]  # Get all layers except the cls embs
            if eos_present:
                length -= 2  # length is used for the mean calculation, 2 is subtracted because we have taken both the cls and eos embeddings out
            else:
                length -= 1  # length is subtracted because just the cls is removed

        batch_embeddings = mean_nonpadding_embs(embeddings, length).cpu().numpy()

    elif emb_mode == "gene":
        if cls_present:
            embeddings = embeddings[:, 1:, :]
            if eos_present:
                embeddings = embeddings[:, :-1, :]

        batch_embeddings = []
        for embedding, ids in zip(embeddings, data_dict["input_ids"]):
            cell_dict = {}
            if cls_present:
                ids = ids[1:]
                if eos_present:
                    ids = ids[:-1]
            for id, gene_emb in zip(ids, embedding):
                cell_dict[token_to_ensembl_dict[id.item()]] = gene_emb.cpu().numpy()

            batch_embeddings.append(pd.Series(cell_dict))

    elif emb_mode == "cls":
        batch_embeddings = embeddings[:, 0, :].cpu().numpy()  # CLS token layer

    return batch_embeddings


def _check_for_expected_special_tokens(
    dataset, emb_mode, cls_present, eos_present, gene_token_dict
):
    """
    Check for the expected special tokens in the dataset.

    Parameters
    -----------
    dataset: dict
        The batch dictionary with input ids.
    emb_mode: str
        The mode in which the embeddings are to be computed.
    cls_present: bool
        Whether the <cls> token is present in the token dictionary.
    eos_present: bool
        Whether the <eos> token is present in the token dictionary.
    gene_token_dict: dict
        The gene token dictionary from the tokenizer.
    """
    if emb_mode == "cls":
        message = "<cls> token missing in token dictionary"
        if not cls_present:
            logger.error(message)
            raise ValueError(message)

        if dataset["input_ids"][0][0] != gene_token_dict["<cls>"]:
            message = "First token is not <cls> token value"
            logger.error(message)
            raise ValueError(message)

    elif emb_mode == "cell":
        if cls_present:
            logger.warning(
                "CLS token present in token dictionary, excluding from average."
            )
        if eos_present:
            logger.warning(
                "EOS token present in token dictionary, excluding from average."
            )


# extract embeddings
def get_embs(
    model,
    filtered_input_data,
    emb_mode,
    layer_to_quant,
    pad_token_id,
    forward_batch_size,
    gene_token_dict,
    token_to_ensembl_dict,
    cls_present,
    eos_present,
    device,
    silent=False,
    output_attentions=False,
    output_genes=False
):
    model_input_size = get_model_input_size(model)
    total_batch_length = len(filtered_input_data)
    embs_list = []
    attn_list = []
    input_genes = []

    _check_for_expected_special_tokens(
        filtered_input_data, emb_mode, cls_present, eos_present, gene_token_dict
    )

    overall_max_len = 0
    for i in trange(0, total_batch_length, forward_batch_size, leave=(not silent)):
        max_range = min(i + forward_batch_size, total_batch_length)

        minibatch = filtered_input_data.select([i for i in range(i, max_range)])

        max_len = int(max(minibatch["length"]))
        minibatch.set_format(type="torch", device=device)

        input_data_minibatch = minibatch["input_ids"]
        input_data_minibatch = pad_tensor_list(
            input_data_minibatch, max_len, pad_token_id, model_input_size
        ).to(device)

        model = model.to(device)
        with torch.no_grad():
            outputs = model(
                input_ids=input_data_minibatch,
                attention_mask=gen_attention_mask(minibatch),
                output_attentions=output_attentions,
            )

        embs_i = outputs.hidden_states[layer_to_quant]
        # attention of size (batch_size, num_heads, sequence_length, sequence_length)
        if output_attentions:
            attn_i = outputs.attentions[layer_to_quant]
            # attn_i = torch.mean(attn_i, dim=1).cpu().numpy()  # average over heads
            attn_list.extend(attn_i.cpu().numpy())

        if output_genes:
            for input_ids in minibatch["input_ids"]:
                gene_list = []
                for id in input_ids:
                    gene_list.append(token_to_ensembl_dict[id.item()])
                input_genes.append(gene_list)

        embs_list.extend(
            _compute_embeddings_depending_on_mode(
                embs_i,
                minibatch,
                emb_mode,
                cls_present,
                eos_present,
                token_to_ensembl_dict,
            )
        )

        overall_max_len = max(overall_max_len, max_len)
        del outputs
        del minibatch
        del input_data_minibatch
        del embs_i

        torch.cuda.empty_cache()
    if emb_mode != "gene":
        embs_list = np.array(embs_list)

    if output_attentions:
        if output_genes:
            return embs_list, attn_list, input_genes
        return embs_list, attn_list
    
    if output_genes:
        return embs_list, input_genes
    
    return embs_list


def downsample_and_sort(data, max_ncells):
    num_cells = len(data)
    # if max number of cells is defined, then shuffle and subsample to this max number
    if max_ncells is not None:
        if num_cells > max_ncells:
            data = data.shuffle(seed=42)
            num_cells = max_ncells
    data_subset = data.select([i for i in range(num_cells)])
    # sort dataset with largest cell first to encounter any memory errors earlier
    data_sorted = data_subset.sort("length", reverse=True)
    return data_sorted


def quant_layers(model):
    layer_nums = []
    for name, parameter in model.named_parameters():
        if "layer" in name:
            layer_nums += [int(name.split("layer.")[1].split(".")[0])]
    return int(max(layer_nums)) + 1


def get_model_input_size(model):
    return int(re.split("\(|,", str(model.bert.embeddings.position_embeddings))[1])


def load_model(model_type, model_directory, device):
    if model_type == "Pretrained":
        model = BertForMaskedLM.from_pretrained(
            model_directory, output_hidden_states=True, output_attentions=False
        )
    # put the model in eval mode for fwd pass and load onto the GPU if available
    model.eval()
    model = model.to(device)
    return model


def pad_tensor(tensor, pad_token_id, max_len):
    tensor = torch.nn.functional.pad(
        tensor, pad=(0, max_len - tensor.numel()), mode="constant", value=pad_token_id
    )

    return tensor


def pad_3d_tensor(tensor, pad_token_id, max_len, dim):
    if dim == 0:
        raise Exception("dim 0 usually does not need to be padded.")
    if dim == 1:
        pad = (0, 0, 0, max_len - tensor.size()[dim])
    elif dim == 2:
        pad = (0, max_len - tensor.size()[dim], 0, 0)
    tensor = torch.nn.functional.pad(
        tensor, pad=pad, mode="constant", value=pad_token_id
    )
    return tensor


# pad list of tensors and convert to tensor
def pad_tensor_list(
    tensor_list,
    dynamic_or_constant,
    pad_token_id,
    model_input_size,
    dim=None,
    padding_func=None,
):
    # determine maximum tensor length
    if dynamic_or_constant == "dynamic":
        max_len = max([tensor.squeeze().numel() for tensor in tensor_list])
    elif isinstance(dynamic_or_constant, int):
        max_len = dynamic_or_constant
    else:
        max_len = model_input_size
        logger.warning(
            "If padding style is constant, must provide integer value. "
            f"Setting padding to max input size {model_input_size}."
        )

    # pad all tensors to maximum length
    if dim is None:
        tensor_list = [
            pad_tensor(tensor, pad_token_id, max_len) for tensor in tensor_list
        ]
    else:
        tensor_list = [
            padding_func(tensor, pad_token_id, max_len, dim) for tensor in tensor_list
        ]
    # return stacked tensors
    if padding_func != pad_3d_tensor:
        return torch.stack(tensor_list)
    else:
        return torch.cat(tensor_list, 0)


def gen_attention_mask(minibatch_encoding, max_len=None):
    if max_len is None:
        max_len = max(minibatch_encoding["length"])
    original_lens = minibatch_encoding["length"]
    attention_mask = [
        (
            [1] * original_len + [0] * (max_len - original_len)
            if original_len <= max_len
            else [1] * max_len
        )
        for original_len in original_lens
    ]
    return torch.tensor(attention_mask, device=minibatch_encoding["length"].device)


# get cell embeddings excluding padding
def mean_nonpadding_embs(embs, original_lens, dim=1):
    # create a mask tensor based on padding lengths
    mask = torch.arange(embs.size(dim), device=embs.device) < original_lens.unsqueeze(1)
    if embs.dim() == 3:
        # fill the masked positions in embs with zeros
        masked_embs = embs.masked_fill(~mask.unsqueeze(2), 0.0)

        # compute the mean across the non-padding dimensions
        mean_embs = masked_embs.sum(dim) / original_lens.view(-1, 1).float()

    elif embs.dim() == 2:
        masked_embs = embs.masked_fill(~mask, 0.0)
        mean_embs = masked_embs.sum(dim) / original_lens.float()
    return mean_embs
