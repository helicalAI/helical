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
from transformers import (
    BertForMaskedLM
)


logger = logging.getLogger(__name__)

def load_mappings(gene_symbols):
    server = "https://rest.ensembl.org"
    ext = "/lookup/symbol/homo_sapiens"
    headers={ "Content-Type" : "application/json", "Accept" : "application/json"}

    # r = requests.post(server+ext, headers=headers, data='{ "symbols" : ["BRCA2", "BRAF" ] }')

    gene_id_to_ensemble = dict()

    for i in range(0, len(gene_symbols), 1000):
        # print(i+1000,"/",len(test.var['gene_symbols']))
        symbols = {'symbols':gene_symbols[i:i+1000].tolist()}
        r = requests.post(server+ext, headers=headers, data=json.dumps(symbols))
        decoded = r.json()
        gene_id_to_ensemble.update(decoded)
        # print(repr(decoded))

    pkl.dump(gene_id_to_ensemble, open('./human_gene_to_ensemble_id.pkl', 'wb'))
    return gene_id_to_ensemble


# extract embeddings
def get_embs(
    model,
    filtered_input_data,
    emb_mode,
    layer_to_quant,
    pad_token_id,
    forward_batch_size,
    gene_token_dict,
    device,
    silent=False,
    
):
    model_input_size = get_model_input_size(model)
    total_batch_length = len(filtered_input_data)
    embs_list = []

    #  Check if CLS and EOS token is present in the token dictionary
    cls_present = any("<cls>" in key for key in gene_token_dict.keys())
    eos_present = any("<eos>" in key for key in gene_token_dict.keys())
    if emb_mode == "cls":
        assert cls_present, "<cls> token missing in token dictionary"
        # Check to make sure that the first token of the filtered input data is cls token
        cls_token_id = gene_token_dict["<cls>"]
        assert (
            filtered_input_data["input_ids"][0][0] == cls_token_id
        ), "First token is not <cls> token value"
    elif emb_mode == "cell":
        if cls_present:
            logger.warning(
                "CLS token present in token dictionary, excluding from average."
            )
        if eos_present:
            logger.warning(
                "EOS token present in token dictionary, excluding from average."
            )

    overall_max_len = 0
    for i in trange(0, total_batch_length, forward_batch_size, leave=(not silent)):
        max_range = min(i + forward_batch_size, total_batch_length)

        minibatch = filtered_input_data.select([i for i in range(i, max_range)])

        max_len = int(max(minibatch["length"]))
        original_lens = torch.tensor(minibatch["length"],device=device)
        minibatch.set_format(type="torch",device=device)

        input_data_minibatch = minibatch["input_ids"]
        input_data_minibatch = pad_tensor_list(
            input_data_minibatch, max_len, pad_token_id, model_input_size
        ).to(device)

        model = model.to(device)
        with torch.no_grad():
            outputs = model(
                input_ids=input_data_minibatch,
                attention_mask=gen_attention_mask(minibatch),
            )

        embs_i = outputs.hidden_states[layer_to_quant]

        if emb_mode == "cell":
            if cls_present:
                non_cls_embs = embs_i[:, 1:, :]  # Get all layers except the cls embs
                if eos_present:
                    mean_embs = mean_nonpadding_embs(non_cls_embs, original_lens - 2)
                else:
                    mean_embs = mean_nonpadding_embs(non_cls_embs, original_lens - 1)
            else:
                mean_embs = mean_nonpadding_embs(embs_i, original_lens)
            
            embs_list.append(mean_embs)
            del mean_embs

        elif emb_mode == "gene":
                embs_list.append(embs_i)
        
        elif emb_mode == "cls":
            cls_embs = embs_i[:, 0, :].clone().detach()  # CLS token layer
            embs_list.append(cls_embs)
            del cls_embs

        overall_max_len = max(overall_max_len, max_len)
        del outputs
        del minibatch
        del input_data_minibatch
        del embs_i

        torch.cuda.empty_cache()

    if emb_mode == "cell" or emb_mode == "cls":
        embs_stack = torch.cat(embs_list, dim=0)
    elif emb_mode == "gene":
        embs_stack = pad_tensor_list(
            embs_list,
            overall_max_len,
            pad_token_id,
            model_input_size,
            1,
            pad_3d_tensor,
        )
    return embs_stack


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
        [1] * original_len + [0] * (max_len - original_len)
        if original_len <= max_len
        else [1] * max_len
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
