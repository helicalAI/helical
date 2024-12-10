import torch
from .model_dir import TransformerModel
from .utils import load_pretrained
from helical.models.scgpt.scgpt_config import scGPTConfig
import json

def load_model(model_configs: scGPTConfig):

    # load model and vocabulary
    model_dir = model_configs["model_path"].parent
    vocab_file = model_dir / "vocab.json"

    # vocabulary
    with vocab_file.open("r") as f:
        vocab = json.load(f)

    model = TransformerModel(
        ntoken=len(vocab),
        d_model=model_configs["embsize"],
        nhead=model_configs["nheads"],
        d_hid=model_configs["d_hid"],
        nlayers=model_configs["nlayers"],
        nlayers_cls=model_configs["n_layers_cls"],
        n_cls=1,
        vocab=vocab,
        dropout=model_configs["dropout"],
        pad_token=model_configs["pad_token"],
        pad_value=model_configs["pad_value"],
        do_mvc=True,
        do_dab=False,
        use_batch_labels=False,
        domain_spec_batchnorm=False,
        explicit_zero_prob=False,
        use_fast_transformer=model_configs["use_fast_transformer"],
        fast_transformer_backend="flash",
        pre_norm=False,
    )

    load_pretrained(model, torch.load(model_configs["model_path"], map_location = model_configs["device"]), verbose = False)
    model.to(model_configs["device"])
    model.eval()
    return model, vocab
