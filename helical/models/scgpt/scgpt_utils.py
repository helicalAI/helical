import torch

from .model_dir import TransformerModel
from .tokenizer import GeneVocab
from .utils import load_pretrained
from helical.models.scgpt.scgpt_config import scGPTConfig

def load_model(model_configs: scGPTConfig):

    # LOAD MODEL
    model_dir = model_configs["model_path"].parent
    vocab_file = model_dir / "vocab.json"
    special_tokens = [model_configs["pad_token"], "<cls>", "<eoc>"]

    # vocabulary
    vocab = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)

    # Binning will be applied after tokenization. A possible way to do is to use the unified way of binning in the data collator.

    vocab.set_default_index(vocab[model_configs["pad_token"]])

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
