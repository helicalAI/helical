import json
import os
import torch
from transformers import PreTrainedModel
import re
from .standalone_hyenadna import HyenaDNAModel
import logging 

LOGGER = logging.getLogger(__name__)

# helper 1
def inject_substring(orig_str):
    """Hack to handle matching keys between models trained with and without
    gradient checkpointing."""

    # modify for mixer keys
    pattern = r"\.mixer"
    injection = ".mixer.layer"

    modified_string = re.sub(pattern, injection, orig_str)

    # modify for mlp keys
    pattern = r"\.mlp"
    injection = ".mlp.layer"

    modified_string = re.sub(pattern, injection, modified_string)

    return modified_string    

# helper 2
def load_weights(scratch_dict, pretrained_dict, checkpointing=False):
        """Loads pretrained (backbone only) weights into the scratch state dict."""

        # loop thru state dict of scratch
        # find the corresponding weights in the loaded model, and set it

        # need to do some state dict "surgery"
        for key, value in scratch_dict.items():
            if 'backbone' in key:
                # the state dicts differ by one prefix, '.model', so we add that
                key_loaded = 'model.' + key
                # breakpoint()
                # need to add an extra ".layer" in key
                if checkpointing:
                    key_loaded = inject_substring(key_loaded)
                try:
                    scratch_dict[key] = pretrained_dict[key_loaded]
                except:
                    raise Exception('key mismatch in the state dicts!')

        # scratch_dict has been updated
        return scratch_dict

class HyenaDNAPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    base_model_prefix = "hyenadna"

    def __init__(self):
        pass

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    @classmethod
    def from_pretrained(cls,
                        config,
                        device='cpu',
                        use_head=False,
                        n_classes=2,
                      ):
        scratch_model = HyenaDNAModel(**config, use_head=use_head, n_classes=n_classes)  # the new model format
        loaded_ckpt = torch.load(config["model_path"], map_location=torch.device(device))

        # need to load weights slightly different if using gradient checkpointing
        if config.get("checkpoint_mixer", False):
            checkpointing = config["checkpoint_mixer"] == True or config["checkpoint_mixer"] == True
        else:
            checkpointing = False

        # grab state dict from both and load weights
        state_dict = load_weights(scratch_model.state_dict(), loaded_ckpt['state_dict'], checkpointing=checkpointing)

        # scratch model has now been updated
        scratch_model.load_state_dict(state_dict)
        LOGGER.info("Loaded pretrained weights ok!")
        return scratch_model
    