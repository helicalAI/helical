"""Re-export shim.

This module used to vendor a full copy of PyTorch's own ``TransformerEncoder``/
``TransformerEncoderLayer`` (plus an ``output_attentions`` flag threaded through
``forward``). That implementation still wrapped stock, unmodified
``torch.nn.MultiheadAttention`` as ``self_attn``, which never calls its own
``out_proj`` as a module (it reads ``out_proj.weight``/``out_proj.bias`` directly
into a functional call), so a LoRA adapter targeting ``out_proj`` never received
gradient (helicalAI/bio-agent#1015). The fixed implementation -- shared with
Nicheformer, which has the identical issue -- now lives in
``helical.utils.transformer_encoder``. Re-exported here for backward
compatibility with anything importing from this path directly.
"""

from helical.utils.transformer_encoder import (  # noqa: F401
    TransformerEncoder,
    TransformerEncoderLayer,
)
