import os
import logging

# Compatibility shim for mamba_ssm with transformers >= 4.53.0.
# mamba_ssm.utils.generation still imports the removed GreedySearchDecoderOnlyOutput
# and SampleDecoderOnlyOutput names; patch them back as aliases before any model
# (Caduceus, HelixmRNA) triggers the mamba_ssm import.
import transformers.generation as _tg

if not hasattr(_tg, "GreedySearchDecoderOnlyOutput"):
    from transformers.generation import GenerateDecoderOnlyOutput as _GDO

    _tg.GreedySearchDecoderOnlyOutput = _GDO
    _tg.SampleDecoderOnlyOutput = _GDO
    del _GDO
del _tg

logging.captureWarnings(True)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(levelname)s:%(name)s:%(message)s")
handler.setFormatter(formatter)

logger.addHandler(handler)
logger.propagate = False  # Prevent propagation to root

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
