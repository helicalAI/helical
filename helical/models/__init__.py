# mamba_ssm (used by Caduceus and HelixmRNA) still imports the greedy/sample decoder
# output names that transformers deleted; re-alias them here before any model loads.
import transformers.generation as _tg

if not hasattr(_tg, "GreedySearchDecoderOnlyOutput"):
    from transformers.generation import GenerateDecoderOnlyOutput as _GDO

    _tg.GreedySearchDecoderOnlyOutput = _GDO
    _tg.SampleDecoderOnlyOutput = _GDO
    del _GDO
del _tg

from .fine_tune.fine_tuning_heads import ClassificationHead, RegressionHead
