from helical.models.scgpt.scgpt_utils import load_model
from helical.models.scgpt.scgpt_config import scGPTConfig
from helical.models.scgpt.model_dir import TransformerModel

expected_model_str = """\
TransformerModel(\n\
  (encoder): GeneEncoder(\n\
    (embedding): Embedding(60697, 512, padding_idx=60694)\n\
    (enc_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)\n\
  )\n\
  (value_encoder): ContinuousValueEncoder(\n\
    (dropout): Dropout(p=0.2, inplace=False)\n\
    (linear1): Linear(in_features=1, out_features=512, bias=True)\n\
    (activation): ReLU()\n\
    (linear2): Linear(in_features=512, out_features=512, bias=True)\n\
    (norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)\n\
  )\n\
  (transformer_encoder): TransformerEncoder(\n\
    (layers): ModuleList(\n\
      (0-11): 12 x TransformerEncoderLayer(\n\
        (self_attn): MultiheadAttention(\n\
          (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)\n\
        )\n\
        (linear1): Linear(in_features=512, out_features=512, bias=True)\n\
        (dropout): Dropout(p=0.2, inplace=False)\n\
        (linear2): Linear(in_features=512, out_features=512, bias=True)\n\
        (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)\n\
        (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)\n\
        (dropout1): Dropout(p=0.2, inplace=False)\n\
        (dropout2): Dropout(p=0.2, inplace=False)\n\
      )\n\
    )\n\
  )\n\
  (decoder): ExprDecoder(\n\
    (fc): Sequential(\n\
      (0): Linear(in_features=512, out_features=512, bias=True)\n\
      (1): LeakyReLU(negative_slope=0.01)\n\
      (2): Linear(in_features=512, out_features=512, bias=True)\n\
      (3): LeakyReLU(negative_slope=0.01)\n\
      (4): Linear(in_features=512, out_features=1, bias=True)\n\
    )\n\
  )\n\
  (cls_decoder): ClsDecoder(\n\
    (_decoder): ModuleList(\n\
      (0): Linear(in_features=512, out_features=512, bias=True)\n\
      (1): ReLU()\n\
      (2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)\n\
      (3): Linear(in_features=512, out_features=512, bias=True)\n\
      (4): ReLU()\n\
      (5): LayerNorm((512,), eps=1e-05, elementwise_affine=True)\n\
    )\n\
    (out_layer): Linear(in_features=512, out_features=1, bias=True)\n\
  )\n\
  (mvc_decoder): MVCDecoder(\n\
    (gene2query): Linear(in_features=512, out_features=512, bias=True)\n\
    (query_activation): Sigmoid()\n\
    (W): Linear(in_features=512, out_features=512, bias=False)\n\
  )\n\
  (sim): Similarity(\n\
    (cos): CosineSimilarity()\n\
  )\n\
  (creterion_cce): CrossEntropyLoss()\n\
)"""


def test_load_model():

    configurer = scGPTConfig()
    model, vocab = load_model(configurer.config)

    assert vocab["<pad>"] == 60694
    assert vocab["<cls>"] == 60695
    assert vocab["<eoc>"] == 60696
    assert len(vocab) == 60697

    assert type(model) == TransformerModel
    assert str(model) == expected_model_str
