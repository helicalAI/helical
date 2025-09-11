import warnings

warnings.filterwarnings("ignore")
import math
import torch.nn.functional as F
import torch
from torch import nn, Tensor
from .flash_transformer import FlashTransformerEncoderLayer
from .flash_transformer import FlashTransformerEncoder
import logging

def get_embedding_cfg(cfg):
    return cfg["embeddings"][cfg["embeddings"]["current"]]

def get_dataset_cfg(cfg):
    return cfg["dataset"][cfg["dataset"]["current"]]
    

class SkipBlock(nn.Module):
    def __init__(self, in_features):
        """
        Given input X of size in_features
        - out = layernorm(x + MLP(MLP(X))

        """
        super().__init__()
        self.dim = in_features
        self.intermediate_dense = nn.Linear(in_features, in_features * 2, bias=True)
        self.dense = nn.Linear(in_features * 2, in_features, bias=True)
        self.activation = nn.ReLU()
        self.layer_norm = nn.LayerNorm(in_features)

    def forward(self, x):
        residual = x
        x = self.intermediate_dense(x)
        x = self.activation(x)
        x = self.dense(x)
        x = self.layer_norm(x + residual)
        return x


def nanstd(x):
    return torch.sqrt(
        torch.nanmean(torch.pow(x - torch.nanmean(x, dim=-1).unsqueeze(-1), 2), dim=-1)
    )


class StateEmbeddingInferenceOnly(nn.Module):
    def __init__(
        self,
        token_dim: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        output_dim: int,
        compiled: bool = False,
        # emb_cnt=145469,
        # emb_size=5120,
        cfg=None,
    ):
        super().__init__()
        self.cfg = cfg
        self.compiled = compiled
        self.model_type = "Transformer"
        self.cls_token = nn.Parameter(torch.randn(1, token_dim))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.d_model = d_model
        # Encodes Tokens
        self.encoder = nn.Sequential(
            nn.Linear(token_dim, d_model, bias=True),
            nn.LayerNorm(d_model),  # Moved before activation
            nn.SiLU(),  # Changed to SiLU
        )

        # Create a list of FlashTransformerEncoderLayer instances
        layers = [
            FlashTransformerEncoderLayer(d_model, nhead, d_hid, dropout=0)
            for _ in range(nlayers)
        ]
        self.transformer_encoder = FlashTransformerEncoder(layers)

        if compiled:
            self.transformer_encoder = torch.compile(self.transformer_encoder)

        self.d_model = d_model

        self.decoder = nn.Sequential(
            SkipBlock(d_model),
            nn.Linear(d_model, output_dim, bias=True),
        )

        if compiled:
            self.decoder = torch.compile(self.decoder)

        self.z_dim_rd = 1 if self.cfg.model.rda else 0
        self.z_dim_ds = 10 if self.cfg.model.get("dataset_correction", False) else 0
        self.z_dim = self.z_dim_rd + self.z_dim_ds

        self.binary_decoder = nn.Sequential(
            SkipBlock(output_dim + d_model + self.z_dim),
            SkipBlock(output_dim + d_model + self.z_dim),
            nn.Linear(output_dim + d_model + self.z_dim, 1, bias=True),
        )

        if self.cfg.model.counts:
            self.bin_encoder = nn.Embedding(10, d_model)
            self.count_encoder = nn.Sequential(
                nn.Linear(1, 512, bias=True),
                nn.LeakyReLU(),
                nn.Linear(512, 10),
            )

        if compiled:
            self.binary_decoder = torch.compile(self.binary_decoder)

        # Encodes Tokens for Decoder
        self.gene_embedding_layer = self.encoder  # reuse this layer

        if compiled:
            self.gene_embedding_layer = torch.compile(self.gene_embedding_layer)

        self.pe_embedding = None  # TODO: make this cleaner for the type checker, right now it gets set externally after model init
        self.true_top_genes = None
        self.protein_embeds = None

        if getattr(self.cfg.model, "dataset_correction", False):
            self.dataset_token = nn.Parameter(torch.randn(1, token_dim))
            self.dataset_embedder = nn.Linear(output_dim, self.z_dim_ds)

            # Assume self.cfg.model.num_datasets is set to the number of unique datasets.
            num_dataset = get_dataset_cfg(self.cfg).num_datasets
            self.dataset_encoder = nn.Sequential(
                nn.Linear(output_dim, d_model),
                nn.SiLU(),
                nn.LayerNorm(d_model),
                nn.Dropout(0.1),
                nn.Linear(d_model, num_dataset),
            )

        else:
            self.dataset_token = None

    def _compute_embedding_for_batch(self, batch):
        batch_sentences = batch[0].to(self.device)
        X = batch[1].to(self.device)
        Y = batch[2]
        batch_weights = batch[4]
        mask = batch[5]
        mask = mask.to(torch.bool).to(self.device)
        batch_sentences_counts = batch[7]
        if batch_sentences_counts is not None:
            batch_sentences_counts = batch_sentences_counts.to(self.device)
        dataset_nums = batch[8]
        if dataset_nums is not None:
            dataset_nums = dataset_nums.to(self.device)

        # convert the cell sentence and task sentence into embeddings
        batch_sentences = self.pe_embedding(batch_sentences)
        X = self.pe_embedding(X)

        # Normalize token outputs now
        batch_sentences = nn.functional.normalize(batch_sentences, dim=2)

        # Add a learnable CLS token to the beginning of the sentence
        batch_sentences[:, 0, :] = self.cls_token.expand(batch_sentences.size(0), -1)

        # Optionally add a learnable dataset token to the end of the sentence
        if self.dataset_token is not None:
            dataset_token = self.dataset_token.expand(
                batch_sentences.size(0), -1
            ).unsqueeze(1).to(self.device)

            batch_sentences = torch.cat((batch_sentences, dataset_token), dim=1)
            # concatenate a False to the mask on dim 1

            mask = torch.cat(
                (mask, torch.zeros(mask.size(0), 1, device=self.device).bool()), dim=1
            )
        
        # mask out the genes embeddings that appear in the task sentence
        _, embedding, dataset_emb = self.forward(
            batch_sentences,
            mask=mask,
            counts=batch_sentences_counts,
            dataset_nums=dataset_nums,
        )

        X = self.gene_embedding_layer(X)
        return X, Y, batch_weights, embedding, dataset_emb

    def get_gene_embedding(self, genes):
        if self.protein_embeds is None:
            self.protein_embeds = torch.load(
                get_embedding_cfg(self.cfg).all_embeddings, weights_only=False
            )

        protein_embeds = [
            (
                self.protein_embeds[x]
                if x in self.protein_embeds
                else torch.zeros(get_embedding_cfg(self.cfg).size)
            )
            for x in genes
        ]
        protein_embeds = torch.stack(protein_embeds).to(self.device)
        if protein_embeds.sum() == 0:
            raise ValueError("No gene embeddings found")

        return self.gene_embedding_layer(protein_embeds)

    @staticmethod
    def resize_batch(
        cell_embeds, task_embeds, task_counts=None, sampled_rda=None, ds_emb=None
    ):
        A = task_embeds.unsqueeze(0).repeat(cell_embeds.size(0), 1, 1)
        B = cell_embeds.unsqueeze(1).repeat(1, task_embeds.size(0), 1)
        if sampled_rda is not None:
            # computes mu and std dev from Y
            reshaped_counts = sampled_rda.unsqueeze(1)
            reshaped_counts = reshaped_counts.repeat(1, A.shape[1], 1)
            combine = torch.cat((A, B, reshaped_counts), dim=2)
        elif task_counts is not None:
            reshaped_counts = task_counts.unsqueeze(1).unsqueeze(2)
            reshaped_counts = reshaped_counts.repeat(1, A.shape[1], 1)

            # Concatenate all three tensors along the third dimension
            combine = torch.cat((A, B, reshaped_counts), dim=2)
        else:
            # Original behavior if total_counts is None
            combine = torch.cat((A, B), dim=2)

        if ds_emb is not None:
            # ds_emb is a tensor of shape (batch_size, 10). concatenate it to the combine tensor
            ds_emb = ds_emb.unsqueeze(1).repeat(1, A.shape[1], 1)
            combine = torch.cat((combine, ds_emb), dim=2)

        return combine

    def forward(self, src: Tensor, mask: Tensor, counts=None, dataset_nums=None):
        """
        Args:
            src: Tensor, shape [batch_size, seq_len, ntoken]
        Returns:
            output Tensor of shape [batch_size, seq_len, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        if counts is not None:
            # scFoundation-style soft binning for counts
            counts = counts.unsqueeze(-1)  # now B x H x 1

            # Step 1: Transform count values into bin distribution
            bin_weights = self.count_encoder(counts)  # B x H x 10
            bin_weights = F.softmax(
                bin_weights, dim=-1
            )  # Convert to probabilities over bins
            # Step 2: Get bin embeddings
            bin_indices = torch.arange(10, device=self.device)  # 10 bins
            bin_embeddings = self.bin_encoder(bin_indices)  # 10 x d_model

            # Step 3: Compute weighted sum of bin embeddings
            count_emb = torch.matmul(bin_weights, bin_embeddings)
            if self.dataset_token is not None:
                # append B x 1 x d_model to count_emb of all zeros
                dataset_count_emb = torch.zeros(
                    count_emb.size(0), 1, count_emb.size(2), device=self.device
                )
                count_emb = torch.cat(
                    (count_emb, dataset_count_emb), dim=1
                )  # B x H x d_model
            # Add count embeddings to token embeddings
            src = (
                src + count_emb
            )  # should both be B x H x self.d_model, or B x H + 1 x self.d_model if dataset correction

        output = self.transformer_encoder(src, src_key_padding_mask=None)
        gene_output = self.decoder(output)  # batch x seq_len x 128
        # In the new format, the cls token, which is at the 0 index mark, is the output.
        embedding = gene_output[:, 0, :]  # select only the CLS token.
        embedding = nn.functional.normalize(embedding, dim=1)  # Normalize.

        # we must be in train mode to use dataset correction
        dataset_emb = None
        if self.dataset_token is not None:
            dataset_emb = gene_output[:, -1, :]

        return gene_output, embedding, dataset_emb


# if __name__ == "__main__":
#     # test model.safetensors and checkpoint.pt
#     import os
#     from omegaconf import OmegaConf

#     def load_esm2_embeddings(cfg):
#         # Load in ESM2 embeddings and special tokens
#         all_pe = torch.load(get_embedding_cfg(cfg).all_embeddings, weights_only=False)
#         if isinstance(all_pe, dict):
#             all_pe = torch.vstack(list(all_pe.values()))

#         all_pe = all_pe.cuda()
#         return all_pe


#     def get_precision_config(device_type="cuda"):
#         """
#         Single source of truth for precision configuration.

#         Args:
#             device_type: Device type ('cuda' or 'cpu')

#         Returns:
#             torch.dtype: The precision to use for autocast and model operations.
#                         Returns torch.bfloat16 for CUDA, torch.float32 for CPU.
#         """
#         if device_type == "cuda":
#             return torch.bfloat16
#         else:
#             return torch.float32

#     model_conf = OmegaConf.load(os.path.join("/home/rasched/.cache/helical/models/state/state_embed/config.yaml"))

#     model = StateEmbeddingInferenceOnly(
#         token_dim=model_conf.tokenizer.token_dim,  # Changed from model_conf.model.token_dim
#         d_model=model_conf.model.emsize,  # Changed from model_conf.model.d_model
#         nhead=model_conf.model.nhead,
#         d_hid=model_conf.model.d_hid,
#         nlayers=model_conf.model.nlayers,
#         output_dim=model_conf.model.output_dim,
#         compiled=model_conf.experiment.compiled,
#         cfg=model_conf,
#     )

#     print("number of free parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

#     print("token_dim: ", model_conf.tokenizer.token_dim)
#     print("d_model: ", model_conf.model.emsize)
#     print("nhead: ", model_conf.model.nhead)
#     print("d_hid: ", model_conf.model.d_hid)
#     print("nlayers: ", model_conf.model.nlayers)
#     print("output_dim: ", model_conf.model.output_dim)
#     print("compiled: ", model_conf.experiment.compiled)

#     # Load the weights
#     loaded_weights = torch.load("/home/rasched/final_helical_with_state/helical/helical/models/state/model_dir/embed_utils/nn/embed_model_epoch16_weights.pt", weights_only=True)

#     # missing_keys are keys that are in the model but NOT in the loaded weights.
#     missing_keys, unexpected_keys = model.load_state_dict(loaded_weights, strict=False)
#     print(f"Missing keys: {missing_keys}")

#     protein_embeds = (
#                 torch.load(os.path.join("/home/rasched/.cache/helical/models/state/state_embed/protein_embeddings.pt"), weights_only=False, map_location="cpu")
#                 if os.path.exists(os.path.join("/home/rasched/.cache/helical/models/state/state_embed/protein_embeddings.pt"))
#                 else None
#             )

#     # Convert model to appropriate precision for faster inference
#     device_type = "cuda" if torch.cuda.is_available() else "cpu"
#     precision = get_precision_config(device_type=device_type)
#     model = model.to(precision)

#     all_pe = protein_embeds or load_esm2_embeddings(model_conf)
#     if isinstance(all_pe, dict):
#         all_pe = torch.vstack(list(all_pe.values()))

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     model.pe_embedding = nn.Embedding.from_pretrained(all_pe)
#     model.pe_embedding.to(device, dtype=precision)
#     model.binary_decoder.requires_grad = False
#     model.eval()

#     if protein_embeds is None:
#         protein_embeds = torch.load(
#             get_embedding_cfg(model_conf).all_embeddings, weights_only=False
#         )
    
#     print("Done loading model")