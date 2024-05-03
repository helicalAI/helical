from typing import Optional

class UCEConfig():
    def __init__(self,
                 batch_size: int = 5,
                 pad_length: int = 1536,
                 pad_token_idx: int = 0,
                 chrom_token_left_idx: int = 1,
                 chrom_token_right_idx: int = 2,
                 cls_token_idx: int = 3,
                 CHROM_TOKEN_OFFSET: int = 143574,
                 sample_size: int = 1024,
                 CXG: bool = True,
                 n_layers: int = 4,
                 output_dim: int = 1280,
                 d_hid: int = 5120,
                 token_dim: int = 5120,
                 multi_gpu: bool = False,
                 accelerator: Optional[dict] = {"cpu": True}
                ):
        self.config = {
            "batch_size": batch_size,
            "pad_length": pad_length,
            "pad_token_idx": pad_token_idx,
            "chrom_token_left_idx": chrom_token_left_idx,
            "chrom_token_right_idx": chrom_token_right_idx,
            "cls_token_idx": cls_token_idx,
            "CHROM_TOKEN_OFFSET": CHROM_TOKEN_OFFSET,
            "sample_size": sample_size,
            "CXG": CXG,
            "n_layers": n_layers,
            "output_dim": output_dim,
            "d_hid": d_hid,
            "token_dim": token_dim,
            "multi_gpu": multi_gpu,
            "accelerator": accelerator,
        }