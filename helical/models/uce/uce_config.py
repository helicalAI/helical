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

        self.batch_size = batch_size,
        self.pad_length = pad_length,
        self.pad_token_idx = pad_token_idx,
        self.chrom_token_left_idx = chrom_token_left_idx,
        self.chrom_token_right_idx = chrom_token_right_idx,
        self.cls_token_idx = cls_token_idx,
        self.CHROM_TOKEN_OFFSET = CHROM_TOKEN_OFFSET,
        self.sample_size = sample_size,
        self.CXG = CXG,
        self.n_layers = n_layers,
        self.output_dim = output_dim,
        self.d_hid = d_hid,
        self.token_dim = token_dim,
        self.multi_gpu = multi_gpu,
        self.accelerator = accelerator