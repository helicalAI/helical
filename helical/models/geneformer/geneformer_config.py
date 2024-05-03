from typing import Optional
class GeneformerConfig():
    def __init__(
            self, 
            model_name: str = "geneformer-12L-30M",
            batch_size: int = 5,
            emb_layer: int = -1,
            emb_mode: str = "cell",
            device: str = "cpu",
            accelerator: Optional[dict] = {"cpu": True}
            ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.emb_layer = emb_layer
        self.emb_mode = emb_mode
        self.device = device
        self.accelerator = accelerator


